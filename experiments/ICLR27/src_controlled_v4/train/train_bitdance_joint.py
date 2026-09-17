"""Reliable shared training entry for the controlled BFM/BLD comparison.

Key differences from the original train_sampler_online.py:
1. The binary autoencoder is frozen and kept in eval mode.
2. The non-DDP sampler is always defined and is used as the EMA source.
3. Gradient accumulation, LR scheduling, EMA updates, logging, and checkpoint
   intervals are counted in optimizer updates rather than micro-batches.
4. Loss statistics are actually accumulated; the original script computed the
   mean of an empty array.
5. Resume loading occurs before DDP wrapping and optimizer state restoration is
   aligned with the loaded model.
6. A warning is emitted when aux == 0, because in that case retraining uses the
   same X0 BCE objective as the legacy model and does not train the corrected
   reverse-posterior objective.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from hparams import get_sampler_hparams
from models.binaryae import BinaryAutoEncoder
from models.binarylatent_flow_expectation_consistent_retrain import (
    BinaryDiffusionFlowDecouple,
)
from models.binarylatent_flow_expectation_consistent_retrain_tminus1 import (
    BinaryDiffusionFlowDecouple as BinaryDiffusionFlowComparison,
)
from models.binarylatent_flow_endpoint_cavity import (
    BinaryDiffusionFlowEndpointCavity,
)
from models.binarylatent_flow_selective_corrector import (
    BinaryDiffusionFlowSelectiveCorrector,
)
from models.binarylatent_flow_cs_decomposed import (
    BinaryDiffusionFlowCSDecomposed,
)
from models.binarylatent_flow_bitdance_joint import (
    BinaryDiffusionFlowBitDanceJoint,
)
from models.binarylatent_flow_bitdance_joint_64d_inner import (
    BinaryDiffusionFlowBitDanceJoint as BinaryDiffusionFlowBitDanceJoint64DInner,
)
from models.binarylatent_flow_bitdance_joint_src_v4 import (
    BinaryDiffusionFlowBitDanceJointSRCV4,
)
from models.binarylatent_flow_controlled_src_v4 import (
    BinaryDiffusionFlowTimeAligned,
    BinaryDiffusionFlowOneStepLossAblation,
    BinaryDiffusionFlowMultiNFESRC,
    BinaryDiffusionFlowV6SensitivityAnchor,
)
from models.transformer import TransformerBD
from models.transformer_bitdance_bce import TransformerBDBitDanceBCE
from utils.reliable_data_utils import get_data_loaders
from utils.log_utils import (
    MovingAverage,
    config_log,
    load_model,
    load_stats,
    log,
    log_stats,
    save_images,
    save_model,
    save_stats,
    start_training_log,
)
from utils.lr_sched import adjust_lr, lr_scheduler
from utils.sampler_utils import (
    get_online_samples,
    get_online_samples_guidance,
    get_sampler,
    retrieve_autoencoder_components_state_dicts,
)
from utils.train_utils import EMA, NativeScalerWithGradNormCount
import misc


import random


def seed_experiment(H):
    seed = int(os.environ.get("EXPERIMENT_SEED", "20260821"))
    rank = dist.get_rank() if H.distributed and dist.is_initialized() else 0
    # Same model seed on every rank; DDP expects identical initialization.
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def is_main_process(H) -> bool:
    return (not H.distributed) or dist.get_rank() == 0


def current_device(H) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA.")
    configured_gpu = getattr(H, "gpu", None)
    gpu = int(0 if configured_gpu is None else configured_gpu)
    return torch.device(f"cuda:{gpu}")


def unpack_batch(data: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(data, (tuple, list)):
        image = data[0]
        label = data[1] if len(data) > 1 else None
        return image, label
    return data, None


def freeze_autoencoder(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_endpoint_transformerbd(
    denoiser: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    """Strictly extract the endpoint Transformer from a BFM checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(
            f"endpoint checkpoint must contain a state dict, got {type(payload)}"
        )

    prefix = "_denoise_fn."
    endpoint_state = {
        key[len(prefix) :]: value
        for key, value in payload.items()
        if key.startswith(prefix)
    }
    if not endpoint_state:
        # Also accept a raw TransformerBD state dict for controlled experiments.
        endpoint_state = payload
    denoiser.load_state_dict(endpoint_state, strict=True)



def audit_frozen_endpoint(
    denoiser: torch.nn.Module,
    checkpoint_path: str,
    output_path: Path,
) -> dict:
    """Compare every frozen endpoint tensor against its source checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    prefix = "_denoise_fn."
    reference = {
        key[len(prefix) :]: value
        for key, value in payload.items()
        if key.startswith(prefix)
    }
    if not reference:
        reference = payload
    current = denoiser.state_dict()
    if set(current) != set(reference):
        raise RuntimeError("Frozen endpoint audit key mismatch")
    per_tensor = {}
    global_max = 0.0
    for name, value in current.items():
        difference = float(
            (value.detach().cpu().float() - reference[name].detach().cpu().float())
            .abs().max().item()
        )
        per_tensor[name] = difference
        global_max = max(global_max, difference)
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "strict_key_match": True,
        "tensor_count": len(per_tensor),
        "global_max_abs_diff": global_max,
        "per_tensor_max_abs_diff": per_tensor,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if global_max != 0.0:
        raise RuntimeError(f"Frozen endpoint changed: max_abs_diff={global_max}")
    return result

def load_training_stats(H):
    losses = np.array([], dtype=np.float64)
    mean_losses = np.array([], dtype=np.float64)
    val_losses = np.array([], dtype=np.float64)
    elbo = np.array([], dtype=np.float64)
    val_elbos = np.array([], dtype=np.float64)

    if int(H.load_step) <= 0:
        return losses, mean_losses, val_losses, elbo, val_elbos

    try:
        stats = load_stats(H, H.load_step)
    except Exception as exc:
        log(f"No usable stats file found for step {H.load_step}: {exc}")
        return losses, mean_losses, val_losses, elbo, val_elbos

    losses = np.asarray(stats.get("losses", losses)).reshape(-1)
    mean_losses = np.asarray(stats.get("mean_losses", mean_losses)).reshape(-1)
    val_losses = np.asarray(stats.get("val_losses", val_losses)).reshape(-1)
    elbo = np.asarray(stats.get("elbo", elbo)).reshape(-1)
    val_elbos = np.asarray(stats.get("val_elbos", val_elbos)).reshape(-1)
    return losses, mean_losses, val_losses, elbo, val_elbos


def save_training_checkpoint(
    H,
    sampler_without_ddp,
    optimizer,
    scaler,
    ema_sampler,
    update_step: int,
    losses,
    mean_losses,
    val_losses,
    elbo,
    val_elbos,
) -> None:
    save_model(sampler_without_ddp, H.sampler, update_step, H.log_dir)
    save_model(optimizer, f"{H.sampler}_optim", update_step, H.log_dir)
    save_model(scaler, f"{H.sampler}_scaler", update_step, H.log_dir)
    if H.ema:
        save_model(ema_sampler, f"{H.sampler}_ema", update_step, H.log_dir)

    train_stats = {
        "losses": np.asarray(losses),
        "mean_losses": np.asarray(mean_losses),
        "val_losses": np.asarray(val_losses),
        "elbo": np.asarray(elbo),
        "val_elbos": np.asarray(val_elbos),
        "steps_per_log": H.steps_per_log,
        "steps_per_eval": H.steps_per_eval,
    }
    save_stats(H, train_stats, update_step)


def generate_preview(H, autoencoder, sampler_model, update_step: int, x=None) -> None:
    was_training = sampler_model.training
    sampler_model.eval()
    try:
        with torch.no_grad():
            if H.guidance:
                images = get_online_samples_guidance(H, autoencoder, sampler_model)
            else:
                # Keep the requested 2K cadence without the generic helper's
                # costly 10-temperature x 8-image BitDance preview.
                if x is None and (
                    hasattr(sampler_model, "diff_head")
                    or hasattr(sampler_model, "cavity_config")
                    or hasattr(sampler_model, "selective_config")
                ):
                    preview_latents = sampler_model.sample(
                        b=4, sample_steps=H.sample_steps
                    )
                    images = get_online_samples(
                        H,
                        autoencoder,
                        sampler_model,
                        x=preview_latents,
                    )
                else:
                    images = get_online_samples(H, autoencoder, sampler_model, x=x)
        save_images(
            images,
            "samples",
            update_step,
            H.log_dir,
            H.save_individually,
        )
    finally:
        if was_training:
            sampler_model.train()


def _decode_latents(H, autoencoder, sampler_model, latents):
    return get_online_samples(
        H,
        autoencoder,
        sampler_model,
        x=latents,
    )


def generate_bitdance_diagnostics(
    H,
    autoencoder,
    raw_sampler,
    ema_sampler,
    update_step: int,
    reference_x: torch.Tensor,
) -> None:
    """Eval-only localization of RAW/EMA, endpoint, and trajectory failures.

    This deliberately does not alter the RF objective, inner sampler, or BFM
    bridge. A forked RNG makes diagnostics reproducible without changing the
    subsequent training batches or noise draws.
    """
    models = {"raw": raw_sampler}
    if ema_sampler is not None:
        models["ema"] = ema_sampler

    diagnostic_root = Path(H.log_dir) / "diagnostics" / f"step_{update_step:06d}"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    reference_x = reference_x[:4].detach()
    seed = int(os.environ.get("EXPERIMENT_SEED", "20260821")) + update_step
    device_index = reference_x.device.index
    rng_devices = [] if device_index is None else [device_index]

    for model_name, model in models.items():
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(seed)
                trace = []
                free_latents = model.sample(
                    b=4,
                    sample_steps=H.sample_steps,
                    diagnostics=trace,
                )
                free_images = _decode_latents(
                    H, autoencoder, model, free_latents
                )
                save_images(
                    free_images,
                    f"samples_{model_name}",
                    update_step,
                    H.log_dir,
                    H.save_individually,
                )

                endpoint_summary = {}
                for physical_t in (64, 32, 16, 8, 1):
                    # Identical per-t seeds give RAW/EMA the same outer
                    # corruption and inner Gaussian noise.
                    torch.manual_seed(seed + physical_t)
                    t = torch.full(
                        (reference_x.shape[0],),
                        physical_t,
                        device=reference_x.device,
                        dtype=torch.long,
                    )
                    x_t = torch.bernoulli(model.q_sample(reference_x, t))
                    endpoint = model._sample_joint_endpoint(x_t, t)
                    endpoint_images = _decode_latents(
                        H, autoencoder, model, endpoint
                    )
                    save_images(
                        endpoint_images,
                        f"teacher_endpoint_{model_name}_t{physical_t:02d}",
                        update_step,
                        H.log_dir,
                        H.save_individually,
                    )
                    endpoint_summary[str(physical_t)] = {
                        "x_t_accuracy_to_x0": float(
                            (x_t == reference_x).float().mean()
                        ),
                        "endpoint_accuracy_to_x0": float(
                            (endpoint == reference_x).float().mean()
                        ),
                        "endpoint_hamming_to_x_t": float(
                            (endpoint != x_t).float().mean()
                        ),
                        "x_t_bit_mean": float(x_t.mean()),
                        "endpoint_bit_mean": float(endpoint.mean()),
                    }

                report = {
                    "step": int(update_step),
                    "model": model_name,
                    "core_algorithm_changed": False,
                    "teacher_forced_endpoint": endpoint_summary,
                    "full_chain": trace,
                }
                (diagnostic_root / f"{model_name}.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
        finally:
            if was_training:
                model.train()


def main(H, vis=None):
    del vis
    misc.init_distributed_mode(H)
    seed = seed_experiment(H)
    device = current_device(H)

    # ------------------------------------------------------------------
    # Frozen binary autoencoder
    # ------------------------------------------------------------------
    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["encoder", "quantize", "generator"],
        remove_component_from_key=False,
    )
    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    del ae_state_dict
    bergan = bergan.to(device)
    freeze_autoencoder(bergan)

    # ------------------------------------------------------------------
    # Sampler loading before DDP and optimizer construction
    # ------------------------------------------------------------------
    experiment_variant = os.environ.get(
        "EXPERIMENT_VARIANT", "aligned_bce"
    ).strip().lower()
    cached_x0_path = os.environ.get("BFM_CACHED_X0_PATH", "").strip()
    mask_pretrain_checkpoint = os.environ.get(
        "BFM_MASK_PRETRAIN_CHECKPOINT", ""
    ).strip()
    endpoint_checkpoint = os.environ.get(
        "BFM_ENDPOINT_CHECKPOINT", ""
    ).strip()
    freeze_endpoint = os.environ.get(
        "BFM_FREEZE_ENDPOINT", "0"
    ).strip().lower() in {"1", "true", "yes"}
    if freeze_endpoint and not endpoint_checkpoint:
        raise ValueError(
            "BFM_FREEZE_ENDPOINT=1 requires BFM_ENDPOINT_CHECKPOINT"
        )
    if cached_x0_path:
        if H.p_flip or float(H.focal) >= 0.0 or float(H.aux) != 0.0:
            raise ValueError(
                "Cached direct-X0 contract requires --focal -1, no --p_flip, and --aux 0"
            )
        if H.loss_final != "mean":
            raise ValueError("Cached direct-X0 contract requires --loss_final mean")

    if H.sampler.startswith("flow_"):
        denoiser_class = (
            TransformerBDBitDanceBCE
            if experiment_variant == "bitdance_bce_control"
            else TransformerBD
        )
        denoiser = denoiser_class(H).to(device)
        cavity_denoiser = None

        if experiment_variant in {
            "cached_endpoint_cavity", "cached_selective_corrector"
        }:
            # An independent Transformer prevents cavity gradients from
            # degrading the endpoint predictor that drives the BFM bridge.
            cavity_denoiser = TransformerBD(H).to(device)

        if mask_pretrain_checkpoint:
            from experiments.ICLR27.diagnostics.masked_bit_pretrain_core import (
                load_mask_pretrained_transformerbd,
            )

            metadata = load_mask_pretrained_transformerbd(
                denoiser, mask_pretrain_checkpoint, map_location="cpu"
            )
            if cavity_denoiser is not None:
                load_mask_pretrained_transformerbd(
                    cavity_denoiser,
                    mask_pretrain_checkpoint,
                    map_location="cpu",
                )
            log(
                f"Strict MASK initialization: {mask_pretrain_checkpoint}; "
                f"stage1_step={metadata.get('step', 'unknown')}"
            )

        if experiment_variant == "cached_endpoint_cavity" and endpoint_checkpoint:
            load_endpoint_transformerbd(denoiser, endpoint_checkpoint)
            log(f"Strict endpoint initialization: {endpoint_checkpoint}")
            if freeze_endpoint:
                for parameter in denoiser.parameters():
                    parameter.requires_grad_(False)
                denoiser.eval()
                log("Endpoint Transformer frozen: True; only cavity model is optimized")

        if experiment_variant == "original":
            sampler_without_ddp = BinaryDiffusionFlowDecouple(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "comparison_bfm":
            sampler_without_ddp = BinaryDiffusionFlowComparison(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "bitdance_bce_control":
            # Exact comparison-BFM process/loss/sampler; only denoiser class differs.
            sampler_without_ddp = BinaryDiffusionFlowComparison(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "cached_direct_x0_bce":
            sampler_without_ddp = BinaryDiffusionFlowComparison(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "cached_cs_bfm":
            sampler_without_ddp = BinaryDiffusionFlowCSDecomposed(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "cached_endpoint_cavity":
            sampler_without_ddp = BinaryDiffusionFlowEndpointCavity(
                H,
                denoiser,
                H.codebook_size,
                cavity_denoise_fn=cavity_denoiser,
            ).to(device)
        elif experiment_variant == "cached_selective_corrector":
            sampler_without_ddp = BinaryDiffusionFlowSelectiveCorrector(
                H,
                denoiser,
                H.codebook_size,
                corrector_denoise_fn=cavity_denoiser,
            ).to(device)
        elif experiment_variant == "bitdance_joint":
            sampler_without_ddp = BinaryDiffusionFlowBitDanceJoint(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "bitdance_joint_64d_inner_v2":
            sampler_without_ddp = BinaryDiffusionFlowBitDanceJoint64DInner(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "bitdance_joint_src_v4":
            sampler_without_ddp = BinaryDiffusionFlowBitDanceJointSRCV4(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "aligned_bce":
            sampler_without_ddp = BinaryDiffusionFlowTimeAligned(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant in {"aligned_brier", "aligned_src"}:
            expected_mode = (
                "unweighted"
                if experiment_variant == "aligned_brier"
                else "sensitivity"
            )
            os.environ["BFM_SRC_MODE"] = expected_mode
            sampler_without_ddp = BinaryDiffusionFlowOneStepLossAblation(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "multi_nfe_src_v4":
            sampler_without_ddp = BinaryDiffusionFlowMultiNFESRC(
                H, denoiser, H.codebook_size
            ).to(device)
        elif experiment_variant == "v6_sensitivity_anchor":
            sampler_without_ddp = BinaryDiffusionFlowV6SensitivityAnchor(
                H, denoiser, H.codebook_size
            ).to(device)
        else:
            raise ValueError(
                f"Unknown EXPERIMENT_VARIANT={experiment_variant!r}"
            )
    else:
        sampler_without_ddp = get_sampler(
            H, bergan.quantize.embed.weight
        ).to(device)

    if int(H.load_model_step) > 0:
        sampler_without_ddp = load_model(
            sampler_without_ddp,
            H.sampler,
            H.load_model_step,
            H.load_model_dir,
            device=device,
        ).to(device)

    if int(H.load_step) > 0:
        sampler_without_ddp = load_model(
            sampler_without_ddp,
            H.sampler,
            H.load_step,
            H.load_dir,
            device=device,
            allow_mismatch=H.allow_mismatch,
        ).to(device)

    sampler_without_ddp.train()

    if H.distributed:
        sampler = torch.nn.parallel.DistributedDataParallel(
            sampler_without_ddp,
            device_ids=[H.gpu],
            find_unused_parameters=bool(H.guidance),
        )
    else:
        sampler = sampler_without_ddp

    # EMA must have the same unwrapped module structure as its source.
    ema = None
    ema_sampler = None
    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler_without_ddp).to(device)
        ema_sampler.eval()
        for parameter in ema_sampler.parameters():
            parameter.requires_grad_(False)

        if int(H.load_step) > 0:
            try:
                ema_sampler = load_model(
                    ema_sampler,
                    f"{H.sampler}_ema",
                    H.load_step,
                    H.load_dir,
                    device=device,
                    allow_mismatch=H.allow_mismatch,
                ).to(device)
                ema_sampler.eval()
            except Exception as exc:
                log(f"EMA checkpoint could not be loaded; using model copy: {exc}")
                ema_sampler = copy.deepcopy(sampler_without_ddp).to(device).eval()

    # Preserve the original optimizer betas for a controlled method ablation.
    # The manuscript currently reports beta2=0.99, while this script historically
    # used beta2=0.95; resolve that discrepancy separately rather than changing it
    # silently in the same experiment.
    configured_beta1 = getattr(H, "adam_beta1", None)
    configured_beta2 = getattr(H, "adam_beta2", None)
    beta1 = float(0.9 if configured_beta1 is None else configured_beta1)
    beta2 = float(0.95 if configured_beta2 is None else configured_beta2)
    named_trainable_parameters = [
        (name, parameter)
        for name, parameter in sampler_without_ddp.named_parameters()
        if parameter.requires_grad
    ]
    if experiment_variant == "cached_endpoint_cavity" and freeze_endpoint:
        unexpected = [
            name for name, _ in named_trainable_parameters
            if not name.startswith("_cavity_denoise_fn.")
        ]
        if unexpected:
            raise RuntimeError(
                f"Frozen endpoint run has non-cavity trainable parameters: {unexpected[:20]}"
            )
    optimizer_parameter_names = [name for name, _ in named_trainable_parameters]
    optimizer_parameters = [parameter for _, parameter in named_trainable_parameters]
    if not optimizer_parameters:
        raise RuntimeError("No trainable parameters available for optimizer")
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=H.lr,
        weight_decay=H.weight_decay,
        betas=(beta1, beta2),
        eps=H.optim_eps,
    )

    if is_main_process(H) and experiment_variant == "cached_endpoint_cavity":
        parameter_audit = {
            "endpoint_checkpoint": str(Path(endpoint_checkpoint).resolve()),
            "endpoint_checkpoint_strict_load": True,
            "cavity_mask_pretrain_checkpoint": str(
                Path(mask_pretrain_checkpoint).resolve()
            ),
            "cavity_mask_pretrain_strict_load": True,
            "endpoint_parameter_count": sum(
                parameter.numel()
                for parameter in sampler_without_ddp._denoise_fn.parameters()
            ),
            "cavity_parameter_count": sum(
                parameter.numel()
                for parameter in sampler_without_ddp._cavity_denoise_fn.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in optimizer_parameters
            ),
            "optimizer_parameter_names": optimizer_parameter_names,
        }
        audit_path = Path(H.log_dir) / "parameter_audit_before.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(parameter_audit, indent=2) + "\n", encoding="utf-8"
        )
        log(f"Endpoint params: {parameter_audit['endpoint_parameter_count']}")
        log(f"Cavity params: {parameter_audit['cavity_parameter_count']}")
        log(f"Trainable params: {parameter_audit['trainable_parameter_count']}")
        log(f"Optimizer parameter names: {optimizer_parameter_names}")

    if int(H.load_step) > 0 and (not H.allow_mismatch) and H.load_optim:
        optimizer = load_model(
            optimizer,
            f"{H.sampler}_optim",
            H.load_step,
            H.load_dir,
            device=device,
            allow_mismatch=False,
        )
        for group in optimizer.param_groups:
            group["lr"] = H.lr

    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)
    if (
        int(H.load_step) > 0
        and not H.reset_step
        and not H.reset_scaler
    ):
        try:
            scaler = load_model(
                scaler,
                f"{H.sampler}_scaler",
                H.load_step,
                H.load_dir,
                device=device,
                allow_mismatch=False,
            )
        except Exception as exc:
            log(f"Scaler state could not be loaded: {exc}")

    if H.reset_step:
        completed_updates = 0
        H.load_step = 0
    else:
        completed_updates = max(0, int(H.load_step))

    losses, mean_losses, val_losses, elbo, val_elbos = load_training_stats(H)
    loss_ma = MovingAverage(100)

    cached_eval_x0 = None
    cached_eval_module = None
    if cached_x0_path:
        if experiment_variant == "cached_endpoint_cavity":
            from experiments.ICLR27.diagnostics.full64_mask_pretrain_endpoint_cavity import (
                full64_cavity_eval as cached_eval_module,
            )
        elif experiment_variant == "cached_selective_corrector":
            from experiments.ICLR27.diagnostics.full64_mask_pretrain_selective_corrector import (
                full64_selective_corrector_eval as cached_eval_module,
            )
        else:
            from experiments.ICLR27.diagnostics.full64_mask_pretrain_direct_x0_bce import (
                full64_cached_eval as cached_eval_module,
            )
        cached_dataset = cached_eval_module.PackedX0Dataset(cached_x0_path)
        train_loader = cached_eval_module.make_train_loader(
            cached_dataset,
            H.batch_size,
            dist.get_rank() if H.distributed else 0,
            dist.get_world_size() if H.distributed else 1,
            seed,
            num_workers=4,
        )
        if is_main_process(H):
            cached_eval_count = int(
                os.environ.get("BFM_FIXED_EVAL_IMAGES", "300")
            )
            cached_eval_x0 = cached_eval_module.load_eval_x0(
                cached_dataset, cached_eval_count
            )
    else:
        train_loader, _ = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            get_val_dataloader=False,
            custom_dataset_path=H.path_to_data,
            num_workers=4,
            distributed=H.distributed,
            random=True,
            args=H,
        )

    total_updates = int(H.train_steps)
    update_freq = max(1, int(H.update_freq))
    warmup_updates = int(H.warmup_iters)
    lr_sched = lr_scheduler(
        base_value=H.lr,
        final_value=1e-6,
        iters=total_updates + 1,
        warmup_steps=warmup_updates,
        start_warmup_value=1e-6,
        lr_type="constant",
    )

    total_parameters = sum(p.numel() for p in sampler.parameters())
    trainable_parameters = sum(
        p.numel() for p in sampler.parameters() if p.requires_grad
    )
    log(f"Sampler params total: {total_parameters / 1e6:.2f}M")
    log(f"Sampler params trainable: {trainable_parameters / 1e6:.2f}M")
    log(f"Controlled experiment variant: {experiment_variant}")
    log(f"Experiment seed: {seed}")
    if cached_x0_path:
        log(f"Training data: cached deterministic clean X0 only: {cached_x0_path}")
        log("RGB dataset instantiated: False; BinaryAE encoder called in loop: False")
        log("Physical timestep: uniform integer t in [1,64]")
        log("Forward corruption: exact BinaryDiffusionFlowComparison.q_sample")
        log("Network timestep: physical t-1")
        log("Prediction target: direct clean X0")
        if experiment_variant == "multi_nfe_src_v4":
            log(
                "Loss: unweighted plain BCEWithLogits + "
                f"{sampler_without_ddp.v4_lambda:g}*W_64(t)*Brier"
            )
            log("p_flip=False; focal=False; class_balance=False; aux=0; sensitivity=V4")
        elif experiment_variant == "cached_endpoint_cavity":
            if sampler_without_ddp.endpoint_frozen:
                log(
                    "Optimized loss: context-only cavity BCE; endpoint forward/loss "
                    "skipped because the baseline endpoint is frozen"
                )
            else:
                log(
                    "Optimized loss: independent direct-X0 endpoint BCE over all bits "
                    "+ context-only cavity BCE"
                )
            log(f"Endpoint+cavity config: {sampler_without_ddp.cavity_config}")
            log(
                "Cavity time support: every physical t in [1,T]; "
                "no t=28 threshold or late-time-only curriculum"
            )
        elif experiment_variant == "cached_selective_corrector":
            log(
                "Loss: plain direct-X0 endpoint BCE + candidate-only cavity BCE "
                "+ balanced GT-supervised counterfactual reverse-advantage gate BCE"
            )
            log("p_flip=False; focal=False; class_balance=False; aux=0")
            log(f"Selective corrector config: {sampler_without_ddp.selective_config}")
            log(
                "GT is train-only supervision; sampling has no GT, no random bit "
                "block and no random refresh."
            )
        else:
            log("Loss: unweighted plain BCEWithLogits over all bits")
            log("p_flip=False; focal=False; class_balance=False; aux=0; sensitivity=False")
        log(
            "Initialization: "
            + ("10% MASK-pretrained" if mask_pretrain_checkpoint else "random")
        )
        if H.distributed:
            dist.barrier()
        if is_main_process(H):
            eval_model = ema_sampler if H.ema else sampler_without_ddp
            initial_metrics = cached_eval_module.record(
                H.log_dir, eval_model, cached_eval_x0, completed_updates,
                H.batch_size, seed, device,
            )
            log(
                f"Fixed-t EMA diagnostic step {completed_updates}: "
                f"{json.dumps(initial_metrics)}"
            )
        if H.distributed:
            dist.barrier()
    if experiment_variant == "original":
        log("Time convention: train=t, sample=t; loss=base only")
    elif experiment_variant == "bitdance_bce_control":
        log(
            "Controlled backbone/head ablation: exact comparison BFM t-1, p_flip BCE, "
            "analytical reverse sampler; only TransformerBD endpoint predictor replaced."
        )
        log(
            f"BitDance BCE predictor config: "
            f"{getattr(sampler_without_ddp._denoise_fn, 'bitdance_bce_config', {})}"
        )
    elif experiment_variant in {"bitdance_joint", "bitdance_joint_64d_inner_v2"}:
        log(
            "Time convention: outer BFM train/sample=t-1; "
            "endpoint predictor=BitDance joint binary DiffHead; loss=BitDance RF velocity-MSE"
        )
        log(f"BitDance joint config: {getattr(sampler_without_ddp, 'bitdance_config', {})}")
    elif experiment_variant == "bitdance_joint_src_v4":
        log(
            "Time convention: outer BFM train/sample=t-1; endpoint predictor=BitDance "
            "joint DiffHead; loss=RF + V4-weighted clean-endpoint MSE; "
            f"lambda={getattr(sampler_without_ddp, 'v4_lambda', float('nan')):g}; "
            f"NFEs={getattr(sampler_without_ddp, 'v4_nfes', ())}; "
            f"mean_W={getattr(sampler_without_ddp, 'v4_mean_weight', float('nan')):.8f}; "
            f"max_W={getattr(sampler_without_ddp, 'v4_max_weight', float('nan')):.8f}"
        )
        log(f"BitDance joint config: {getattr(sampler_without_ddp, 'bitdance_config', {})}")
        log(f"V4 grids: {getattr(sampler_without_ddp, 'v4_grids_by_nfe', {})}")
        log(f"V4 mean raw S^2 by NFE: {getattr(sampler_without_ddp, 'v4_mean_s2_by_nfe', {})}")
    elif experiment_variant == "aligned_bce":
        log("Time convention: train=t-1, sample=t-1; loss=base only")
    elif experiment_variant == "aligned_brier":
        log(
            "Time convention: train=t-1, sample=t-1; "
            f"loss=base + {getattr(sampler_without_ddp, 'src_lambda', float('nan')):g}*plain-Brier"
        )
    elif experiment_variant == "aligned_src":
        log(
            "Time convention: train=t-1, sample=t-1; "
            f"loss=base + {getattr(sampler_without_ddp, 'src_lambda', float('nan')):g}*normalized-S2-Brier; "
            f"mean_S2={getattr(sampler_without_ddp, 'src_mean_s2', float('nan')):.8f}"
        )
    elif experiment_variant == "multi_nfe_src_v4":
        log(
            "Time convention: train=t-1, sample=t-1; "
            f"loss=base + {getattr(sampler_without_ddp, 'v4_lambda', float('nan')):g}*W_multi(t)*Brier; "
            f"NFEs={getattr(sampler_without_ddp, 'v4_nfes', ())}; "
            f"NFE_weights={getattr(sampler_without_ddp, 'v4_nfe_weights', ())}; "
            f"mean_W={getattr(sampler_without_ddp, 'v4_mean_weight', float('nan')):.8f}; "
            f"max_W={getattr(sampler_without_ddp, 'v4_max_weight', float('nan')):.8f}"
        )
        log(f"V4 grids: {getattr(sampler_without_ddp, 'v4_grids_by_nfe', {})}")
        log(f"V4 mean raw S^2 by NFE: {getattr(sampler_without_ddp, 'v4_mean_s2_by_nfe', {})}")
    elif experiment_variant == "cached_endpoint_cavity":
        log(
            "Time convention: endpoint and cavity train/sample=t-1; "
            "independent cavity input removes queried self-evidence with the "
            "neutral 0.5 value."
        )
        log(f"Endpoint+cavity config: {sampler_without_ddp.cavity_config}")
    elif experiment_variant == "cached_selective_corrector":
        log(
            "Time convention: endpoint/gate/cavity train/sample=t-1; endpoint and "
            "corrector are independently MASK initialized and jointly trained from scratch."
        )
        log(f"Selective corrector config: {sampler_without_ddp.selective_config}")
    elif experiment_variant == "cached_cs_bfm":
        log(
            "CS-BFM: Transformer predicts c; analytic s*d reconstructs full z; "
            "BCEWithLogits(z,X0); d has no parameters."
        )
    if experiment_variant in {
        "bitdance_joint", "bitdance_joint_64d_inner_v2", "bitdance_joint_src_v4"
    }:
        log(
            "Sampling algorithm: BitDance joint X0 sample per spatial cell, then exact "
            "BFM endpoint bridge; outer np.linspace grid unchanged."
        )
    elif experiment_variant == "cached_endpoint_cavity":
        log(
            "Sampling algorithm: at every noisy level t>=1, an independent lazy "
            "context-only pseudo-Gibbs corrector runs BEFORE the expectation-consistent "
            "BFM predictor; no physical-time cutoff and no claim of exact block invariance."
        )
    elif experiment_variant == "cached_selective_corrector":
        log(
            "Sampling algorithm: deterministic gate-first candidate selection, "
            "context-only cavity proposal and unchanged expectation-consistent reverse."
        )
    else:
        log(
            "Sampling algorithm for all aligned variants: SAME expectation-consistent "
            "BFM posterior, SAME np.linspace grid, SAME hard final; no adaptive sampling."
        )
    log(f"AdamW betas: ({beta1}, {beta2}); update_freq={update_freq}")

    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    data_iterator = iter(train_loader)
    first_batch = True
    last_saved_step = completed_updates
    diagnostic_reference_x = None

    while completed_updates < total_updates:
        adjust_lr(optimizer, lr_sched, completed_updates)
        update_loss_values = []
        last_stats = None
        grad_norm = None
        step_start_time = time.time()

        for micro_index in range(update_freq):
            try:
                data = next(data_iterator)
            except StopIteration:
                epoch += 1
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
                data_iterator = iter(train_loader)
                data = next(data_iterator)

            if cached_x0_path:
                label = None
                x = cached_eval_module.unpack_x0(
                    data.to(device, non_blocking=True)
                )
            else:
                image, label = unpack_batch(data)
                image = image.to(device, non_blocking=True)
                if label is not None:
                    label = label.to(device, non_blocking=True)

                with torch.no_grad():
                    code = bergan(image, code_only=True).detach()
                    b, c, h, w = code.shape
                    x = code.view(b, c, -1).permute(0, 2, 1).contiguous()

            if diagnostic_reference_x is None and is_main_process(H):
                diagnostic_reference_x = x[:4].detach().clone()

            if first_batch:
                if is_main_process(H):
                    preview_model = ema_sampler if H.ema else sampler_without_ddp
                    generate_preview(H, bergan, preview_model, 999999999, x=x)
                if H.distributed:
                    dist.barrier()
                first_batch = False

            with torch.cuda.amp.autocast(enabled=H.amp):
                if H.dataset.startswith("imagenet"):
                    stats = sampler(x, label)
                else:
                    stats = sampler(x)
                raw_loss = stats["loss"]
                scaled_loss = raw_loss / update_freq

            should_update = micro_index == update_freq - 1
            grad_norm = scaler(
                scaled_loss,
                optimizer,
                clip_grad=H.grad_norm,
                parameters=sampler_without_ddp.parameters(),
                create_graph=False,
                update_grad=should_update,
            )

            update_loss_values.append(float(raw_loss.detach().item()))
            last_stats = stats

        optimizer.zero_grad(set_to_none=True)
        completed_updates += 1

        if H.ema and completed_updates % int(H.steps_per_update_ema) == 0:
            ema.update_model_average(ema_sampler, sampler_without_ddp)

        mean_update_loss = float(np.mean(update_loss_values))
        losses = np.append(losses, mean_update_loss)
        loss_ma.update(mean_update_loss)

        torch.cuda.synchronize(device)

        if is_main_process(H) and completed_updates % int(H.steps_per_log) == 0:
            logged_stats = dict(last_stats)
            logged_stats["lr"] = optimizer.param_groups[0]["lr"]
            logged_stats["step_time"] = time.time() - step_start_time
            logged_stats["mean_loss"] = loss_ma.avg()
            if grad_norm is not None:
                logged_stats["grad_norm"] = grad_norm
            if "scale" in scaler.state_dict():
                logged_stats["loss scale"] = scaler.state_dict()["scale"]
            mean_losses = np.append(mean_losses, loss_ma.avg())
            log_stats(completed_updates, logged_stats)

        fixed_eval_every = int(os.environ.get("BFM_FIXED_EVAL_EVERY", "500"))
        fixed_eval_due = (
            bool(cached_x0_path) and completed_updates % fixed_eval_every == 0
        )
        if fixed_eval_due and H.distributed:
            dist.barrier()
        if fixed_eval_due and is_main_process(H):
            eval_model = ema_sampler if H.ema else sampler_without_ddp
            fixed_metrics = cached_eval_module.record(
                H.log_dir, eval_model, cached_eval_x0, completed_updates,
                H.batch_size, seed, device,
            )
            log(
                f"Fixed-t EMA diagnostic step {completed_updates}: "
                f"{json.dumps(fixed_metrics)}"
            )
        if fixed_eval_due and H.distributed:
            dist.barrier()

        preview_due = completed_updates % int(H.steps_per_save_output) == 0
        if preview_due and is_main_process(H):
            if experiment_variant in {
                "bitdance_joint", "bitdance_joint_64d_inner_v2", "bitdance_joint_src_v4"
            }:
                preview_seed = int(
                    os.environ.get("EXPERIMENT_SEED", "20260821")
                ) + completed_updates
                device_index = device.index
                rng_devices = [] if device_index is None else [device_index]
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(preview_seed)
                    generate_preview(
                        H,
                        bergan,
                        sampler_without_ddp,
                        completed_updates,
                    )
                raw_source = (
                    Path(H.log_dir)
                    / "images"
                    / f"samples_{completed_updates:09d}.jpg"
                )
                raw_named = (
                    Path(H.log_dir)
                    / "images"
                    / f"samples_raw_{completed_updates:09d}.jpg"
                )
                if raw_source.exists():
                    raw_source.replace(raw_named)
                if ema_sampler is not None:
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(preview_seed)
                        generate_preview(
                            H,
                            bergan,
                            ema_sampler,
                            completed_updates,
                        )
                    ema_source = (
                        Path(H.log_dir)
                        / "images"
                        / f"samples_{completed_updates:09d}.jpg"
                    )
                    ema_named = (
                        Path(H.log_dir)
                        / "images"
                        / f"samples_ema_{completed_updates:09d}.jpg"
                    )
                    if ema_source.exists():
                        ema_source.replace(ema_named)
                diagnostic_save_steps = {
                    int(value)
                    for value in os.environ.get(
                        "BFM_BITDANCE_DIAGNOSTIC_SAVE_STEPS", "2000,4000"
                    ).split(",")
                    if value.strip()
                }
                if completed_updates in diagnostic_save_steps:
                    save_model(
                        sampler_without_ddp,
                        f"{H.sampler}_raw_diag",
                        completed_updates,
                        H.log_dir,
                    )
                    if ema_sampler is not None:
                        save_model(
                            ema_sampler,
                            f"{H.sampler}_ema_diag",
                            completed_updates,
                            H.log_dir,
                        )
            else:
                preview_model = ema_sampler if H.ema else sampler_without_ddp
                if experiment_variant == "cached_cs_bfm":
                    # Every CS-BFM preview uses identical initial/reverse draws.
                    # This makes changes across checkpoints attributable to the model.
                    preview_seed = int(
                        os.environ.get("EXPERIMENT_SEED", "20260821")
                    ) + 64_000_003
                    device_index = device.index
                    rng_devices = [] if device_index is None else [device_index]
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(preview_seed)
                        generate_preview(
                            H, bergan, preview_model, completed_updates
                        )
                    from PIL import Image
                    source = (
                        Path(H.log_dir) / "images"
                        / f"samples_{completed_updates:09}.jpg"
                    )
                    target = (
                        Path(H.log_dir) / "samples_preview"
                        / f"step_{completed_updates:06}.png"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(source) as preview_image:
                        preview_image.save(target)
                else:
                    generate_preview(H, bergan, preview_model, completed_updates)
        if preview_due and H.distributed:
            dist.barrier()

        checkpoint_due = completed_updates % int(H.steps_per_checkpoint) == 0
        if checkpoint_due and is_main_process(H):
            save_training_checkpoint(
                H,
                sampler_without_ddp,
                optimizer,
                scaler,
                ema_sampler,
                completed_updates,
                losses,
                mean_losses,
                val_losses,
                elbo,
                val_elbos,
            )
            last_saved_step = completed_updates
        if checkpoint_due and H.distributed:
            dist.barrier()

    if is_main_process(H) and last_saved_step != completed_updates:
        save_training_checkpoint(
            H,
            sampler_without_ddp,
            optimizer,
            scaler,
            ema_sampler,
            completed_updates,
            losses,
            mean_losses,
            val_losses,
            elbo,
            val_elbos,
        )

    if is_main_process(H) and experiment_variant == "cached_endpoint_cavity" and freeze_endpoint:
        endpoint_audit = audit_frozen_endpoint(
            sampler_without_ddp._denoise_fn,
            endpoint_checkpoint,
            Path(H.log_dir) / f"endpoint_frozen_audit_step_{completed_updates:06d}.json",
        )
        log(f"Frozen endpoint global max_abs_diff: {endpoint_audit['global_max_abs_diff']}")

    if H.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    H = get_sampler_hparams()
    experiment_variant = os.environ.get(
        "EXPERIMENT_VARIANT", "aligned_bce"
    ).strip().lower()
    if experiment_variant not in {
        "original", "comparison_bfm", "bitdance_bce_control", "bitdance_joint", "bitdance_joint_64d_inner_v2", "bitdance_joint_src_v4", "aligned_bce", "aligned_brier", "aligned_src",
        "multi_nfe_src_v4", "v6_sensitivity_anchor", "cached_direct_x0_bce",
        "cached_endpoint_cavity", "cached_selective_corrector", "cached_cs_bfm"
    }:
        raise ValueError(
            f"Unknown EXPERIMENT_VARIANT={experiment_variant!r}"
        )
    H.sampler = f"flow_{experiment_variant}"
    # This experiment intentionally disables the old posterior-CE auxiliary.
    if float(H.aux) != 0.0:
        raise ValueError("Use --aux 0 for the controlled SRC ablation.")
    config_log(H.log_dir)
    log("---------------------------------")
    log(f"Setting up training for {H.sampler}")
    start_training_log(H)
    main(H, None)
