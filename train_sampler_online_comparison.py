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
import os
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from hparams import get_sampler_hparams
from models.binaryae import BinaryAutoEncoder
from models.binarylatent_flow_expectation_consistent_retrain import (
    BinaryDiffusionFlowDecouple,
)
from models.transformer import TransformerBD
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


def main(H, vis=None):
    del vis
    misc.init_distributed_mode(H)
    device = current_device(H)

    if H.sampler.startswith("flow_") and float(H.aux) <= 0.0:
        log(
            "WARNING: H.aux <= 0. The corrected posterior is used during "
            "sampling, but the training objective remains the same legacy X0 "
            "BCE. For a genuinely corrected retraining run, use e.g. --aux 0.1."
        )

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
    if H.sampler.startswith("flow_"):
        denoiser = TransformerBD(H).to(device)
        sampler_without_ddp = BinaryDiffusionFlowDecouple(
            H, denoiser, H.codebook_size
        ).to(device)
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
    optimizer = torch.optim.AdamW(
        sampler_without_ddp.parameters(),
        lr=H.lr,
        weight_decay=H.weight_decay,
        betas=(beta1, beta2),
        eps=H.optim_eps,
    )

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

    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters()) / 1e6:.2f}M")
    log(f"Training objective: BCE + {float(H.aux):g} * auxiliary BCE")
    if H.sampler.startswith("flow_"):
        log(
            "Posterior mode: "
            f"{getattr(sampler_without_ddp, 'x0_posterior_mode', 'unknown')}; "
            "aux interval mode: "
            f"{getattr(sampler_without_ddp, 'aux_interval_mode', 'unknown')}"
        )
    log(f"AdamW betas: ({beta1}, {beta2}); update_freq={update_freq}")

    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    data_iterator = iter(train_loader)
    first_batch = True
    last_saved_step = completed_updates

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

            image, label = unpack_batch(data)
            image = image.to(device, non_blocking=True)
            if label is not None:
                label = label.to(device, non_blocking=True)

            with torch.no_grad():
                code = bergan(image, code_only=True).detach()
                b, c, h, w = code.shape
                x = code.view(b, c, -1).permute(0, 2, 1).contiguous()

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

        preview_due = completed_updates % int(H.steps_per_save_output) == 0
        if preview_due and is_main_process(H):
            preview_model = ema_sampler if H.ema else sampler_without_ddp
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

    if H.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    H = get_sampler_hparams()
    experiment_method = os.environ.get("EXPERIMENT_METHOD", "bld").lower()
    if experiment_method == "bfm":
        H.sampler = "flow_lsun"
    elif experiment_method != "bld":
        raise ValueError(f"Unknown EXPERIMENT_METHOD={experiment_method!r}")
    config_log(H.log_dir)
    log("---------------------------------")
    log(f"Setting up training for {H.sampler}")
    start_training_log(H)
    main(H, None)
