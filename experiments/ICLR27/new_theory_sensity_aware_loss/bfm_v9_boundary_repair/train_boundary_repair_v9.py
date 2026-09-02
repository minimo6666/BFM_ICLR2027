#!/usr/bin/env python3
"""Small frozen-backbone training run for the V9 terminal boundary repair."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

import misc
from hparams import get_sampler_hparams
from models.binaryae import BinaryAutoEncoder
from models.binarylatent_flow_boundary_repair_v9 import (
    BinaryDiffusionFlowBoundaryRepairV9,
)
from models.transformer_boundary_repair_v9 import TransformerBDBoundaryRepairV9
from utils.reliable_data_utils import get_data_loaders
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts
from utils.train_utils import NativeScalerWithGradNormCount


def is_main(H) -> bool:
    return (not H.distributed) or dist.get_rank() == 0


def seed_everything(H) -> int:
    seed = int(os.environ.get("EXPERIMENT_SEED", "20260901"))
    rank = dist.get_rank() if H.distributed else 0
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    # Identical model/head initialization on every rank.  The rank-specific
    # training RNG is installed only after DDP construction below.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def unpack_batch(data: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(data, (tuple, list)):
        return data[0], data[1] if len(data) > 1 else None
    return data, None


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    candidate = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "module", "ema", "ema_state_dict"):
            if isinstance(payload.get(key), Mapping):
                candidate = payload[key]
                break
    if not isinstance(candidate, Mapping):
        raise TypeError("V8 checkpoint does not contain a state dict")
    state = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    if not state:
        raise ValueError("No tensors found in V8 checkpoint")
    return state


def load_v8_into_v9(model: torch.nn.Module, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = extract_state_dict(torch_load(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_prefix = "_denoise_fn.terminal_carry."
    illegal_missing = [key for key in missing if not key.startswith(allowed_prefix)]
    if illegal_missing or unexpected:
        raise RuntimeError(
            "V8->V9 load mismatch. "
            f"illegal_missing={illegal_missing[:10]}, unexpected={unexpected[:10]}"
        )
    expected_missing = {
        key for key in model.state_dict() if key.startswith(allowed_prefix)
    }
    if set(missing) != expected_missing:
        raise RuntimeError(
            f"Terminal-head mismatch: missing={sorted(missing)}, "
            f"expected={sorted(expected_missing)}"
        )


class HeadEMA:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for key, value in module.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def cpu_state(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.shadow.items()}


def reduce_metrics(stats: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, float]:
    names = sorted(stats)
    values = torch.stack([stats[name].detach().float() for name in names]).to(device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(value.item()) for name, value in zip(names, values)}


def save_checkpoint(
    *,
    output: Path,
    step: int,
    model: BinaryDiffusionFlowBoundaryRepairV9,
    ema: HeadEMA,
    optimizer: torch.optim.Optimizer,
    scaler: NativeScalerWithGradNormCount,
    config: Mapping[str, object],
) -> Path:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "bfm_v9_boundary_head_v1",
        "step": int(step),
        "head_state": {
            key: value.detach().cpu()
            for key, value in model._denoise_fn.terminal_carry.state_dict().items()
        },
        "ema_head_state": ema.cpu_state(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dict(config),
    }
    path = checkpoint_dir / f"boundary_head_step_{step}.pt"
    torch.save(payload, path)
    latest = checkpoint_dir / "boundary_head_latest.pt"
    torch.save(payload, latest)
    return path


def main(H) -> None:
    misc.init_distributed_mode(H)
    seed = seed_everything(H)
    if not torch.cuda.is_available():
        raise RuntimeError("V9 boundary training requires CUDA")
    device = torch.device(f"cuda:{H.gpu if H.distributed else 0}")

    v8_checkpoint = Path(os.environ["V8_CHECKPOINT"]).expanduser().resolve()
    output = Path(H.log_dir).expanduser().resolve()
    if is_main(H):
        output.mkdir(parents=True, exist_ok=True)

    ae_state = retrieve_autoencoder_components_state_dicts(
        H, ["encoder", "quantize", "generator"], remove_component_from_key=False
    )
    autoencoder = BinaryAutoEncoder(H)
    autoencoder.load_state_dict(ae_state, strict=True)
    del ae_state
    autoencoder = autoencoder.to(device).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    denoiser = TransformerBDBoundaryRepairV9(H).to(device)
    model_without_ddp = BinaryDiffusionFlowBoundaryRepairV9(
        H, denoiser, H.codebook_size
    ).to(device)
    load_v8_into_v9(model_without_ddp, v8_checkpoint)
    model_without_ddp.freeze_v8_for_boundary_training()

    initial_up_max = float(
        model_without_ddp._denoise_fn.terminal_carry.up.weight.detach().abs().max().item()
    )
    if initial_up_max != 0.0:
        raise RuntimeError(
            "Fresh V9 terminal head is not zero initialized; V8 identity would fail: "
            f"max_abs={initial_up_max}"
        )

    trainable = list(model_without_ddp.boundary_parameters())
    trainable_ids = {id(parameter) for parameter in trainable}
    leaked = [name for name, parameter in model_without_ddp.named_parameters()
              if parameter.requires_grad and id(parameter) not in trainable_ids]
    if leaked:
        raise RuntimeError(f"Non-boundary trainable parameters detected: {leaked[:10]}")
    if not trainable:
        raise RuntimeError("No trainable boundary parameters")

    if H.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp,
            device_ids=[H.gpu],
            find_unused_parameters=False,
        )
    else:
        model = model_without_ddp

    training_rank = dist.get_rank() if H.distributed else 0
    torch.manual_seed(seed + 1009 * training_rank + 17)
    torch.cuda.manual_seed_all(seed + 1009 * training_rank + 17)

    learning_rate = float(os.environ.get("BFM_V9_LR", "0.001"))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=float(os.environ.get("BFM_V9_WEIGHT_DECAY", "0.0")),
    )
    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)
    ema = HeadEMA(
        model_without_ddp._denoise_fn.terminal_carry,
        decay=float(os.environ.get("BFM_V9_EMA_DECAY", "0.995")),
    )

    config = {
        "seed": seed,
        "v8_checkpoint": str(v8_checkpoint),
        "nfes": list(model_without_ddp.boundary_nfes),
        "preterminal_steps": {
            str(nfe): int(model_without_ddp.lr_grids_by_nfe[nfe][-2])
            for nfe in model_without_ddp.boundary_nfes
        },
        "branches": model_without_ddp.boundary_branches,
        "hard_tau": model_without_ddp.boundary_hard_tau,
        "lambda_bce": model_without_ddp.boundary_lambda_bce,
        "lambda_prob": model_without_ddp.boundary_lambda_prob,
        "lambda_hard": model_without_ddp.boundary_lambda_hard,
        "lambda_anchor": model_without_ddp.boundary_lambda_anchor,
        "learning_rate": learning_rate,
        "train_steps": int(H.train_steps),
        "batch_size_per_rank": int(H.batch_size),
        "world_size": dist.get_world_size() if H.distributed else 1,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_terminal_up_max_abs": initial_up_max,
        "hard_final_unchanged": True,
        "backbone_frozen": True,
        "sensitivity_adapter_frozen": True,
    }
    if is_main(H):
        with (output / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    train_loader, _ = get_data_loaders(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_dataloader=False,
        custom_dataset_path=H.path_to_data,
        num_workers=int(os.environ.get("BFM_V9_WORKERS", "4")),
        distributed=H.distributed,
        random=True,
        args=H,
    )
    epoch = 0
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    update_freq = max(1, int(H.update_freq))
    checkpoint_every = int(H.steps_per_checkpoint)
    log_every = int(H.steps_per_log)
    optimizer.zero_grad(set_to_none=True)
    log_path = output / "metrics.jsonl"

    for step in range(1, int(H.train_steps) + 1):
        started = time.time()
        last_stats = None
        grad_norm = None
        for micro in range(update_freq):
            try:
                data = next(iterator)
            except StopIteration:
                epoch += 1
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                data = next(iterator)
            image, _ = unpack_batch(data)
            image = image.to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=H.amp):
                code = autoencoder(image, code_only=True).detach()
                batch, channels, height, width = code.shape
                x0 = code.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()

            with torch.cuda.amp.autocast(enabled=H.amp):
                stats = model(x0)
                scaled_loss = stats["loss"] / update_freq
            should_update = micro == update_freq - 1
            grad_norm = scaler(
                scaled_loss,
                optimizer,
                clip_grad=float(H.grad_norm),
                parameters=trainable,
                create_graph=False,
                update_grad=should_update,
            )
            last_stats = stats
        optimizer.zero_grad(set_to_none=True)
        ema.update(model_without_ddp._denoise_fn.terminal_carry)

        if step % log_every == 0 or step == 1:
            reduced = reduce_metrics(last_stats, device)
            reduced.update({
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - started,
                "grad_norm": float(grad_norm.item()) if grad_norm is not None else 0.0,
            })
            if is_main(H):
                print(json.dumps(reduced, sort_keys=True), flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(reduced, sort_keys=True) + "\n")

        if step % checkpoint_every == 0 and is_main(H):
            path = save_checkpoint(
                output=output,
                step=step,
                model=model_without_ddp,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
            )
            print(f"saved {path}", flush=True)
        if H.distributed and step % checkpoint_every == 0:
            dist.barrier()

    if is_main(H) and int(H.train_steps) % checkpoint_every != 0:
        path = save_checkpoint(
            output=output,
            step=int(H.train_steps),
            model=model_without_ddp,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
        )
        print(f"saved {path}", flush=True)
    if H.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    H = get_sampler_hparams()
    H.sampler = "flow_boundary_repair_v9"
    if float(H.aux) != 0.0:
        raise ValueError("Use --aux 0; V9 has its own isolated objective")
    main(H)
