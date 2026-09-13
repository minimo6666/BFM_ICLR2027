"""Shared loading and reproducibility utilities for the V8 post-hoc audit.

The scripts in this directory intentionally construct the exact aligned-BFM
and V8 classes directly.  They do not use ``utils.get_sampler`` because that
factory still points at the historical BLD implementation.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_int_list(text: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate values are not allowed: {values}")
    return values


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one floating-point value.")
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"All values must be finite: {values}")
    return values


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    return device


def build_hparams(
    *,
    dataset: str = "churches",
    codebook_size: int = 64,
    img_size: int = 256,
    total_steps: int = 64,
    batch_size: int = 8,
    ae_deterministic: bool = False,
):
    """Rebuild the exact H object used by the controlled V4/V8 runs."""
    from hparams.defaults.binarygan_default import HparamsBinaryAE
    from hparams.defaults.sampler_defaults import HparamsBianryLatent

    H = HparamsBinaryAE(dataset)
    H.update(HparamsBianryLatent(dataset))

    # Explicit controlled-run overrides from common_v4.sh/common_v8.sh.
    H.dataset = dataset
    H.sampler = "flow_lowrank_sensitivity_v8"
    H.codebook_size = int(codebook_size)
    H.img_size = int(img_size)
    H.total_steps = int(total_steps)
    H.sample_steps = int(total_steps)
    H.batch_size = int(batch_size)
    H.latent_shape = [1, 16, 16]
    H.loss_final = "mean"
    H.beta_type = "linear"
    H.p_flip = True
    H.norm_first = True
    H.aux = 0.0
    H.guidance = False
    H.cross = False
    H.use_softmax = False
    H.use_tanh = False
    H.hard_final = True
    H.x0_posterior_mode = "expectation_consistent"
    H.deterministic = bool(ae_deterministic)
    H.quantizer = "binary"
    H.deepspeed = False
    return H


def _torch_load(path: Path):
    """Load a trusted local tensor checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    """Extract a model state dict and remove a possible DDP prefix."""
    candidate = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "module", "ema", "ema_state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping) and value:
                candidate = value
                break

    if not isinstance(candidate, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {type(candidate)!r}")

    state: Dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue
        key = raw_key
        while key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    if not state:
        raise ValueError("No tensor entries were found in the checkpoint.")
    return state


def load_checkpoint_strict(model: torch.nn.Module, checkpoint: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    state = extract_state_dict(_torch_load(path))
    model.load_state_dict(state, strict=True)
    return state


def build_v8_model(H, checkpoint: str | Path, device: torch.device):
    from models.binarylatent_flow_lowrank_sensitivity_v8 import (
        BinaryDiffusionFlowLowRankSensitivityV8,
    )
    from models.transformer_lowrank_sensitivity_v8 import (
        TransformerBDLowRankSensitivityV8,
    )

    denoiser = TransformerBDLowRankSensitivityV8(H)
    model = BinaryDiffusionFlowLowRankSensitivityV8(H, denoiser, H.codebook_size)
    load_checkpoint_strict(model, checkpoint)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_aligned_base_model(
    H,
    device: torch.device,
    checkpoint: Optional[str | Path] = None,
):
    from models.binarylatent_flow_controlled_src_v4 import BinaryDiffusionFlowTimeAligned
    from models.transformer import TransformerBD

    denoiser = TransformerBD(H)
    model = BinaryDiffusionFlowTimeAligned(H, denoiser, H.codebook_size)
    if checkpoint is not None:
        load_checkpoint_strict(model, checkpoint)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def copy_v8_base_into_aligned(v8_model, aligned_model) -> None:
    """Copy only the V8 checkpoint's base tensors into an aligned-BFM model."""
    source = v8_model.state_dict()
    target = aligned_model.state_dict()
    copied: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    for key, target_value in target.items():
        source_value = source.get(key)
        if source_value is None or source_value.shape != target_value.shape:
            missing.append(key)
        else:
            copied[key] = source_value
    if missing:
        raise RuntimeError(f"V8 base is missing {len(missing)} aligned tensors: {missing[:8]}")
    aligned_model.load_state_dict(copied, strict=True)


def compare_matching_state_dicts(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    ignored_substrings: Sequence[str] = ("sensitivity_adapter",),
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    total_numel = 0
    weighted_abs_sum = 0.0
    overall_max = 0.0
    missing_left: List[str] = []
    missing_right: List[str] = []

    keys = sorted(set(left) | set(right))
    for key in keys:
        if any(token in key for token in ignored_substrings):
            continue
        if key not in left:
            missing_left.append(key)
            continue
        if key not in right:
            missing_right.append(key)
            continue
        a, b = left[key], right[key]
        if a.shape != b.shape:
            rows.append({"key": key, "shape_mismatch": [list(a.shape), list(b.shape)]})
            continue
        diff = (a.float() - b.float()).abs()
        numel = diff.numel()
        mean_abs = float(diff.mean().item()) if numel else 0.0
        max_abs = float(diff.max().item()) if numel else 0.0
        rows.append({"key": key, "numel": numel, "mean_abs": mean_abs, "max_abs": max_abs})
        total_numel += numel
        weighted_abs_sum += mean_abs * numel
        overall_max = max(overall_max, max_abs)

    rows.sort(key=lambda row: float(row.get("max_abs", -1.0)), reverse=True)
    return {
        "matched_tensor_count": sum("numel" in row for row in rows),
        "matched_numel": total_numel,
        "mean_abs": weighted_abs_sum / max(total_numel, 1),
        "max_abs": overall_max,
        "missing_left": missing_left,
        "missing_right": missing_right,
        "largest_differences": rows[:20],
    }


def load_autoencoder(H, checkpoint: str | Path, device: torch.device):
    from models.binaryae import BinaryAutoEncoder

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    full_state = extract_state_dict(_torch_load(path))
    selected: Dict[str, torch.Tensor] = {}
    prefixes = ("encoder.", "quantize.", "generator.")
    for raw_key, value in full_state.items():
        key = raw_key[3:] if raw_key.startswith("ae.") else raw_key
        if key.startswith(prefixes):
            selected[key] = value
    if not selected:
        raise RuntimeError(f"No autoencoder tensors found in {path}")

    autoencoder = BinaryAutoEncoder(H)
    autoencoder.load_state_dict(selected, strict=True)
    autoencoder = autoencoder.to(device).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    return autoencoder


def binary_code_to_sequence(code: torch.Tensor) -> torch.Tensor:
    """Convert [B,C,H,W] binary AE codes to the sampler's [B,L,C]."""
    if code.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] code, got {tuple(code.shape)}")
    return code.flatten(2).permute(0, 2, 1).contiguous().float()


def sequence_to_binary_code(sequence: torch.Tensor, latent_hw: Tuple[int, int] = (16, 16)) -> torch.Tensor:
    """Convert [B,L,C] sampler output to the AE decoder's [B,C,H,W]."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B,L,C] sequence, got {tuple(sequence.shape)}")
    b, length, channels = sequence.shape
    h, w = latent_hw
    if length != h * w:
        raise ValueError(f"Sequence length {length} does not match latent grid {h}x{w}")
    return sequence.permute(0, 2, 1).reshape(b, channels, h, w).contiguous()


def make_analysis_loader(
    *,
    H,
    data_root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    from torch.utils.data import DataLoader
    from utils.reliable_data_utils import get_datasets

    need_validation = split == "val"
    train_dataset, val_dataset = get_datasets(
        H.dataset,
        H.img_size,
        get_val_dataset=need_validation,
        custom_dataset_path=str(Path(data_root).expanduser()),
        random=False,
    )
    if split == "val":
        dataset = val_dataset
    elif split == "train":
        dataset = train_dataset
    else:
        raise ValueError("split must be 'train' or 'val'")
    if dataset is None:
        raise RuntimeError(f"Dataset split {split!r} is unavailable.")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)


def scale_tag(scale: float) -> str:
    sign = "m" if scale < 0 else ""
    return sign + f"{abs(float(scale)):.2f}".replace(".", "p")
