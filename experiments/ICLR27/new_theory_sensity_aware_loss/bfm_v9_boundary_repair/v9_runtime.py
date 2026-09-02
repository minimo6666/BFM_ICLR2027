"""Self-contained runtime helpers for the V9 boundary-repair experiment."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_sampling_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
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
    """Rebuild the exact hyperparameters used by the controlled V8/V9 runs."""
    from hparams.defaults.binarygan_default import HparamsBinaryAE
    from hparams.defaults.sampler_defaults import HparamsBianryLatent

    H = HparamsBinaryAE(dataset)
    H.update(HparamsBianryLatent(dataset))
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
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
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
    if code.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] code, got {tuple(code.shape)}")
    return code.flatten(2).permute(0, 2, 1).contiguous().float()


def sequence_to_binary_code(
    sequence: torch.Tensor, latent_hw: Tuple[int, int] = (16, 16)
) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B,L,C] sequence, got {tuple(sequence.shape)}")
    batch, length, channels = sequence.shape
    height, width = latent_hw
    if length != height * width:
        raise ValueError(
            f"Sequence length {length} does not match latent grid {height}x{width}"
        )
    return sequence.permute(0, 2, 1).reshape(
        batch, channels, height, width
    ).contiguous()


def make_analysis_loader(
    *, H, data_root: str | Path, split: str, batch_size: int, num_workers: int, seed: int
):
    from torch.utils.data import DataLoader
    from utils.reliable_data_utils import get_datasets

    train_dataset, val_dataset = get_datasets(
        H.dataset,
        H.img_size,
        get_val_dataset=split == "val",
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


def read_completed(progress_path: Path) -> int:
    if not progress_path.is_file():
        return 0
    with progress_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return int(payload.get("completed", 0))


def validate_existing_images(image_dir: Path, num_samples: int) -> None:
    if not image_dir.is_dir():
        return
    indices = []
    for path in image_dir.glob("*.png"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    if indices and max(indices) >= num_samples:
        raise RuntimeError(
            f"{image_dir} contains sample index {max(indices)}, but "
            f"--num-samples={num_samples}. Use a new output directory."
        )
