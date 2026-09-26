"""Binary-latent I/O for the otherwise unmodified official SEDD core."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
SEDD_ROOT = HERE.parents[1]
PROJECT_ROOT = HERE.parents[4]

for path in (str(PROJECT_ROOT), str(SEDD_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SEDD_ROOT))

# Import the official SEDD components before exposing the BLD project's
# top-level ``utils`` package required by BinaryAutoEncoder.
import graph_lib  # noqa: E402
import losses  # noqa: E402
import noise_lib  # noqa: E402
import sampling  # noqa: E402
from packed_binary_sedd import PackedBinarySEDD  # noqa: E402
from model.ema import ExponentialMovingAverage  # noqa: E402


def binary_sedd_config():
    """Official SEDD-small settings with packed binary coordinates."""
    return OmegaConf.create(
        {
            "tokens": 2,
            "ngpus": 2,
            "graph": {"type": "uniform"},
            "noise": {
                "type": "geometric",
                "sigma_min": 1e-4,
                "sigma_max": 20.0,
            },
            "model": {
                "name": "small",
                "type": "ddit",
                "hidden_size": 768,
                "cond_dim": 128,
                # Internal DDiT length. The diffusion coordinate length remains
                # 64 * 16 * 16 = 16384 at the public model interface.
                "length": 1024,
                "n_blocks": 12,
                "n_heads": 12,
                "scale_by_sigma": False,
                "dropout": 0.1,
            },
            "training": {"ema": 0.9999, "accum": 12},
            "optim": {
                "weight_decay": 0.0,
                "optimizer": "AdamW",
                "lr": 3e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "warmup": 2500,
                "grad_clip": 1.0,
            },
            "sampling": {
                "predictor": "analytic",
                "steps": 63,
                "noise_removal": True,
            },
        }
    )


def _expose_project_utils_package():
    package = types.ModuleType("utils")
    package.__path__ = [str(PROJECT_ROOT / "utils")]
    package.__package__ = "utils"
    sys.modules["utils"] = package
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(1, str(PROJECT_ROOT))


def load_binaryae_class():
    _expose_project_utils_package()
    package_name = "bld_project_models"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(PROJECT_ROOT / "models")]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.binaryae").BinaryAutoEncoder


def build_ae_hparams(ae_load_dir: str, ae_load_step: int, batch_size: int = 2):
    from hparams.defaults.binarygan_default import HparamsBinaryAE

    hparams = HparamsBinaryAE("churches")
    hparams.update(
        codebook_size=64,
        norm_first=True,
        img_size=256,
        latent_shape=[1, 16, 16],
        batch_size=batch_size,
        ae_load_dir=ae_load_dir,
        ae_load_step=ae_load_step,
    )
    return hparams


def load_autoencoder(ae_load_dir: str, ae_load_step: int, device: torch.device):
    binary_autoencoder = load_binaryae_class()
    hparams = build_ae_hparams(ae_load_dir, ae_load_step)
    saved_models = Path(ae_load_dir) / "saved_models"
    checkpoint_path = saved_models / f"binaryae_ema_{ae_load_step}.th"
    if not checkpoint_path.exists():
        checkpoint_path = saved_models / f"binaryae_{ae_load_step}.th"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BAE checkpoint not found: {checkpoint_path}")
    print(f"Loading Binary Autoencoder from {checkpoint_path}", flush=True)
    full_state = torch.load(checkpoint_path, map_location="cpu")
    state = {
        key[3:]: value
        for key, value in full_state.items()
        if any(component in key for component in ("encoder", "quantize", "generator"))
    }
    model = binary_autoencoder(hparams)
    model.load_state_dict(state, strict=True)
    del state, full_state
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_images(autoencoder, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        bits = autoencoder(images, code_only=True).detach()
    if bits.shape[1:] != (64, 16, 16):
        raise RuntimeError(f"Unexpected BAE latent shape: {tuple(bits.shape)}")
    bits = bits.long()
    if torch.any((bits != 0) & (bits != 1)):
        raise RuntimeError("BAE returned values outside {0,1}")
    return bits


def latent_to_tokens(bits: torch.Tensor) -> torch.Tensor:
    if bits.ndim != 4 or bits.shape[1:] != (64, 16, 16):
        raise ValueError(f"Expected [B,64,16,16], got {tuple(bits.shape)}")
    return bits.permute(0, 2, 3, 1).contiguous().reshape(bits.shape[0], 16384).long()


def tokens_to_latent(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[1] != 16384:
        raise ValueError(f"Expected [B,16384], got {tuple(tokens.shape)}")
    return tokens.reshape(tokens.shape[0], 16, 16, 64).permute(0, 3, 1, 2).contiguous()


def decode_bits(autoencoder, bits: torch.Tensor) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=bits.is_cuda):
        images, _, _ = autoencoder(None, code=bits.float())
    return images


def create_sedd_components(device: torch.device):
    cfg = binary_sedd_config()
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    model = PackedBinarySEDD(cfg).to(device)
    return cfg, graph, noise, model


def build_strict_64_nfe_sampler(graph, noise, batch_size: int, device: torch.device):
    # 63 analytic predictor calls + official final denoiser call = exactly 64 NFE.
    return sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(batch_size, 16384),
        predictor="analytic",
        steps=63,
        denoise=True,
        eps=1e-5,
        device=device,
    )

