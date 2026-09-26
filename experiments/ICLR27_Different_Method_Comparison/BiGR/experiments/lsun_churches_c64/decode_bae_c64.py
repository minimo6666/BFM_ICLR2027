#!/usr/bin/env python3
"""Decode packed C64 latents with the unchanged BFM BAE-C64 decoder."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


BFM_ROOT = Path("/home/mohao/WorkSpace/Projects/BFM_ICLR2027")
sys.path.insert(0, str(BFM_ROOT))

from hparams.defaults.binarygan_default import HparamsBinaryAE  # noqa: E402
from hparams.defaults.sampler_defaults import HparamsBianryLatent  # noqa: E402
from models.binaryae import Generator  # noqa: E402
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts  # noqa: E402


def build_decoder():
    H = HparamsBinaryAE("churches")
    H.vqgan_batch_size = H.batch_size
    H.update(HparamsBianryLatent("churches"))
    H.codebook_size = 64
    H.latent_shape = [1, 16, 16]
    H.norm_first = True
    H.ae_load_dir = str(BFM_ROOT / "logs/BAE_C64")
    H.ae_load_step = 8100000
    state = retrieve_autoencoder_components_state_dicts(
        H, ["quantize", "generator"], remove_component_from_key=True
    )
    embedding = state.pop("embed.weight").cuda()
    generator = Generator(H)
    generator.load_state_dict(state, strict=False)
    return H, embedding, generator.cuda().eval()


@torch.no_grad()
def decode_packed(path, output_dir, grid_name=None, output_offset=0, decoder=None):
    packed = np.load(path)
    if packed.ndim != 2 or packed.shape[1] != 2048 or packed.dtype != np.uint8:
        raise ValueError(f"Expected packed uint8 [B,2048], got {packed.shape} {packed.dtype}")
    bits = np.unpackbits(packed, axis=1, count=256 * 64, bitorder="little")
    bits = torch.from_numpy(bits.reshape(-1, 256, 64).copy()).float().cuda()
    H, embedding, generator = decoder or build_decoder()
    quantized = (bits @ embedding).permute(0, 2, 1).reshape(-1, H.emb_dim, 16, 16)
    with torch.cuda.amp.autocast(enabled=True):
        images = generator(quantized).float().cpu().clamp(0, 1)
    if tuple(images.shape[1:]) != (3, 256, 256) or not torch.isfinite(images).all():
        raise RuntimeError(f"Invalid decoded images: {images.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        destination = output_dir / f"{output_offset + index:05d}.png"
        if not destination.exists():
            save_image(image, destination)
    if grid_name:
        save_image(images, output_dir / grid_name, nrow=8, padding=0)
    print(f"DECODED shape={tuple(images.shape)} finite=True output={output_dir}", flush=True)


def decode_directory(packed_dir, output_dir):
    decoder = build_decoder()
    files = sorted(packed_dir.glob("packed_*.npy"))
    if not files:
        raise RuntimeError(f"No packed_*.npy files in {packed_dir}")
    for path in files:
        match = re.fullmatch(r"packed_(\d+)_(\d+)\.npy", path.name)
        if not match:
            continue
        start, end = map(int, match.groups())
        expected = [output_dir / f"{index:05d}.png" for index in range(start, end)]
        if all(path.exists() for path in expected):
            continue
        decode_packed(path, output_dir, output_offset=start, decoder=decoder)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packed", type=Path)
    source.add_argument("--packed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-name")
    args = parser.parse_args()
    if args.packed_dir:
        if args.grid_name:
            raise ValueError("--grid-name is only valid with --packed")
        decode_directory(args.packed_dir, args.output_dir)
    else:
        decode_packed(args.packed, args.output_dir, args.grid_name)


if __name__ == "__main__":
    main()
