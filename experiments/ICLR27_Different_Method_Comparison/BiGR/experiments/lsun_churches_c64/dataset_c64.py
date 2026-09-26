"""Packed BAE-C64 latent dataset for the official BiGR LSUN experiment."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedC64Dataset(Dataset):
    packed_width = 256 * 64 // 8

    def __init__(self, path):
        self.path = str(Path(path).expanduser().resolve())
        self.latents = np.load(self.path, mmap_mode="r")
        if self.latents.shape[1:] != (self.packed_width,):
            raise ValueError(
                f"Expected packed cache [N,{self.packed_width}], got {self.latents.shape}"
            )
        if self.latents.dtype != np.uint8:
            raise ValueError(f"Expected uint8 cache, got {self.latents.dtype}")

    def __len__(self):
        return int(self.latents.shape[0])

    def __getitem__(self, index):
        packed = np.asarray(self.latents[index])
        bits = np.unpackbits(
            packed, count=256 * 64, bitorder="little"
        ).reshape(256, 64)
        return torch.from_numpy(bits).float(), torch.tensor(0, dtype=torch.long)
