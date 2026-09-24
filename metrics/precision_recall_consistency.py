#!/usr/bin/env python3
"""Improved Precision/Recall for image samples.

This is a PyTorch/NumPy port of the manifold estimator used by
openai/consistency_models/evaluations/evaluator.py.  Features are extracted
with the same ``torch-fidelity`` Inception-v3-compatible pool features used by
the repository's FID scripts, then the k-NN hypersphere test is applied.

Examples:
  python metrics/precision_recall_consistency.py \
      --real-dir /path/to/real --fake-dir /path/to/fake \
      --output-json pr.json --device cuda

Precomputed feature arrays (N x D, .npy or .npz with ``features``/``arr_0``)
can be supplied with --real-features and --fake-features to skip extraction.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_REFERENCE_IMAGES = 10_000


class ImageFiles(Dataset):
    def __init__(self, root: Path, limit: Optional[int] = None):
        self.paths = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_features(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.ndarray):
        features = obj
    else:
        key = "features" if "features" in obj.files else "arr_0"
        if key not in obj.files:
            raise ValueError(f"{path} must contain features or arr_0")
        features = obj[key]
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N,D] features, got {features.shape}")
    return features


@torch.inference_mode()
def extract_features(image_dir: Path, batch_size: int, device: torch.device, limit: Optional[int]):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    dataset = ImageFiles(image_dir, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
    chunks = []
    started = time.time()
    for index, batch in enumerate(loader, 1):
        values = extractor(batch.to(device, non_blocking=True))[0]
        chunks.append(values.float().cpu().numpy())
        if index == 1 or index % 50 == 0 or index * batch_size >= len(dataset):
            print(f"features {min(index * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    result = np.concatenate(chunks, axis=0)
    print(f"extracted {len(result)} features in {time.time() - started:.1f}s", flush=True)
    return result


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norm - 2.0 * (left @ right.T) + right_norm
    return np.maximum(distances, 0.0)


def manifold_radii(features: np.ndarray, k: int, row_batch_size: int, col_batch_size: int) -> np.ndarray:
    if k < 1 or k >= len(features):
        raise ValueError(f"k must satisfy 1 <= k < number of features ({len(features)})")
    radii = np.empty(len(features), dtype=np.float32)
    for begin in range(0, len(features), row_batch_size):
        end = min(begin + row_batch_size, len(features))
        row = features[begin:end]
        distances = np.empty((end - begin, len(features)), dtype=np.float32)
        for col_begin in range(0, len(features), col_batch_size):
            col_end = min(col_begin + col_batch_size, len(features))
            distances[:, col_begin:col_end] = squared_distances(row, features[col_begin:col_end])
        radii[begin:end] = np.partition(distances, k, axis=1)[:, k]
        print(f"radii {end}/{len(features)}", flush=True)
    return radii


def precision_recall(
    real: np.ndarray,
    fake: np.ndarray,
    real_radii: np.ndarray,
    fake_radii: np.ndarray,
    row_batch_size: int,
    col_batch_size: int,
):
    fake_in_real = np.zeros(len(fake), dtype=np.bool_)
    real_in_fake = np.zeros(len(real), dtype=np.bool_)
    for real_begin in range(0, len(real), row_batch_size):
        real_end = min(real_begin + row_batch_size, len(real))
        real_batch = real[real_begin:real_end]
        for fake_begin in range(0, len(fake), col_batch_size):
            fake_end = min(fake_begin + col_batch_size, len(fake))
            distances = squared_distances(real_batch, fake[fake_begin:fake_end])
            fake_in_real[fake_begin:fake_end] |= np.any(
                distances <= real_radii[real_begin:real_end, None], axis=0
            )
            real_in_fake[real_begin:real_end] |= np.any(
                distances <= fake_radii[fake_begin:fake_end][None, :], axis=1
            )
        print(f"precision/recall {real_end}/{len(real)}", flush=True)
    return float(fake_in_real.mean()), float(real_in_fake.mean())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--fake-dir", type=Path)
    parser.add_argument("--real-features", type=Path)
    parser.add_argument("--fake-features", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument(
        "--limit-real",
        type=int,
        default=DEFAULT_REFERENCE_IMAGES,
        help=(
            "Number of real reference images used for Precision/Recall "
            f"(official reference-batch default: {DEFAULT_REFERENCE_IMAGES}). "
            "Use 0 to use all real features."
        ),
    )
    parser.add_argument("--limit-fake", type=int)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--row-batch-size", type=int, default=2048)
    parser.add_argument("--col-batch-size", type=int, default=2048)
    parser.add_argument("--save-real-features", type=Path)
    parser.add_argument("--save-fake-features", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.real_dir is None) == (args.real_features is None):
        raise ValueError("Provide exactly one of --real-dir and --real-features")
    if (args.fake_dir is None) == (args.fake_features is None):
        raise ValueError("Provide exactly one of --fake-dir and --fake-features")
    device = torch.device(args.device)
    real = load_features(args.real_features) if args.real_features else extract_features(
        args.real_dir,
        args.feature_batch_size,
        device,
        None if args.limit_real == 0 else args.limit_real,
    )
    fake = load_features(args.fake_features) if args.fake_features else extract_features(
        args.fake_dir, args.feature_batch_size, device, args.limit_fake
    )
    if args.limit_real != 0:
        if args.limit_real < 1:
            raise ValueError("--limit-real must be 0 (all) or a positive integer")
        if len(real) < args.limit_real:
            raise ValueError(
                f"Requested {args.limit_real} real reference images, but only {len(real)} are available"
            )
        real = real[: args.limit_real]
    if args.save_real_features:
        args.save_real_features.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_real_features, real)
    if args.save_fake_features:
        args.save_fake_features.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_fake_features, fake)
    real_radii = manifold_radii(real, args.k, args.row_batch_size, args.col_batch_size)
    fake_radii = manifold_radii(fake, args.k, args.row_batch_size, args.col_batch_size)
    precision, recall = precision_recall(
        real, fake, real_radii, fake_radii, args.row_batch_size, args.col_batch_size
    )
    result = {
        "metric": "consistency_models_improved_precision_recall",
        "precision": precision,
        "recall": recall,
        "k": args.k,
        "num_real": int(len(real)),
        "num_fake": int(len(fake)),
        "reference_images_default": DEFAULT_REFERENCE_IMAGES,
        "real_limit": int(args.limit_real),
        "feature_dim": int(real.shape[1]),
        "feature_extractor": "torch-fidelity inception-v3-compat pool_3/2048",
        "row_batch_size": args.row_batch_size,
        "col_batch_size": args.col_batch_size,
        "source": "openai/consistency_models/evaluations/evaluator.py",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
