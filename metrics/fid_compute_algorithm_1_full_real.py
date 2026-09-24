#!/usr/bin/env python3
"""Dataset-aware implementation of the repository's full-real Algorithm 1 FID.

Algorithm 1 keeps the original evaluator and preprocessing while comparing
against an entire real training set:

* fake input: top-level PNG files in one directory;
* real input: full LSUN Churches/Bedrooms or full FFHQ ImageFolder;
* preprocessing: ``Resize(256) -> CenterCrop(256) -> ToTensor``;
* both inputs are presented to torch-fidelity as uint8 RGB images;
* evaluator: torch-fidelity InceptionV3 FID; KID and IS are disabled.

The default remains LSUN Churches for backward compatibility. Select the real
distribution explicitly with ``--dataset`` and ``--data-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


PathLike = Union[str, os.PathLike]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LSUN_ROOT = Path("/mnt/data/0/mohao/data/lsun/scenes")
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "metrics" / "fid_cache_algorithm_1"
DATASET_CHOICES = ("churches", "bedrooms", "ffhq")




def _top_level_pngs(image_dir: Path) -> Sequence[Path]:
    """Return the exact fake-image set consumed by Algorithm 1."""

    return sorted(path for path in image_dir.glob("*.png") if path.is_file())


def _fake_cache_name(image_dir: Path, image_paths: Sequence[Path]) -> str:
    """Build a cache key that changes when the directory contents change.

    torch-fidelity does not validate a user-supplied cache name against the
    files.  Including names, sizes, and mtimes prevents accidental reuse after
    a resumed or replaced generation run without hashing all image pixels.
    """

    digest = hashlib.sha256()
    digest.update(str(image_dir).encode("utf-8"))
    for path in image_paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return f"fid-algorithm-1-fake-{len(image_paths)}-{digest.hexdigest()[:16]}"


def compute_fid_for_folder(
    image_dir: PathLike,
    *,
    dataset: str = "churches",
    data_root: PathLike = DEFAULT_LSUN_ROOT,
    expected_num_fake: Optional[int] = None,
    fid_batch_size: int = 64,
    cache_root: PathLike = DEFAULT_CACHE_ROOT,
    fake_cache_name: Optional[str] = None,
    use_cache: bool = True,
    cuda: bool = True,
    result_json: Optional[PathLike] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute the current project FID for a generated-image folder.

    Parameters
    ----------
    image_dir:
        Directory containing generated ``*.png`` files directly at its top
        level.  Subdirectories are intentionally ignored to match the current
        evaluation code.
    dataset / data_root:
        Complete real training distribution. Churches and bedrooms use the
        corresponding torchvision LSUN split; FFHQ uses an ImageFolder root.
    expected_num_fake:
        Optional strict count check.  This is recommended for final tables so
        partially generated folders cannot yield a plausible-looking FID.
    fid_batch_size:
        Batch size used for Inception feature extraction.
    cache_root / fake_cache_name / use_cache:
        torch-fidelity cache controls.  If no fake cache name is supplied, a
        metadata fingerprint of the folder is used to avoid stale cache hits.
    cuda:
        Use CUDA for feature extraction.  Select the GPU with
        ``CUDA_VISIBLE_DEVICES`` before calling this function.
    result_json:
        Optional output JSON.  No result file is written when omitted.

    Returns
    -------
    dict
        Protocol metadata plus torch-fidelity's metric dictionary.  The FID is
        available at ``result["metrics"]["frechet_inception_distance"]``.
    """

    # Imports stay local so callers can inspect ``--help`` or import this
    # module in CPU-only tooling without eagerly initializing PyTorch/CUDA.
    import torch
    import torch_fidelity
    import torchvision
    from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

    class ImagesOnlyUint8(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset):
            self.dataset = dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index: int) -> torch.Tensor:
            image = self.dataset[index][0]
            return image.mul(255).clamp_(0, 255).to(torch.uint8)

    image_dir = Path(image_dir).expanduser().resolve()
    dataset = str(dataset).lower()
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got {dataset!r}")
    data_root = Path(data_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Generated-image directory not found: {image_dir}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Real-data root not found: {data_root}")
    if expected_num_fake is not None and expected_num_fake <= 0:
        raise ValueError("expected_num_fake must be positive when provided")
    if fid_batch_size <= 0:
        raise ValueError("fid_batch_size must be positive")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but torch.cuda.is_available() is false")

    image_paths = _top_level_pngs(image_dir)
    num_fake = len(image_paths)
    if num_fake == 0:
        raise RuntimeError(f"No top-level PNG images found in {image_dir}")
    if expected_num_fake is not None and num_fake != expected_num_fake:
        raise RuntimeError(
            f"Expected {expected_num_fake} top-level PNGs in {image_dir}, "
            f"found {num_fake}"
        )

    transform = Compose([Resize(256), CenterCrop(256), ToTensor()])
    if dataset == "churches":
        real_dataset = torchvision.datasets.LSUN(
            str(data_root), classes=["church_outdoor_train"], transform=transform
        )
        real_split_name = "LSUN church_outdoor_train"
    elif dataset == "bedrooms":
        real_dataset = torchvision.datasets.LSUN(
            str(data_root), classes=["bedroom_train"], transform=transform
        )
        real_split_name = "LSUN bedroom_train"
    else:
        real_dataset = torchvision.datasets.ImageFolder(
            str(data_root), transform=transform
        )
        real_split_name = "FFHQ ImageFolder"
    num_real = len(real_dataset)
    if num_real == 0:
        raise RuntimeError(f"{real_split_name} is empty")
    real_dataset = ImagesOnlyUint8(real_dataset)

    cache_root.mkdir(parents=True, exist_ok=True)
    if fake_cache_name is None:
        fake_cache_name = _fake_cache_name(image_dir, image_paths)
    real_cache_name = f"{dataset}-train-full-{num_real}-resize256-centercrop"

    started = time.time()
    metrics = torch_fidelity.calculate_metrics(
        input1=str(image_dir),
        input2=real_dataset,
        cuda=cuda,
        batch_size=fid_batch_size,
        fid=True,
        isc=False,
        kid=False,
        samples_find_deep=False,
        samples_ext_lossy="",
        cache=use_cache,
        cache_root=str(cache_root),
        input1_cache_name=fake_cache_name,
        input2_cache_name=real_cache_name,
        verbose=verbose,
    )

    result: Dict[str, Any] = {
        "algorithm": "fid_compute_algorithm_1_full_real",
        "dataset": dataset,
        "fake_images": str(image_dir),
        "num_fake": num_fake,
        "num_real": num_real,
        "real_root": str(data_root),
        "real_split": f"{real_split_name} full split ({num_real} images)",
        "preprocess": "Resize(256), CenterCrop(256), ToTensor, uint8",
        "fake_scan": "top-level *.png only",
        "evaluator": "torch_fidelity.calculate_metrics",
        "torch_fidelity": getattr(torch_fidelity, "__version__", "unknown"),
        "fid_batch_size": fid_batch_size,
        "cuda": cuda,
        "cache_enabled": use_cache,
        "cache_root": str(cache_root),
        "fake_cache_name": fake_cache_name,
        "real_cache_name": real_cache_name,
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
    }

    if result_json is not None:
        result_path = Path(result_json).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Algorithm 1 FID against a full real training split."
        )
    )
    parser.add_argument(
        "image_dir",
        help="Folder containing generated PNG files directly at the top level",
    )
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, default="churches",
        help="Real reference dataset (default: churches)",
    )
    parser.add_argument(
        "--data-root", "--lsun-root", dest="data_root",
        default=str(DEFAULT_LSUN_ROOT),
        help=f"Real dataset root (default: {DEFAULT_LSUN_ROOT})",
    )
    parser.add_argument(
        "--expected-num-fake",
        type=int,
        default=None,
        help="Fail unless the folder contains exactly this many top-level PNGs",
    )
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--fake-cache-name", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--result-json",
        default=None,
        help=(
            "Output JSON path. Default: a sibling file named "
            "<image-folder>_fid_algorithm_1_full_real.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    result_json = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json is not None
        else image_dir.parent / f"{image_dir.name}_fid_algorithm_1_full_real.json"
    )
    result = compute_fid_for_folder(
        image_dir,
        dataset=args.dataset,
        data_root=args.data_root,
        expected_num_fake=args.expected_num_fake,
        fid_batch_size=args.fid_batch_size,
        cache_root=args.cache_root,
        fake_cache_name=args.fake_cache_name,
        use_cache=not args.no_cache,
        cuda=not args.cpu,
        result_json=result_json,
        verbose=not args.quiet,
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"Result written to: {result_json}", flush=True)


if __name__ == "__main__":
    main()
