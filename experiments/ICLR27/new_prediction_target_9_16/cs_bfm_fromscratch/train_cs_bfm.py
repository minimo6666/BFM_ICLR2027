#!/usr/bin/env python3
"""Thin entry that selects the isolated cached CS-BFM training variant."""

import os
import runpy
from pathlib import Path

ROOT = Path("/mnt/data/b/mohao/Projects/BinaryLatentDiffusion")
os.environ["EXPERIMENT_VARIANT"] = "cached_cs_bfm"
runpy.run_path(
    str(ROOT / "experiments/ICLR27/src_controlled_v4/train/train_bitdance_joint.py"),
    run_name="__main__",
)
