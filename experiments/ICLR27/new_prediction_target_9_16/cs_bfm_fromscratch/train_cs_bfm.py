#!/usr/bin/env python3
"""Thin entry that selects the isolated cached CS-BFM training variant."""

import os
import runpy
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["EXPERIMENT_VARIANT"] = "cached_cs_bfm"
runpy.run_path(str(THIS_DIR / "train_bitdance_joint.py"), run_name="__main__")
