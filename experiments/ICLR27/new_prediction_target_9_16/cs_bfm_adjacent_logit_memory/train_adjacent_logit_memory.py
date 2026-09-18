#!/usr/bin/env python3
"""Train CS-BFM with detached adjacent clean-logit memory."""

import os
import runpy
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["EXPERIMENT_VARIANT"] = "cached_adjacent_logit_memory"
runpy.run_path(
    str(THIS_DIR.parent / "cs_bfm_fromscratch" / "train_bitdance_joint.py"),
    run_name="__main__",
)
