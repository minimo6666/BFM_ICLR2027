#!/usr/bin/env python3
"""Summarize the paired V9 head-off/head-on FID screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fid_from_json(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["metrics"]["frechet_inception_distance"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for nfe in (64, 32):
        v8 = fid_from_json(args.v9_root / "fid" / f"nfe_{nfe:03d}_v9_head_off.json")
        v9 = fid_from_json(args.v9_root / "fid" / f"nfe_{nfe:03d}_v9_head_on.json")
        rows.append({"nfe": nfe, "v8_alpha1_fid": v8, "v9_fid": v9, "delta": v9 - v8})

    summary = {
        "reference": "V9 head-off, bit-exact current V8 sampler, paired seed protocol",
        "rows": rows,
    }
    with (args.v9_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    lines = [
        "# BFM V9 boundary repair — 10k FID",
        "",
        "Reference: V9 head-off, verified bit-exact to the current V8 sampler.",
        "",
        "| NFE | V9 head-off FID | V9 head-on FID | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['nfe']} | {row['v8_alpha1_fid']:.6f} | {row['v9_fid']:.6f} | {row['delta']:+.6f} |"
        )
    (args.v9_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
