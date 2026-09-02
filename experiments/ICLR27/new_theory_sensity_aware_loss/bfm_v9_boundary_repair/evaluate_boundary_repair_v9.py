#!/usr/bin/env python3
"""Paired validation of V8 vs V9 at the localized terminal boundary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from v9_runtime import (  # noqa: E402
    binary_code_to_sequence,
    build_hparams,
    ensure_device,
    load_autoencoder,
    make_analysis_loader,
    seed_everything,
)
from models.binarylatent_flow_boundary_repair_v9 import (  # noqa: E402
    BinaryDiffusionFlowBoundaryRepairV9,
)
from models.transformer_boundary_repair_v9 import (  # noqa: E402
    TransformerBDBoundaryRepairV9,
)
from train_boundary_repair_v9 import load_v8_into_v9, torch_load  # noqa: E402


METRICS = (
    "teacher_local_brier",
    "endpoint_probability_brier",
    "endpoint_hard_brier",
    "probability_realization_gap",
    "hard_realization_gap",
    "hard_sample_bit_error",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfes", default="64,32")
    parser.add_argument("--branches", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--raw-head", action="store_true", help="use non-EMA head")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def load_head(model, checkpoint: Path, use_ema: bool) -> int:
    payload = torch_load(checkpoint.expanduser().resolve())
    if not isinstance(payload, dict) or payload.get("format") != "bfm_v9_boundary_head_v1":
        raise ValueError(f"Not a V9 boundary checkpoint: {checkpoint}")
    key = "ema_head_state" if use_ema else "head_state"
    model._denoise_fn.terminal_carry.load_state_dict(payload[key], strict=True)
    return int(payload["step"])


@torch.no_grad()
def collect_latents(autoencoder, loader, max_images: int):
    batches = []
    count = 0
    for data in loader:
        image = data[0] if isinstance(data, (tuple, list)) else data
        remaining = max_images - count
        if remaining <= 0:
            break
        image = image[:remaining].to(next(autoencoder.parameters()).device, non_blocking=True)
        code = autoencoder(image, code_only=True).detach()
        x0 = binary_code_to_sequence(code)
        batches.append(x0.cpu())
        count += x0.shape[0]
    return batches


@torch.no_grad()
def evaluate_mode(model, latent_batches, *, nfe, enabled, args):
    model.boundary_nfes = (int(nfe),)
    sums = {name: 0.0 for name in METRICS}
    count = 0
    for batch_index, latent_cpu in enumerate(latent_batches):
        x0 = latent_cpu.to(next(model.parameters()).device, non_blocking=True)
        generator = torch.Generator(device=x0.device)
        # Identical source Xt and all Bernoulli branches for V8/V9 modes.
        generator.manual_seed(args.seed + 100003 * int(nfe) + batch_index)
        with torch.cuda.amp.autocast(enabled=args.amp):
            stats = model.boundary_objective(
                x0,
                branches=args.branches,
                enable_repair=enabled,
                generator=generator,
            )
        weight = x0.shape[0]
        for name in METRICS:
            sums[name] += float(stats[name].item()) * weight
        count += weight
    return dict({name: sums[name] / max(count, 1) for name in METRICS}, images=count)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nfes = tuple(int(item) for item in args.nfes.split(",") if item.strip())
    device = ensure_device(args.device)
    seed_everything(args.seed)

    H = build_hparams(batch_size=args.batch_size)
    denoiser = TransformerBDBoundaryRepairV9(H)
    model = BinaryDiffusionFlowBoundaryRepairV9(H, denoiser, H.codebook_size)
    load_v8_into_v9(model, args.v8_checkpoint.expanduser().resolve())
    trained_step = load_head(model, args.boundary_checkpoint, use_ema=not args.raw_head)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    autoencoder = load_autoencoder(H, args.ae_checkpoint, device)
    loader = make_analysis_loader(
        H=H,
        data_root=args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    latent_batches = collect_latents(autoencoder, loader, args.max_images)

    rows = []
    for nfe in nfes:
        baseline = evaluate_mode(
            model, latent_batches, nfe=nfe, enabled=False, args=args
        )
        repaired = evaluate_mode(
            model, latent_batches, nfe=nfe, enabled=True, args=args
        )
        for mode, values in (("v8_head_off", baseline), ("v9_head_on", repaired)):
            rows.append({"nfe": nfe, "mode": mode, **values})
        rows.append({
            "nfe": nfe,
            "mode": "delta_v9_minus_v8",
            "images": repaired["images"],
            **{name: repaired[name] - baseline[name] for name in METRICS},
        })

    csv_path = args.output / "boundary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("nfe", "mode", "images", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    decisions = {}
    for nfe in nfes:
        base = next(row for row in rows if row["nfe"] == nfe and row["mode"] == "v8_head_off")
        v9 = next(row for row in rows if row["nfe"] == nfe and row["mode"] == "v9_head_on")
        decisions[str(nfe)] = {
            "hard_gap_relative_change": v9["hard_realization_gap"] / max(base["hard_realization_gap"], 1e-12) - 1.0,
            "prob_gap_relative_change": v9["probability_realization_gap"] / max(base["probability_realization_gap"], 1e-12) - 1.0,
            "endpoint_hard_brier_change": v9["endpoint_hard_brier"] - base["endpoint_hard_brier"],
            "teacher_local_brier_abs_change": abs(v9["teacher_local_brier"] - base["teacher_local_brier"]),
            "pass_for_10k_fid": bool(
                v9["hard_realization_gap"] <= 0.5 * base["hard_realization_gap"]
                and v9["endpoint_hard_brier"] < base["endpoint_hard_brier"]
                and abs(v9["teacher_local_brier"] - base["teacher_local_brier"]) < 1e-6
            ),
        }
    report = {
        "trained_step": trained_step,
        "used_ema_head": not args.raw_head,
        "branches": args.branches,
        "max_images": args.max_images,
        "decisions": decisions,
        "csv": str(csv_path),
    }
    with (args.output / "decision.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
