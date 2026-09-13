"""Fixed-seed cached-X0 diagnostics for the full 64-step direct-X0 experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler


PACKED_WIDTH = (16 * 16 * 64) // 8
EVAL_TIMESTEPS = (1, 2, 4, 8, 16, 32, 48, 64)
FIELDS = (
    "step",
    "t",
    "x0_bce",
    "input_ber",
    "pred_x0_ber",
    "correction_recall",
    "introduced_error_rate",
    "correction_precision",
    "changed_rate",
)


class PackedX0Dataset(Dataset):
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.array = np.load(self.path, mmap_mode="r")
        if (
            self.array.ndim != 2
            or self.array.shape[1] != PACKED_WIDTH
            or self.array.dtype != np.uint8
        ):
            raise RuntimeError(
                f"invalid packed X0 cache: shape={self.array.shape}, dtype={self.array.dtype}"
            )

    def __len__(self):
        return int(self.array.shape[0])

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.array[index], copy=True))


def unpack_x0(packed):
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts), 1
    )
    return bits.reshape(packed.shape[0], 256, 64).float()


def make_train_loader(dataset, batch_size, rank, world_size, seed, num_workers=4):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + 200 + rank)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    return loader


def load_eval_x0(dataset, count):
    if count > len(dataset):
        raise ValueError(f"eval count {count} exceeds cache size {len(dataset)}")
    packed = torch.from_numpy(np.array(dataset.array[:count], copy=True))
    return unpack_x0(packed).cpu()


@torch.inference_mode()
def evaluate(sampler, eval_x0, batch_size, seed, device):
    if sampler.p_flip:
        raise RuntimeError("full64 direct-X0 evaluation requires p_flip=False")
    was_training = sampler.training
    sampler.eval()
    rows = []
    try:
        for physical_t in EVAL_TIMESTEPS:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 3_000_000 + physical_t)
            totals = {
                "bce": 0.0,
                "bits": 0,
                "input_errors": 0,
                "pred_errors": 0,
                "corrected": 0,
                "introduced": 0,
                "clean": 0,
                "changed": 0,
            }
            for start in range(0, eval_x0.shape[0], batch_size):
                x0 = eval_x0[start : start + batch_size].to(device)
                t = torch.full(
                    (x0.shape[0],), physical_t, device=device, dtype=torch.long
                )
                x_t = torch.bernoulli(sampler.q_sample(x0, t), generator=generator)
                with torch.cuda.amp.autocast(enabled=True):
                    logits = sampler._denoise_fn(x_t, time_steps=t - 1)
                logits = logits.float()
                pred = logits >= 0.0
                truth = x0.bool()
                corrupted = x_t.bool() != truth
                clean = ~corrupted
                corrected = corrupted & (pred == truth)
                introduced = clean & (pred != truth)
                changed = pred != x_t.bool()
                totals["bce"] += float(
                    F.binary_cross_entropy_with_logits(
                        logits, x0, reduction="sum"
                    ).item()
                )
                totals["bits"] += x0.numel()
                totals["input_errors"] += int(corrupted.sum().item())
                totals["pred_errors"] += int((pred != truth).sum().item())
                totals["corrected"] += int(corrected.sum().item())
                totals["introduced"] += int(introduced.sum().item())
                totals["clean"] += int(clean.sum().item())
                totals["changed"] += int(changed.sum().item())
            rows.append(
                {
                    "t": physical_t,
                    "x0_bce": totals["bce"] / totals["bits"],
                    "input_ber": totals["input_errors"] / totals["bits"],
                    "pred_x0_ber": totals["pred_errors"] / totals["bits"],
                    "correction_recall": totals["corrected"]
                    / max(totals["input_errors"], 1),
                    "introduced_error_rate": totals["introduced"]
                    / max(totals["clean"], 1),
                    "correction_precision": totals["corrected"]
                    / max(totals["changed"], 1),
                    "changed_rate": totals["changed"] / totals["bits"],
                }
            )
    finally:
        sampler.train(was_training)
    return rows


def record(output_dir, sampler, eval_x0, step, batch_size, seed, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fixed_t_metrics.csv"
    old_rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            old_rows = list(csv.DictReader(handle))
        old_rows = [row for row in old_rows if int(row["step"]) != int(step)]
    new_rows = evaluate(sampler, eval_x0, batch_size, seed, device)
    all_rows = old_rows + [{"step": step, **row} for row in new_rows]
    all_rows.sort(key=lambda row: (int(row["step"]), int(row["t"])))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    _write_plot(output_dir / "fixed_t_ber.png", all_rows)
    return new_rows


def _write_plot(path, rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t in EVAL_TIMESTEPS:
        selected = [row for row in rows if int(row["t"]) == t]
        steps = [int(row["step"]) for row in selected]
        axes[0].plot(
            steps,
            [100 * float(row["pred_x0_ber"]) for row in selected],
            marker="o",
            label=f"t={t}",
        )
    axes[0].set_title("Hard direct-X0 BER")
    axes[0].set_ylabel("BER (%)")
    axes[0].legend(ncol=2, fontsize=8)
    for t in (1, 2, 4, 8):
        selected = [row for row in rows if int(row["t"]) == t]
        axes[1].plot(
            [int(row["step"]) for row in selected],
            [100 * float(row["correction_recall"]) for row in selected],
            marker="o",
            label=f"t={t}",
        )
    axes[1].set_title("Low-noise correction recall")
    axes[1].set_ylabel("Recall (%)")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
