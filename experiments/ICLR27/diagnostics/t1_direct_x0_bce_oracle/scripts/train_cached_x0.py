#!/usr/bin/env python3
"""Train the fixed-t=1 direct-X0 oracle from cached deterministic clean bits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
for path in (HERE, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_direct_x0 as base


PACKED_WIDTH = (16 * 16 * 64) // 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--update-freq", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--init-transformerbd-checkpoint", type=Path, default=None)
    parser.add_argument("--ae-load-dir", type=Path, default=PROJECT_ROOT / "logs/BAE_C64")
    parser.add_argument("--ae-load-step", type=int, default=8_100_000)
    return parser.parse_args()


class PackedX0Dataset(Dataset):
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        array = np.load(self.path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != PACKED_WIDTH or array.dtype != np.uint8:
            raise RuntimeError(f"invalid packed X0 cache: shape={array.shape}, dtype={array.dtype}")
        self.length = int(array.shape[0])
        self._array = array

    def __len__(self):
        return self.length

    def _get_array(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self._get_array()[index], copy=True))


def unpack_x0(packed):
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = torch.bitwise_and(torch.bitwise_right_shift(packed.unsqueeze(-1), shifts), 1)
    return bits.reshape(packed.shape[0], 256, 64).float()


def verify_cache(path):
    complete_path = path.parent / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"cache is incomplete: missing {complete_path}")
    metadata = json.loads(complete_path.read_text(encoding="utf-8"))
    if not metadata.get("complete") or not metadata.get("ae_deterministic"):
        raise RuntimeError("cache must be complete and use deterministic BinaryAE quantization")
    if metadata.get("num_samples") != 126227:
        raise RuntimeError(f"expected all 126227 train latents, got {metadata.get('num_samples')}")
    return metadata


def make_loader(dataset, args, rank, world_size):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed, drop_last=True,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + 200 + rank)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        generator=generator,
    )
    return loader, sampler


def load_eval_x0(dataset, count, device):
    packed = dataset._get_array()
    if count > len(packed):
        raise ValueError(f"eval-images={count}, cache-size={len(packed)}")
    tensor = torch.from_numpy(np.array(packed[:count], copy=True)).to(device)
    return unpack_x0(tensor).cpu()


def main():
    from models.transformer import TransformerBD
    from utils.train_utils import NativeScalerWithGradNormCount

    args = parse_args()
    local_rank, rank, world_size = base.distributed_setup()
    device = torch.device("cuda", local_rank)
    if world_size != 2:
        raise RuntimeError(f"expected baseline 2-rank DDP, got {world_size}")
    metadata = verify_cache(args.cache.resolve())
    base.set_seed(args.seed)
    h = base.build_hparams(args)
    h.p_flip = False
    h.deterministic = True

    denoiser = TransformerBD(h)
    init_metadata = None
    if args.init_transformerbd_checkpoint is not None:
        from experiments.ICLR27.diagnostics.masked_bit_pretrain_core import (
            load_mask_pretrained_transformerbd,
        )

        init_metadata = load_mask_pretrained_transformerbd(
            denoiser,
            str(args.init_transformerbd_checkpoint.resolve()),
            map_location="cpu",
        )
    model = DistributedDataParallel(denoiser.to(device).train(), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=args.learning_rate, weight_decay=0.0,
        betas=(0.9, 0.95), eps=1e-8,
    )
    scaler = NativeScalerWithGradNormCount(True, 0)
    dataset = PackedX0Dataset(args.cache)
    loader, sampler = make_loader(dataset, args, rank, world_size)

    eval_x0 = None
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        eval_x0 = load_eval_x0(dataset, args.eval_images, device)
        sanity_x0 = eval_x0[:args.batch_size].to(device)
        sanity_generator = torch.Generator(device=device)
        sanity_generator.manual_seed(args.seed + 900_000)
        sanity_x1 = base.sample_x1(sanity_x0, generator=sanity_generator)
        sanity_values = torch.unique(sanity_x0).cpu().tolist()
        sanity_ber = float((sanity_x1 != sanity_x0).float().mean().item())
        if tuple(sanity_x0.shape[1:]) != (256, 64) or sanity_values != [0.0, 1.0]:
            raise RuntimeError("cached X0 batch sanity check failed")
        print(f"sanity_X0_shape = {list(sanity_x0.shape)}", flush=True)
        print(f"sanity_X0_values = {sanity_values}", flush=True)
        print(f"sanity_t1_input_BER = {sanity_ber}", flush=True)
        print(f"cache = {args.cache.resolve()}", flush=True)
        print(f"cache_samples = {len(dataset)}", flush=True)
        print(f"cache_format = {metadata['format']}", flush=True)
        print("cache_contains = deterministic clean X0 only; no X1", flush=True)
        print("training_input = cached packed binary X0; no RGB; no BinaryAE", flush=True)
        print("AE_instantiated = False", flush=True)
        print("RGB_dataset_instantiated = False", flush=True)
        print(f"learning_rate = {args.learning_rate}", flush=True)
        print("warmup_steps = 0", flush=True)
        print("scheduler = constant", flush=True)
        print("physical_t = 1; network_time = 0", flush=True)
        print("prediction_target = direct_X0; p_flip = False", flush=True)
        print("loss = binary_cross_entropy_with_logits(logits, X0.float())", flush=True)
        if args.init_transformerbd_checkpoint is None:
            print("initialization = random", flush=True)
        else:
            print(
                f"initialization = strict masked-pretrained TransformerBD from "
                f"{args.init_transformerbd_checkpoint.resolve()}",
                flush=True,
            )
            print(
                f"stage1_checkpoint_step = {init_metadata.get('step', 'unknown')}",
                flush=True,
            )
            print("optimizer_and_scaler = freshly initialized", flush=True)
        print("evaluation_split = cached training latents", flush=True)
        print("evaluation_corruption_seed = seed + 1000000 (fixed)", flush=True)
        print(
            f"world_size = {world_size}; per_rank_batch = {args.batch_size}; "
            f"update_freq = {args.update_freq}; global_batch = "
            f"{world_size * args.batch_size * args.update_freq}",
            flush=True,
        )

    base.set_seed(args.seed + 10_000 + rank)
    rows = []
    initial_extra = {"train_loss": float("nan"), "step_seconds": float("nan")}
    base.evaluate_and_record(model, eval_x0, args, device, 0, rows, extra=initial_extra)
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(loader)

    for step in range(1, args.train_steps + 1):
        step_start = time.time()
        update_losses = []
        for micro in range(args.update_freq):
            try:
                packed = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                packed = next(iterator)
            x0 = unpack_x0(packed.to(device, non_blocking=True))
            with torch.no_grad():
                x1 = base.sample_x1(x0)
            network_time = torch.zeros(x0.shape[0], device=device, dtype=torch.long)
            sync_context = model.no_sync() if micro < args.update_freq - 1 else nullcontext()
            with sync_context:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(x1, time_steps=network_time)
                    loss = F.binary_cross_entropy_with_logits(logits, x0)
                    scaled_loss = loss / args.update_freq
                scaler(
                    scaled_loss, optimizer, clip_grad=0.0,
                    parameters=model.module.parameters(), create_graph=False,
                    update_grad=(micro == args.update_freq - 1),
                )
            update_losses.append(float(loss.detach().item()))
        optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0:
            extra = {
                "train_loss": float(np.mean(update_losses)),
                "step_seconds": time.time() - step_start,
            }
            base.evaluate_and_record(model, eval_x0, args, device, step, rows, extra=extra)

    if rank == 0:
        print("COMPLETE t1_direct_x0_bce_oracle_cached_x0", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
