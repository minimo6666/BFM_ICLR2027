#!/usr/bin/env python3
"""Generate LSUN Churches with corrected samplers applied to a trained Binary DFM.

The checkpoint is the *same* dfm_binary EMA checkpoint used by the original
Euler baseline.  No corrected-sampler retraining is required.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DFM_DIR = SCRIPT_DIR.parent / "Discrete_Flow_Matching"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DFM_DIR))

from hparams.defaults.binarygan_default import HparamsBinaryAE  # noqa: E402
from hparams.defaults.sampler_defaults import HparamsBianryLatent  # noqa: E402
from models.binaryae import Generator  # noqa: E402
from models.transformer import TransformerBD  # noqa: E402
from binarylatent_dfm_corrected_samplers import BinaryCorrectedDFMDecouple  # noqa: E402
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ae_load_dir", type=Path, required=True)
    p.add_argument("--ae_load_step", type=int, default=8100000)
    p.add_argument("--model_total_steps", type=int, default=64)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--samplers",
        nargs="+",
        default=["paper_euler", "time_corrected", "location_corrected"],
        choices=["paper_euler", "time_corrected", "location_corrected"],
    )
    p.add_argument(
        "--sampling_steps",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help=(
            "Integration steps for every sampler. Location correction can use "
            "two denoiser evaluations per step."
        ),
    )
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--uniform_grid", action="store_true")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--sample_batch_size", type=int, default=10)
    p.add_argument("--decode_batch_size", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_stride", type=int, default=10)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress_every", type=int, default=10)
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    return p.parse_args()


def build_hparams(args):
    H = HparamsBinaryAE("churches")
    H.vqgan_batch_size = H.batch_size
    H.update(HparamsBianryLatent("churches"))
    H.sampler = "dfm_binary"
    H.dataset = "churches"
    H.codebook_size = 64
    H.img_size = 256
    H.latent_shape = [1, 16, 16]
    H.total_steps = args.model_total_steps
    H.sample_steps = args.model_total_steps
    H.beta_type = "linear"
    H.loss_final = "mean"
    H.p_flip = False
    H.focal = -1
    H.aux = 0.0
    H.norm_first = True
    H.batch_size = args.sample_batch_size
    H.amp = args.amp
    H.ema = True
    H.ae_load_dir = str(args.ae_load_dir.resolve())
    H.ae_load_step = args.ae_load_step
    H.dfm_stochasticity = 0.0
    H.dfm_continuous_time = True
    return H


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_process_shard(num_samples, rank, world_size):
    q, r = divmod(num_samples, world_size)
    size = q + int(rank < r)
    start = rank * q + min(rank, r)
    return start, start + size


def is_valid_rgb_png(path):
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.mode == "RGB" and im.size == (256, 256)
    except (OSError, ValueError):
        return False


def load_models(H, checkpoint):
    ae_state = retrieve_autoencoder_components_state_dicts(
        H, ["quantize", "generator"], remove_component_from_key=True
    )
    embedding_weight = ae_state.pop("embed.weight").cuda()
    generator = Generator(H)
    generator.load_state_dict(ae_state, strict=False)
    generator = generator.cuda().eval()
    del ae_state

    denoise_fn = TransformerBD(H).cuda()
    sampler = BinaryCorrectedDFMDecouple(H, denoise_fn, H.codebook_size).cuda()
    state_dict = torch.load(checkpoint, map_location="cpu")
    incompatible = sampler.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    sampler.eval()
    return sampler, generator, embedding_weight


@torch.no_grad()
def decode_latents(H, embedding_weight, generator, latents, batch_size, amp):
    out = []
    for latent in torch.split(latents, batch_size):
        latent = latent.float()
        if H.use_tanh:
            latent = (latent - 0.5) * 2.0
        if not H.norm_first:
            latent = latent / float(H.codebook_size)
        latent = latent @ embedding_weight
        latent = latent.permute(0, 2, 1).reshape(
            latent.shape[0], H.emb_dim, H.latent_shape[1], H.latent_shape[2]
        )
        with torch.cuda.amp.autocast(enabled=amp):
            out.append(generator(latent).float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def generate_setting(H, sampler, generator, embedding_weight, args, method, nfe, shard):
    start_idx, end_idx = shard
    shard_size = end_idx - start_idx
    step_dir = args.output_dir / method / str(nfe)
    step_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for batch_index, offset in enumerate(range(0, shard_size, args.sample_batch_size)):
        bs = min(args.sample_batch_size, shard_size - offset)
        first = start_idx + offset
        paths = [step_dir / f"{i:05d}.png" for i in range(first, first + bs)]

        if args.resume and all(is_valid_rgb_png(p) for p in paths):
            continue

        batch_seed = args.seed + args.rank + batch_index * args.seed_stride
        set_seed(batch_seed)

        with torch.cuda.amp.autocast(enabled=args.amp):
            latents = sampler.sample_corrected(
                method=method,
                sample_steps=nfe,
                temp=args.temperature,
                b=bs,
                return_all=False,
                delta=args.delta,
                opt_grid=not args.uniform_grid,
            )

        if latents.shape != (bs, 256, 64):
            raise RuntimeError(f"Unexpected latent shape: {tuple(latents.shape)}")
        unique = torch.unique(latents)
        if not torch.all((unique == 0) | (unique == 1)):
            raise RuntimeError(f"Non-binary corrected DFM output: {unique.tolist()}")

        images = decode_latents(
            H, embedding_weight, generator, latents,
            args.decode_batch_size, args.amp
        )
        for path, image in zip(paths, images):
            tmp = path.with_name(f".{path.name}.tmp.png")
            save_image(image.clamp(0, 1), tmp)
            os.replace(tmp, path)

        if (
            batch_index % args.progress_every == 0
            or offset + bs == shard_size
        ):
            done = offset + bs
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0.0
            print(
                f"[{method}] steps={nfe} generated={done}/{shard_size} "
                f"rate={rate:.2f} img/s seed={batch_seed} "
                f"stats={sampler.last_sampling_stats}",
                flush=True,
            )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if any(n <= 0 for n in args.sampling_steps):
        raise ValueError("sampling_steps must be positive.")
    if not (0.0 < args.delta < 1.0):
        raise ValueError("--delta must lie in (0,1).")

    args.checkpoint = args.checkpoint.resolve()
    args.ae_load_dir = args.ae_load_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    H = build_hparams(args)
    sampler, generator, embedding_weight = load_models(H, args.checkpoint)
    shard = get_process_shard(args.num_samples, args.rank, args.world_size)
    print(
        f"Loaded SAME Binary DFM EMA checkpoint: {args.checkpoint}\n"
        f"samplers={args.samplers}, steps={args.sampling_steps}, "
        f"delta={args.delta}, opt_grid={not args.uniform_grid}, "
        f"shard={shard}",
        flush=True,
    )

    for method in args.samplers:
        for nfe in args.sampling_steps:
            # Re-seeding inside generate_setting makes same global image indices
            # paired across methods/NFEs.
            generate_setting(
                H, sampler, generator, embedding_weight,
                args, method, nfe, shard
            )


if __name__ == "__main__":
    main()
