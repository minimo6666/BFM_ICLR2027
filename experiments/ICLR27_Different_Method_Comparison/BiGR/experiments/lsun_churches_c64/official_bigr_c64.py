"""Minimal LSUN/C64 adapters around the unmodified official BiGR model."""

from types import MethodType
from pathlib import Path
import sys

import torch

BIGR_ROOT = Path(__file__).resolve().parents[2]
if str(BIGR_ROOT) not in sys.path:
    sys.path.insert(0, str(BIGR_ROOT))

from llama.gpt import BIGR_models


MODEL_KWARGS = dict(
    block_size=256,
    seq_len=256,
    binary_size=64,
    num_classes=1,
    cls_token_num=1,
    class_dropout_prob=0.0,
    p_flip=True,
    focal=0.0,
    alpha=-1,
    aux=0.0,
    n_repeat=1,
    n_sample_steps=256,
    infer_steps=100,
    sample_temperature=1.0,
    use_adaLN=True,
)


def build_official_bigr_l(**overrides):
    kwargs = dict(MODEL_KWARGS)
    kwargs.update(overrides)
    return BIGR_models["BiGR-L"](**kwargs)


def bind_cfg_off_sampling(model):
    """Make cfg_scale=1 batch-safe without changing official source.

    The official BinaryDiffusion sampler always expects a concatenated
    conditional/unconditional context. For CFG-off, duplicate only the final
    context immediately at that API boundary; both halves are identical.
    Outer latent and mask batches remain B throughout the global Transformer.
    """
    processor = model.diffusion_processor
    official_sample = processor.sample
    official_decode_one_step = model.decode_one_step

    def sample_cfg_safe(self, *args, cond=None, cfg_scale=None, **kwargs):
        if cfg_scale is None or float(cfg_scale) <= 1.0:
            cond = torch.cat([cond, cond], dim=0)
            cfg_scale = 1.0
        return official_sample(*args, cond=cond, cfg_scale=cfg_scale, **kwargs)

    def decode_one_step_cfg_safe(
        self, x, cond_idx, token_all_mask, input_pos, cfg_scale,
        interpolate=None, **sampling_kwargs
    ):
        if float(cfg_scale) > 1.0:
            return official_decode_one_step(
                x, cond_idx, token_all_mask, input_pos, cfg_scale,
                interpolate=interpolate, **sampling_kwargs
            )
        logits, _ = self(
            x,
            cond_idx=cond_idx,
            token_all_mask=token_all_mask,
            input_pos=input_pos,
            cfg_scale=1.0,
            interpolate=interpolate,
        )
        return logits

    processor.sample = MethodType(sample_cfg_safe, processor)
    model.decode_one_step = MethodType(decode_one_step_cfg_safe, model)
    return model


@torch.no_grad()
def generate_cfg_off(
    model,
    batch_size,
    outer_iterations=20,
    inner_steps=100,
    temperature=1.0,
    gumbel_temp=0.01,
):
    model.eval()
    model.infer_steps = int(inner_steps)
    model.sample_temperature = float(temperature)
    labels = torch.zeros(batch_size, dtype=torch.long, device=next(model.parameters()).device)
    samples = model.generate_with_cfg(
        cond=labels,
        max_new_tokens=256,
        cond_padding=1,
        num_iter=int(outer_iterations),
        out_dim=64,
        cfg_scale=1.0,
        cfg_schedule="constant",
        gumbel_temp=float(gumbel_temp),
        gumbel_schedule="constant",
    )
    if tuple(samples.shape) != (batch_size, 256, 64):
        raise RuntimeError(f"Unexpected BiGR sample shape: {tuple(samples.shape)}")
    if not torch.all((samples == 0) | (samples == 1)):
        raise RuntimeError("Official BiGR produced non-binary C64 samples")
    return samples
