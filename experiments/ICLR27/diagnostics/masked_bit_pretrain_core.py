"""
Masked-bit pattern pretraining core for BFM / TransformerBD.

Purpose
-------
Stage 1:
    clean cached X0 in {0,1}^{B x 256 x 64}
      -> randomly hide 10% of bits by setting them to 0.5
      -> TransformerBD(time_steps=0)
      -> predict clean X0
      -> BCE only on hidden positions

Why 0.5?
--------
The current TransformerBD forms each spatial token with

    ((idx - 0.5) * 2.0) @ tok_emb.weight

therefore:
    bit 0   -> -1 * channel embedding
    bit 1   -> +1 * channel embedding
    bit 0.5 ->  0 * channel embedding

So 0.5 is a neutral "missing contribution" used ONLY during pretraining.
The BFM state space remains binary {0,1}; no MASK state is introduced into
the Bernoulli flow.

Stage 2:
    load the exact same TransformerBD weights into the existing cached-X0
    t=1 direct-X0 plain-BCE oracle and fine-tune there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerBD


MASK_VALUE = 0.5
T1_FLIP_RATE = 1.0 / 128.0
T1_OBSERVATION_RELIABILITY = 1.0 - T1_FLIP_RATE  # 0.9921875


@dataclass
class MaskedBitConfig:
    mask_ratio: float = 0.10
    network_time: int = 0
    mask_value: float = MASK_VALUE

    def validate(self) -> None:
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0,1), got {self.mask_ratio}")


def sample_bit_mask(
    x0: torch.Tensor,
    mask_ratio: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    x0: [B, 256, 64] (or any same-shape binary tensor)
    returns bool mask of exactly the same shape.

    Training: call every micro-batch => fresh mask.
    Evaluation: pass a fixed-seed generator => reproducible mask.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in (0,1), got {mask_ratio}")
    return torch.rand(
        x0.shape,
        device=x0.device,
        generator=generator,
        dtype=torch.float32,
    ) < mask_ratio


def make_masked_input(
    x0: torch.Tensor,
    bit_mask: torch.Tensor,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """
    Return float input with hidden bits replaced by 0.5.

    IMPORTANT:
      x0 stays the clean {0,1} target.
      Only the network input gets 0.5 at hidden positions.
    """
    if x0.shape != bit_mask.shape:
        raise ValueError(f"shape mismatch: x0={x0.shape}, mask={bit_mask.shape}")
    x_in = x0.float().clone()
    x_in.masked_fill_(bit_mask, float(mask_value))
    return x_in


def masked_bce_with_logits(
    logits: torch.Tensor,
    target_x0: torch.Tensor,
    bit_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Proper BCE, but ONLY hidden positions contribute to Stage-1 training.

    No focal.
    No alpha.
    No positive-class weighting.
    No sensitivity weighting.
    """
    if logits.shape != target_x0.shape or logits.shape != bit_mask.shape:
        raise ValueError(
            f"shape mismatch: logits={logits.shape}, "
            f"target={target_x0.shape}, mask={bit_mask.shape}"
        )

    per_bit = F.binary_cross_entropy_with_logits(
        logits,
        target_x0.float(),
        reduction="none",
    )
    denom = bit_mask.sum().clamp_min(1)
    return (per_bit * bit_mask.float()).sum() / denom


@torch.no_grad()
def masked_metrics(
    logits: torch.Tensor,
    target_x0: torch.Tensor,
    bit_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Metrics on hidden positions only.

    strong_context_fraction:
        fraction of hidden positions where the model gives the correct clean bit
        probability > 0.9921875.

    This threshold is scientifically useful for t=1:
        the observed bit itself has reliability 1 - 1/128 = 0.9921875.
    A context-only predictor must become extremely confident before it can
    plausibly overturn such a reliable observed bit.
    """
    target = target_x0.bool()
    pred = logits >= 0.0
    probs = torch.sigmoid(logits)

    m = bit_mask.bool()
    n = int(m.sum().item())
    if n == 0:
        return {
            "masked_ber": float("nan"),
            "masked_accuracy": float("nan"),
            "masked_gt_confidence": float("nan"),
            "strong_context_fraction": float("nan"),
        }

    err = (pred != target) & m
    masked_ber = err.sum().float() / m.sum().float()

    p_gt = torch.where(target, probs, 1.0 - probs)
    gt_conf = p_gt[m]

    return {
        "masked_ber": float(masked_ber.item()),
        "masked_accuracy": float((1.0 - masked_ber).item()),
        "masked_gt_confidence": float(gt_conf.mean().item()),
        "strong_context_fraction": float(
            (gt_conf > T1_OBSERVATION_RELIABILITY).float().mean().item()
        ),
    }


class MaskedBitPatternPretrainer(nn.Module):
    """
    Thin wrapper around the ORIGINAL TransformerBD.

    Crucially, no architecture change is made to TransformerBD, so the Stage-1
    denoiser state_dict can be loaded strictly into the Stage-2 direct-X0 oracle.
    """

    def __init__(self, H, cfg: Optional[MaskedBitConfig] = None):
        super().__init__()
        self.cfg = cfg or MaskedBitConfig()
        self.cfg.validate()
        self.denoiser = TransformerBD(H)

    def forward(
        self,
        x0: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        bit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x0: clean deterministic latent, shape [B,256,64], values {0,1}.
        """
        if x0.ndim != 3 or x0.shape[-1] != 64:
            raise ValueError(f"expected [B,256,64], got {tuple(x0.shape)}")

        if bit_mask is None:
            bit_mask = sample_bit_mask(
                x0,
                self.cfg.mask_ratio,
                generator=generator,
            )

        x_masked = make_masked_input(
            x0,
            bit_mask,
            mask_value=self.cfg.mask_value,
        )

        time_steps = torch.full(
            (x0.shape[0],),
            fill_value=int(self.cfg.network_time),
            device=x0.device,
            dtype=torch.long,
        )

        # Exact original TransformerBD; clean-X0 logits [B,256,64].
        logits = self.denoiser(
            x_masked,
            time_steps=time_steps,
        )

        loss = masked_bce_with_logits(
            logits=logits,
            target_x0=x0,
            bit_mask=bit_mask,
        )

        return {
            "loss": loss,
            "logits": logits,
            "bit_mask": bit_mask,
            "x_masked": x_masked,
        }


def export_transformerbd_state(
    model: nn.Module,
    path: str,
    *,
    extra: Optional[dict] = None,
) -> None:
    """
    Save ONLY the underlying TransformerBD weights.

    Works with a bare MaskedBitPatternPretrainer or a DDP-wrapped one.
    """
    raw = model.module if hasattr(model, "module") else model
    if not isinstance(raw, MaskedBitPatternPretrainer):
        raise TypeError(f"expected MaskedBitPatternPretrainer, got {type(raw)}")

    payload = {
        "transformerbd": raw.denoiser.state_dict(),
        "stage": "masked_bit_pattern_pretrain",
        "mask_ratio": raw.cfg.mask_ratio,
        "network_time": raw.cfg.network_time,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_mask_pretrained_transformerbd(
    denoiser: TransformerBD,
    checkpoint_path: str,
    *,
    map_location: str = "cpu",
) -> dict:
    """
    Stage 2: load Stage-1 weights into the ordinary TransformerBD.

    strict=True is intentional. If it fails, do NOT silently continue: it means
    Stage 1 and Stage 2 architectures/configs are not identical.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt["transformerbd"] if "transformerbd" in ckpt else ckpt
    denoiser.load_state_dict(state, strict=True)
    return ckpt


# ---------------------------------------------------------------------------
# Minimal training-step examples for Codex integration
# ---------------------------------------------------------------------------

def stage1_masked_pretrain_step(
    model: MaskedBitPatternPretrainer,
    x0: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Dataset/optimizer/DDP are intentionally left to the existing trainer.
    """
    out = model(x0)
    metrics = masked_metrics(
        out["logits"].detach(),
        x0,
        out["bit_mask"],
    )
    return out["loss"], metrics


def make_fresh_t1_corruption(x0: torch.Tensor) -> torch.Tensor:
    """
    Standalone sanity helper only.

    In the real Stage-2 oracle, PREFER the exact existing BFM t=1 corruption
    routine already used by your current diagnostic, rather than duplicating it.
    """
    flips = torch.rand_like(x0.float()) < T1_FLIP_RATE
    return torch.logical_xor(x0.bool(), flips).to(x0.dtype)


def stage2_t1_direct_x0_step(
    denoiser: TransformerBD,
    x0: torch.Tensor,
    *,
    x1: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference Stage-2 objective:
      cached clean X0
      -> fresh exact t=1 corruption X1
      -> direct clean-X0 logits
      -> plain BCE over ALL bits

    For production integration, pass x1 from the existing exact BFM t=1
    corruption routine.
    """
    if x1 is None:
        x1 = make_fresh_t1_corruption(x0)

    time_steps = torch.zeros(
        x0.shape[0],
        device=x0.device,
        dtype=torch.long,
    )
    logits = denoiser(x1.float(), time_steps=time_steps)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        x0.float(),
        reduction="mean",
    )
    return loss, logits
