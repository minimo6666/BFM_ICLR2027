"""Parameter-efficient low-rank sensitivity residual head for TransformerBD.

Design goal
-----------
Keep the original TransformerBD backbone and its original t-conditioning exactly
unchanged.  A tiny low-rank residual head consumes DETACHED final spatial
features plus interval metadata (t, s, normalized sensitivity) and predicts an
additive residual in the raw-logit space.

The base BFM loss never uses this residual.  The sensitivity correction loss
never backpropagates through the backbone.  This gives exact gradient isolation
between the base predictor and the low-rank correction branch.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from models.transformer import TransformerBD


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class LowRankIntervalResidual(nn.Module):
    """A tiny conditional low-rank residual head.

    For each spatial token h in R^d:
        z = V h,                     rank r
        z' = z * (1 + tanh(a_c)) + b_c
        delta_logits = U z'

    The conditioner sees only four scalars:
        t/T, s/T, (t-s)/T, S/sqrt(mu_K)

    U is zero-initialized, so the whole model starts EXACTLY at the base BFM
    prediction even though the adapter parameters are present.
    """

    def __init__(self, hidden_dim: int, out_dim: int, num_steps: int, rank: int = 32):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_steps = float(num_steps)
        self.rank = int(rank)

        self.down = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.cond = nn.Sequential(
            nn.Linear(4, 2 * self.rank),
            nn.SiLU(),
            nn.Linear(2 * self.rank, 2 * self.rank),
        )
        self.up = nn.Linear(self.rank, self.out_dim, bias=False)

        # Baseline-preserving initialization: delta_logits == 0 at step 0.
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
        sensitivity_norm: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be [B,L,D], got {tuple(hidden.shape)}")
        b = hidden.shape[0]
        if t_current.shape[0] != b or t_target.shape[0] != b:
            raise ValueError("time tensors must have the same batch size as hidden")

        t = t_current.float() / self.num_steps
        s = t_target.float() / self.num_steps
        delta = (t_current.float() - t_target.float()) / self.num_steps
        sn = sensitivity_norm.float()
        cond_in = torch.stack([t, s, delta, sn], dim=-1)

        scale, shift = self.cond(cond_in).chunk(2, dim=-1)
        z = self.down(hidden.detach())
        z = z * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        return self.up(z)


class TransformerBDLowRankSensitivityV8(TransformerBD):
    """Original TransformerBD + a tiny isolated low-rank residual head.

    `forward()` is intentionally identical in semantics to the original
    TransformerBD: it returns BASE raw logits and does not touch the adapter.
    Therefore the original BFM loss trains exactly the original backbone path.
    """

    def __init__(self, H, avg_pooling: bool = False):
        super().__init__(H, avg_pooling=avg_pooling)
        rank = _env_int("BFM_LR_RANK", 32)
        self.sensitivity_adapter = LowRankIntervalResidual(
            hidden_dim=self.n_embd,
            out_dim=self.codebook_size,
            num_steps=int(H.total_steps),
            rank=rank,
        )
        self.sensitivity_adapter_rank = rank

    def _forward_base_hidden(self, idx, label=None, time_steps=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Copy the current public TransformerBD forward, but also return h.

        `h` is the post-ln_f spatial representation used by the original head.
        The token sequence / attention path is unchanged from TransformerBD.
        """
        if idx.shape[1] == 0:
            token_embeddings = torch.zeros(
                idx.shape[0], 0, self.n_embd, device=idx.device, dtype=self.tok_emb.weight.dtype
            )
        else:
            token_embeddings = (idx * 1.0 - 0.5) * 2.0 @ self.tok_emb.weight

        n_spatial = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :n_spatial, :]
        x = token_embeddings + position_embeddings

        time_emb = self.time_step_embedding(time_steps)
        x = torch.cat([x, time_emb], dim=1)

        if self.exp_type.endswith("tkn") and label is not None:
            cls_emb = self.cls_embedding(label).unsqueeze(1)
            x = torch.cat([x, cls_emb], dim=1)

        x = self.drop(x)
        for block in self.blocks:
            if self.exp_type == "t2i_cross":
                x = block(x, label)
            else:
                x = block(x)

        x = x[:, : self.block_size, :]
        hidden = self.ln_f(x)
        base_logits = self.head(hidden)
        return base_logits, hidden

    def forward(self, idx, label=None, time_steps=None):
        base_logits, _ = self._forward_base_hidden(idx, label=label, time_steps=time_steps)
        return base_logits

    def forward_base_with_hidden(self, idx, label=None, time_steps=None):
        return self._forward_base_hidden(idx, label=label, time_steps=time_steps)

    def adapter_residual(
        self,
        hidden: torch.Tensor,
        t_current: torch.Tensor,
        t_target: torch.Tensor,
        sensitivity_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.sensitivity_adapter(
            hidden=hidden,
            t_current=t_current,
            t_target=t_target,
            sensitivity_norm=sensitivity_norm,
        )

    def adapter_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.sensitivity_adapter.parameters()

    def adapter_parameter_count(self) -> int:
        return sum(p.numel() for p in self.sensitivity_adapter.parameters())
