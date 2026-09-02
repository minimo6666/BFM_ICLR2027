"""V9 terminal boundary-repair head for the V8 sensitivity model.

The V8 backbone and sensitivity adapter are reused without modification.  This
module adds one zero-initialized low-rank head at physical time ``t=1``.  The
head receives the terminal hidden state together with the clean posterior that
was predicted immediately before the last probabilistic transition.  It emits
an additive residual in *clean-X0 logit* space.

Zero initialization is important: before any V9 training, enabling this module
is exactly the released V8 sampler (including its hard terminal projection).
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn

from models.transformer_lowrank_sensitivity_v8 import (
    TransformerBDLowRankSensitivityV8,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class TerminalPosteriorCarryHead(nn.Module):
    """A small residual head that carries the preterminal posterior to t=1.

    Inputs
    ------
    terminal_hidden:
        Frozen V8 terminal features, shape ``[B,L,D]``.
    source_clean_prob:
        Detached adapter-corrected ``P(X0=1|Xt)`` from the last probabilistic
        source node, shape ``[B,L,C]``.
    source_t, sensitivity_norm, adapter_scale:
        Boundary metadata.  Conditioning on the actual source node prevents a
        single correction from being forced onto 64/32/16/8-NFE boundaries.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        out_dim: int,
        num_steps: int,
        rank: int = 32,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_steps = float(num_steps)
        self.rank = int(rank)

        self.hidden_down = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.posterior_down = nn.Linear(self.out_dim, self.rank, bias=False)
        self.conditioner = nn.Sequential(
            nn.Linear(4, 2 * self.rank),
            nn.SiLU(),
            nn.Linear(2 * self.rank, 2 * self.rank),
        )
        self.fuse_norm = nn.LayerNorm(self.rank)
        self.up = nn.Linear(self.rank, self.out_dim, bias=False)

        # Exact V8 identity at initialization.
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        *,
        terminal_hidden: torch.Tensor,
        source_clean_prob: torch.Tensor,
        source_t: torch.Tensor,
        sensitivity_norm: torch.Tensor,
        adapter_scale: torch.Tensor,
    ) -> torch.Tensor:
        if terminal_hidden.ndim != 3 or source_clean_prob.ndim != 3:
            raise ValueError("terminal_hidden and source_clean_prob must be [B,L,D/C]")
        if terminal_hidden.shape[:2] != source_clean_prob.shape[:2]:
            raise ValueError("terminal features and source posterior must share [B,L]")
        batch = terminal_hidden.shape[0]
        for name, tensor in (
            ("source_t", source_t),
            ("sensitivity_norm", sensitivity_norm),
            ("adapter_scale", adapter_scale),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != batch:
                raise ValueError(f"{name} must have shape [B]")

        source_probability = source_clean_prob.float().clamp(1.0e-5, 1.0 - 1.0e-5)
        source_logit = torch.logit(source_probability)
        t_normalized = source_t.float() / self.num_steps
        delta_normalized = (source_t.float() - 1.0) / self.num_steps
        condition = torch.stack(
            [
                t_normalized,
                delta_normalized,
                sensitivity_norm.float(),
                adapter_scale.float(),
            ],
            dim=-1,
        )
        scale, shift = self.conditioner(condition).chunk(2, dim=-1)

        z = self.hidden_down(terminal_hidden.detach())
        z = z + self.posterior_down(source_logit.detach())
        z = self.fuse_norm(z)
        z = z * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        z = torch.nn.functional.silu(z)
        return self.up(z)


class TransformerBDBoundaryRepairV9(TransformerBDLowRankSensitivityV8):
    """V8 transformer plus an isolated terminal posterior-carry head."""

    def __init__(self, H, avg_pooling: bool = False):
        super().__init__(H, avg_pooling=avg_pooling)
        rank = _env_int("BFM_V9_BOUNDARY_RANK", 32)
        self.terminal_carry = TerminalPosteriorCarryHead(
            hidden_dim=self.n_embd,
            out_dim=self.codebook_size,
            num_steps=int(H.total_steps),
            rank=rank,
        )
        self.terminal_carry_rank = rank

    def terminal_carry_residual(
        self,
        *,
        terminal_hidden: torch.Tensor,
        source_clean_prob: torch.Tensor,
        source_t: torch.Tensor,
        sensitivity_norm: torch.Tensor,
        adapter_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.terminal_carry(
            terminal_hidden=terminal_hidden,
            source_clean_prob=source_clean_prob,
            source_t=source_t,
            sensitivity_norm=sensitivity_norm,
            adapter_scale=adapter_scale,
        )

    def boundary_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.terminal_carry.parameters()

    def boundary_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.terminal_carry.parameters())
