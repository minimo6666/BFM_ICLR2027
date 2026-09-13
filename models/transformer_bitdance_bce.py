"""
Controlled BFM ablation: keep the BFM process/loss/sampler unchanged and replace
only TransformerBD's final Linear(768 -> 64) endpoint predictor with a
BitDance-inspired p=1 conditional vector mixer.

Important scientific scope
--------------------------
This module does NOT introduce BitDance's inner rectified-flow objective,
Gaussian corruption, Euler-Maruyama sampling, autoregressive outer model, or a
new reverse process.

Input/output contract remains exactly the BFM denoiser contract:
    idx        : [B, N, 64] noisy binary latent bits
    time_steps : [B]
    output     : [B, N, 64] raw logits

Therefore the existing BFM code still decides the semantics of these logits.
With the baseline --p_flip flag, they are flip logits and the original BFM code
converts them to clean-X0 logits before applying the original BCE/focal loss and
the original analytical reverse sampler.

Why "BitDance-inspired"?
------------------------
BitDance's p=1 binary diffusion head treats the full binary code of one token as
one vector, projects it to a latent representation, conditions the residual
network with AdaLN-style modulation, and predicts the whole vector in parallel.
Here we retain that deterministic vector-mixing architecture but remove the
inner diffusion process so that the BFM training objective is not changed.

This head can make every output bit depend on all 64 current bits and on the
global BFM Transformer context.  However, because the unchanged BFM objective
and sampler still use per-bit Bernoulli logits, this experiment improves
joint-aware REPRESENTATION but does not change the output distribution family
into an explicit non-factorized joint distribution.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerBD


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


class BitDanceAdaSwiGLUBlock(nn.Module):
    """BitDance-style AdaLN + gated SwiGLU residual block for one code vector."""

    def __init__(self, hidden_dim: int, mlp_ratio: float = 1.5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True)
        inner_dim = max(8, int(hidden_dim * mlp_ratio))
        self.w1 = nn.Linear(hidden_dim, inner_dim * 2, bias=True)
        self.w2 = nn.Linear(inner_dim, hidden_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm(x) * (1.0 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + gate * h


class BitDanceDeterministicP1Head(nn.Module):
    """
    Deterministic p=1 adaptation of the BitDance vector head.

    For each spatial latent cell:
      current 64 bits -----------------> dense vector projection --+
                                                                    |
      global BFM Transformer context --> condition projection ------+--> AdaSwiGLU
                                                                    |
                                                condition ----------+--> 64 logits

    There is deliberately NO inner time/noise variable here.  Adding it would
    change the BFM training objective and invalidate this controlled ablation.
    """

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        num_modulators: int = 2,
    ):
        super().__init__()
        if target_dim <= 0 or cond_dim <= 0 or hidden_dim <= 0 or depth <= 0:
            raise ValueError("All dimensions/depth must be positive.")
        if num_modulators <= 0:
            raise ValueError("num_modulators must be positive.")

        self.target_dim = int(target_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_modulators = min(int(num_modulators), self.depth)

        # Full-vector input projection: unlike 64 separate scalar heads, this
        # immediately allows all current bits to interact in the hidden state.
        self.bit_input = nn.Linear(self.target_dim, self.hidden_dim, bias=True)
        self.cond_input = nn.Linear(self.cond_dim, self.hidden_dim, bias=True)

        self.blocks = nn.ModuleList(
            [BitDanceAdaSwiGLUBlock(self.hidden_dim) for _ in range(self.depth)]
        )

        # Shared AdaLN controllers across consecutive blocks, following the
        # parameter-saving pattern of the BitDance vision head.
        self.modulators = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=True)
             for _ in range(self.num_modulators)]
        )

        self.final_norm = nn.LayerNorm(
            self.hidden_dim, eps=1e-6, elementwise_affine=False
        )
        self.final_modulation = nn.Linear(
            self.hidden_dim, self.hidden_dim * 2, bias=True
        )
        self.out = nn.Linear(self.hidden_dim, self.target_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use normal initialization for the main projections so BCE training has
        # a useful gradient path from step 0, while keeping AdaLN gates initially
        # near identity.  This is more suitable for the baseline BCE control than
        # copying BitDance's diffusion-specific zero-output initialization.
        nn.init.normal_(self.bit_input.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bit_input.bias)
        nn.init.normal_(self.cond_input.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cond_input.bias)

        for block in self.blocks:
            nn.init.xavier_uniform_(block.w1.weight)
            nn.init.zeros_(block.w1.bias)
            nn.init.xavier_uniform_(block.w2.weight)
            nn.init.zeros_(block.w2.bias)

        for mod in self.modulators:
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
            # Give the gate a small nonzero starting value so the extra blocks
            # learn immediately instead of being completely dormant.
            with torch.no_grad():
                gate_start = 2 * self.hidden_dim
                mod.bias[gate_start:].fill_(0.1)

        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out.bias)

    def _modulator_index(self, block_index: int) -> int:
        # Evenly map depth blocks to a small number of shared controllers.
        return min(
            self.num_modulators - 1,
            (block_index * self.num_modulators) // self.depth,
        )

    def forward(
        self,
        noisy_bits: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_bits.shape[:-1] != context.shape[:-1]:
            raise ValueError(
                f"noisy_bits/context prefix mismatch: "
                f"{tuple(noisy_bits.shape)} vs {tuple(context.shape)}"
            )
        if noisy_bits.shape[-1] != self.target_dim:
            raise ValueError(
                f"Expected {self.target_dim} bits, got {noisy_bits.shape[-1]}"
            )
        if context.shape[-1] != self.cond_dim:
            raise ValueError(
                f"Expected context dim {self.cond_dim}, got {context.shape[-1]}"
            )

        # BFM bits are {0,1}; BitDance-style binary vectors are naturally
        # represented symmetrically as {-1,+1}.
        signed_bits = noisy_bits.float().mul(2.0).sub(1.0)

        # Flatten only B and spatial-cell dimensions.  Each row is one complete
        # 64-bit code vector, so dense layers mix all bits jointly.
        prefix = signed_bits.shape[:-1]
        bits_flat = signed_bits.reshape(-1, self.target_dim)
        cond_flat = context.reshape(-1, self.cond_dim)

        cond = self.cond_input(cond_flat)
        h = self.bit_input(bits_flat) + cond

        # Non-autoregressive: all 64 output logits are produced in one pass.
        for i, block in enumerate(self.blocks):
            mod = self.modulators[self._modulator_index(i)](F.silu(cond))
            scale, shift, gate = mod.chunk(3, dim=-1)
            h = block(h, scale, shift, gate)

        final_scale, final_shift = self.final_modulation(F.silu(cond)).chunk(
            2, dim=-1
        )
        h = self.final_norm(h) * (1.0 + final_scale) + final_shift
        logits = self.out(h)

        return logits.view(*prefix, self.target_dim)


class TransformerBDBitDanceBCE(TransformerBD):
    """
    Drop-in TransformerBD replacement for the controlled ablation.

    Everything before the final predictor is copied from the baseline
    TransformerBD behavior.  Only the old
        self.head = Linear(bert_n_emb, 64)
    is replaced by BitDanceDeterministicP1Head.
    """

    def __init__(self, H, avg_pooling: bool = False):
        super().__init__(H, avg_pooling=avg_pooling)

        hidden_dim = _env_int("BFM_BITDANCE_BCE_HIDDEN", 256)
        depth = _env_int("BFM_BITDANCE_BCE_DEPTH", 4)
        num_modulators = _env_int("BFM_BITDANCE_BCE_MODULATORS", 2)

        self.head = BitDanceDeterministicP1Head(
            target_dim=self.codebook_size,
            cond_dim=self.n_embd,
            hidden_dim=hidden_dim,
            depth=depth,
            num_modulators=num_modulators,
        )

        self.bitdance_bce_config = {
            "hidden_dim": hidden_dim,
            "depth": depth,
            "num_modulators": num_modulators,
        }

    def forward(self, idx, label=None, time_steps=None):
        # This is intentionally the same spatial Transformer path as the
        # repository's TransformerBD, up to the final prediction head.
        if idx.shape[1] == 0:
            token_embeddings = torch.zeros(
                idx.shape[0],
                0,
                self.n_embd,
                device=idx.device,
                dtype=self.tok_emb.weight.dtype,
            )
        else:
            token_embeddings = (
                (idx.float() - 0.5) * 2.0
            ) @ self.tok_emb.weight

        token_count = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :token_count, :]
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
        context = self.ln_f(x)

        # Same output shape/semantics expected by the untouched BFM code.
        return self.head(idx, context)
