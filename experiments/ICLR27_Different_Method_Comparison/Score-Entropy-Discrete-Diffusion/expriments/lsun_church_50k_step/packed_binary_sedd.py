"""Packed binary-coordinate parameterization for the official SEDD DDiT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import DDitFinalLayer, SEDD


class PackedBinarySEDD(SEDD):
    """SEDD model with 16 binary coordinates packed into each DDiT token.

    The public interface is unchanged from the official model:
    ``[B, 16384] -> [B, 16384, 2]``. Internally, the 64x16x16 BAE
    representation is arranged as 1024 tokens of width 768.
    """

    CHANNELS = 64
    HEIGHT = 16
    WIDTH = 16
    BITS_PER_TOKEN = 16
    CHANNEL_GROUPS = CHANNELS // BITS_PER_TOKEN
    NUM_COORDINATES = CHANNELS * HEIGHT * WIDTH
    NUM_TOKENS = HEIGHT * WIDTH * CHANNEL_GROUPS

    def __init__(self, config):
        if config.tokens != 2:
            raise ValueError("PackedBinarySEDD requires tokens=2")
        if config.graph.type != "uniform":
            raise ValueError("PackedBinarySEDD requires the Uniform graph")

        super().__init__(config)
        hidden_size = config.model.hidden_size
        if hidden_size != 768:
            raise ValueError("PackedBinarySEDD expects hidden_size=768")
        if hidden_size % self.BITS_PER_TOKEN:
            raise ValueError("hidden_size must be divisible by BITS_PER_TOKEN")

        self.bit_embedding_dim = hidden_size // self.BITS_PER_TOKEN
        self.bit_embedding = nn.Embedding(self.CHANNELS * 2, self.bit_embedding_dim)
        nn.init.kaiming_uniform_(self.bit_embedding.weight, a=math.sqrt(5))
        del self.vocab_embed

        self.output_layer = DDitFinalLayer(
            hidden_size,
            self.BITS_PER_TOKEN * 2,
            config.model.cond_dim,
        )
        offsets = (2 * torch.arange(self.CHANNELS)).reshape(1, 1, 1, self.CHANNELS)
        self.register_buffer("channel_offsets", offsets, persistent=False)

    def _pack_input(self, indices):
        if indices.ndim != 2 or indices.shape[1] != self.NUM_COORDINATES:
            raise ValueError(
                f"Expected [B,{self.NUM_COORDINATES}], got {tuple(indices.shape)}"
            )

        batch_size = indices.shape[0]
        bits = indices.reshape(batch_size, self.HEIGHT, self.WIDTH, self.CHANNELS)
        embedded = self.bit_embedding(bits + self.channel_offsets)
        embedded = embedded.reshape(
            batch_size,
            self.HEIGHT,
            self.WIDTH,
            self.CHANNEL_GROUPS,
            self.BITS_PER_TOKEN,
            self.bit_embedding_dim,
        )
        return embedded.reshape(batch_size, self.NUM_TOKENS, -1)

    def forward(self, indices, sigma):
        if indices.dtype != torch.long:
            indices = indices.long()
        batch_size = indices.shape[0]
        x = self._pack_input(indices)
        c = F.silu(self.sigma_map(sigma.reshape(-1)))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)

        x = x.reshape(
            batch_size,
            self.HEIGHT,
            self.WIDTH,
            self.CHANNEL_GROUPS,
            self.BITS_PER_TOKEN,
            2,
        )
        x = x.reshape(batch_size, self.NUM_COORDINATES, 2)

        # SEDD scores are normalized so the score of the observed state is 1
        # in ratio space, i.e. zero in log space.
        return torch.scatter(
            x,
            -1,
            indices[..., None],
            torch.zeros_like(x[..., :1]),
        )