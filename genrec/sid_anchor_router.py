"""Frozen history-to-SID router used by anchor-conditioned generation."""

from __future__ import annotations

import torch
from torch import nn


class SIDAnchorRouter(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_sids: torch.Tensor,
        max_history: int,
        dimension: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("item_sids", item_sids)
        self.item_embedding = nn.Embedding(
            num_items + 1, dimension, padding_idx=0
        )
        self.sid_embeddings = nn.ModuleList(
            nn.Embedding(256, dimension) for _ in range(4)
        )
        self.position_embedding = nn.Embedding(max_history, dimension)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.sid_heads = nn.ModuleList(
            nn.Linear(dimension, 256) for _ in range(4)
        )

    def forward(self, history: torch.Tensor) -> list[torch.Tensor]:
        positions = torch.arange(history.shape[1], device=history.device)
        semantic_ids = self.item_sids[history]
        semantic = torch.stack(
            [
                embedding(semantic_ids[..., digit])
                for digit, embedding in enumerate(self.sid_embeddings)
            ],
            dim=0,
        ).mean(dim=0)
        valid = history.ne(0).unsqueeze(-1)
        hidden = (
            self.item_embedding(history)
            + semantic * valid
            + self.position_embedding(positions).unsqueeze(0)
        )
        hidden = self.encoder(
            self.dropout(hidden),
            src_key_padding_mask=history.eq(0),
        )
        state = self.norm(hidden[:, -1])
        return [head(state) for head in self.sid_heads]
