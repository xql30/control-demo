"""Reusable hierarchy-supervised history-to-category router."""

from __future__ import annotations

import torch
from torch import nn


class HierarchicalCategoryRouter(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_categories: torch.Tensor,
        category_count: int,
        hierarchy_sizes: list[int],
        max_history: int,
        dimension: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("item_categories", item_categories)
        self.item_embedding = nn.Embedding(
            num_items + 1, dimension, padding_idx=0
        )
        self.category_embedding = nn.Embedding(category_count, dimension)
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
        self.leaf_head = nn.Linear(dimension, category_count)
        self.hierarchy_heads = nn.ModuleList(
            nn.Linear(dimension, size) for size in hierarchy_sizes
        )

    def forward(self, history: torch.Tensor):
        positions = torch.arange(history.shape[1], device=history.device)
        categories = self.item_categories[history]
        hidden = (
            self.item_embedding(history)
            + self.category_embedding(categories)
            + self.position_embedding(positions).unsqueeze(0)
        )
        hidden = self.encoder(
            self.dropout(hidden),
            src_key_padding_mask=history.eq(0),
        )
        state = self.norm(hidden[:, -1])
        return self.leaf_head(state), [
            head(state) for head in self.hierarchy_heads
        ]
