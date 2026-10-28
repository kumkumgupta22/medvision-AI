"""Reusable ViT-Tiny architecture shared by training and web inference."""
from __future__ import annotations

import torch
from torch import nn

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = 96
PATCH_SIZE = 16
EMBED_DIM = 64
DEPTH = 3
NUM_HEADS = 4
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)


class VisionTransformer(nn.Module):
    """ViT-Tiny with 6x6 image patches, a class token, and three encoder blocks."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        patch_count = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch_embed = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.class_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, EMBED_DIM))
        self.embedding_dropout = nn.Dropout(0.1)
        block = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=DEPTH, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBED_DIM, num_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, patches), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        encoded = self.norm(self.encoder(tokens))
        return self.classifier(encoded[:, 0])
