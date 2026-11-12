"""Each token looks backward for useful information. Never forward."""
import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        # TODO 1: Require embed_dim % num_heads == 0.
        # Each head gets D = embed_dim // num_heads features.
        # Create separate query, key, value, and output Linear layers (C -> C).
        # Use bias=False. Add attention dropout and output dropout.
        raise NotImplementedError("TODO: define the attention layers")

    def forward(self, x):
        # x: [B, T, C]. Return the SAME shape.

        # TODO 2: Project x into Q, K, V. Split each into heads.
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D].

        # TODO 3: Q @ K.transpose(-2, -1), divided by sqrt(D).
        # Scores: [B, H, T, T]. Rows ask; columns are positions to look at.

        # TODO 4: Mask future positions with -inf BEFORE softmax.
        # Upper triangle, diagonal=1. Build the mask on x.device.
        # Softmax over the final dimension, then apply attention dropout.

        # TODO 5: Multiply attention weights by V, then merge the heads.
        # Transpose back before reshaping to [B, T, C].
        # Finish with the output projection and output dropout.
        # Check: in eval mode, changing a future token must not change earlier outputs.
        raise NotImplementedError("TODO: compute causal multi-head attention")
