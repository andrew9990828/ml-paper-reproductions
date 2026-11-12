"""Build the GPT out of smaller pieces. Keep track of the shapes."""
import torch
import torch.nn as nn
from attention import MultiHeadAttention

# Small learning model. Andrew's model used C=384, 6 heads, and 16 layers.
GPT_CONFIGS = {
    "context_length": 256,
    "vocab_size": 50257,  # GPT-2 tokenizer; don't shrink this independently.
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
}


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        # TODO 1: Linear C -> 4C, GELU, Linear 4C -> C, then dropout.
        # Use bias=False for both Linear layers.
        raise NotImplementedError("TODO: define the feed-forward layers")

    def forward(self, x):
        # TODO 2: Pass x through those layers. [B, T, C] stays [B, T, C].
        raise NotImplementedError("TODO: run the MLP")


class TransformerBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # TODO 3: Two LayerNorms over C, one MultiHeadAttention, one MLP.
        raise NotImplementedError("TODO: define the block")

    def forward(self, x):
        # TODO 4: Normalize -> attention -> add the original x (residual).
        # Then normalize that result -> MLP -> add the second residual.
        # Normalization happens BEFORE each sublayer. Shape stays [B, T, C].
        raise NotImplementedError("TODO: connect the block")


class AndrewsGPT(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # TODO 5: Token embeddings [V, C] and learned position embeddings [Tmax, C].
        # Initialize embeddings with normal std=0.02. Add embedding dropout.
        # Create num_layers INDEPENDENT blocks, a final LayerNorm, and a
        # bias-free Linear C -> V. Tie its weight to the token embedding weight.
        raise NotImplementedError("TODO: assemble the GPT")

    def forward(self, x):
        # TODO 6: x contains integer token IDs [B, T]. Require 0 < T <= Tmax.
        # Make positions 0..T-1 on x.device. Add token + position embeddings.
        # Apply embedding dropout, all blocks, final norm, then output head.
        # Return raw logits [B, T, V]. No softmax here.
        raise NotImplementedError("TODO: run the GPT forward pass")
