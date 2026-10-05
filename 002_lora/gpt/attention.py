# =======================================================================
# My GPT from scratch - Attention is all you need
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 26, 2026
#
# File: attention.py
#
# Description:
# "Attention is All You Need"
# This is the heart of the transformer and makes GPTs
# real. Here we will be implementing multi-head attention (MHA).
# You should internalize single head self-attention before
# implementing MHA. MHA is simply the exact same idea (done in parallel),
# but tracking and internalizing the shapes can be difficult.
#
# Mental Model:
#   Attention:
#       1. Project the input embeddings into Q, K, and V
#       2. Forward pass
#       3. Output = Contextualized token embeddings
# =======================================================================
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.qkv_bias = False
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # .1 is the standard choice
        self.dropout = nn.Dropout(p=dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim % num_heads isn't equal to 0. Check initialization."

        self.head_dim = self.embed_dim // self.num_heads

        self.queries = nn.Linear(self.embed_dim, self.embed_dim, self.qkv_bias)
        self.keys = nn.Linear(self.embed_dim, self.embed_dim, self.qkv_bias)
        self.values = nn.Linear(self.embed_dim, self.embed_dim, self.qkv_bias)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim, self.qkv_bias)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        D = self.head_dim        
        H = self.num_heads

        assert C == self.embed_dim, "Embed_dim C must be equal to the set embed_dim in model"

        queries = self.queries(x)       # [B, T, C]
        keys = self.keys(x)             # [B, T, C]
        values = self.values(x)         # [B, T, C]
        
        q = torch.reshape(queries, (B, T, H, D))    # [B, T, H, D]
        k = torch.reshape(keys, (B, T, H, D))       # [B, T, H, D]
        v = torch.reshape(values, (B, T, H, D))     # [B, T, H, D]

        q = q.transpose(2, 1)                   # [B, H, T, D]
        k = k.transpose(2, 1)                   # [B, H, T, D]
        v = v.transpose(2, 1)                   # [B, H, T, D]

        # Scores = (q @ k.T) / sqrt(D)
        k_t = k.transpose(-1, -2)     # [B, H, T, D] -> [B, H, D, T]
        scores = (q @ k_t) / D**0.5   # [B, H, T, T]

        # Causal mask prevents the model from cheating
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)                 # [T, T]
        masked_scores = scores.masked_fill(mask.bool(), -torch.inf)

        # After the mask, get probabilities calculated per row. Each row has all tokens add to 1.
        attention = torch.softmax(masked_scores, dim=-1)

        # Dropout is a core regularization technique to prevent overfitting
        attention = self.dropout(attention)

        # Now since we have probabilities, get context vectors by grabbing their values
        context_vectors = attention @ v                                 # [B, H, T, D]

        context_vectors = context_vectors.transpose(dim0=1, dim1=-2)    # [B, T, H, D]
        context_vectors = torch.flatten(context_vectors, start_dim=-2, end_dim=-1)  # [B, T, H, D] -> [B, T, C]
        proj_context_vectors = self.output_proj(context_vectors)                    # [B, T, C] @ [C, C] -> [B, T, C]

        contextualized_embeds = self.dropout(proj_context_vectors)

        return contextualized_embeds    # -> [B, T, C]

    
if __name__ == "__main__":
    attention = MultiHeadAttention(384, 6, 0.1)

    B, T, C = 2, 12, 384
    x = torch.rand(B, T, C)

    output = attention(x)

    print("Input: ", x.shape)
    print("Output:", output.shape)

    assert output.shape == x.shape