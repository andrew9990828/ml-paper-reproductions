# =======================================================================
# My GPT from scratch - The model
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: October 5, 2026
#
# File: model.py
#
# Description:
# We are using the GPT-2 tokenizer, but besides that, we have some
# freedom to choose our own size of this GPT. I wanted to aim for
# Roughly 50million parameters which is under half the size of base
# GPT-2 which has 124million params.
#
# Params:
#
# VOCAB_SIZE = 50_257
# CONTEXT_LENGTH = 256
#
# EMBED_DIM = 384
# NUM_HEADS = 6
# HEAD_DIM = EMBED_DIM // NUM_HEADS
#
# NUM_LAYERS = 16
# MLP_DIM = 4 * EMBED_DIM
#
# DROPOUT = 0.1
# WEIGHT TYING: yes
# Roughly 40 Million+ Params
#
#
# Model:
#   1. Input
#   2. Layernorm
#   3. Attention
#   4. Residual Add
#   5. Layernorm
#   6. MLP
#   7. Residual Add
# =======================================================================
import torch
import torch.nn as nn
from attention import MultiHeadAttention


GPT_CONFIGS = {
    "context_length" : 256,
    "vocab_size" : 50257,
    "embed_dim" : 384,
    "num_heads" : 6,
    "num_layers" : 16,
    "dropout" : 0.1
}


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer1 = nn.Linear(embed_dim, 4*embed_dim, bias=False)
        self.layer2 = nn.Linear(4*embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        assert C == self.embed_dim, "C must match model embed_dim"

        x = self.layer1(x)      # [B, T, C] -> [B, T, 4C]
        x = self.activation(x)  # [B, T, 4C]
        x = self.layer2(x)      # [B, T, 4C] -> [B, T, C]
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, configs=GPT_CONFIGS):
        super().__init__()
        self.embed_dim = configs["embed_dim"]
        self.num_heads = configs["num_heads"]
        self.dropout = nn.Dropout(p=configs["dropout"])
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(self.embed_dim, configs["dropout"])
        self.attention = MultiHeadAttention(
            self.embed_dim, 
            self.num_heads, 
            configs["dropout"]
        )

    def forward(self, x: torch.Tensor):
        # Layernorm1 applied
        x_prev1 = x
        x = self.norm1(x)
        
        # Apply attention
        res = self.attention(x)
        
        # Residual add
        x = x_prev1 + res
        x_prev2 = x
        
        # Layernorm2 applied
        x = self.norm2(x)
        
        # Send through MLP
        res2 = self.mlp(x)
        
        # Residual add #2
        x = x_prev2 + res2
        
        return x


class AndrewsGPT(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.embed_dim = configs["embed_dim"]
        self.num_heads = configs["num_heads"]
        self.num_layers = configs["num_layers"]
        self.context_length = configs["context_length"]
        self.vocab_size = configs["vocab_size"]
        self.dropout = nn.Dropout(p=configs["dropout"])
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(configs) for _ in range(self.num_layers)]
        )
        self.embed_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_table = nn.Embedding(self.context_length, self.embed_dim)
        self.language_modeling_head = nn.Linear(
            self.embed_dim, 
            self.vocab_size, 
            bias=False
        )

    def forward(self, x: torch.Tensor): 
        B, T= x.shape   # [B, T]

        assert T <= self.context_length, "Num of tokens (T) need to equal context_length"

        tok_embeds = self.embed_table(x)    # [B, T, C]
        pos_embeds = self.pos_table(
            torch.arange(T, device=x.device)    # [T, C]
        )      

        x = tok_embeds + pos_embeds         # [B, T, C]
        x = self.dropout(x)                 # [B, T, C]

        for block in self.blocks:
            x = block(x)        

        x = self.final_norm(x)              # [B, T, C] C = embed_dim = 384

        # Apply LM head to get logits
        x = self.language_modeling_head(x)  # [B, T, C] -> [B, T, V] V = vocab_size = 50257

        return x    # [B, T, V] which are logits
    

if __name__ == "__main__":
    x = torch.tensor([
        [1, 4, 5, 1233, 5444, 40000, 50000, 2345],
        [54, 23232, 5354, 5243, 7897, 12323, 43, 45],
    ])

    model = AndrewsGPT(GPT_CONFIGS)
    out = model(x)

    print("Input: ", x.shape)
    print("Output:", out.shape)
