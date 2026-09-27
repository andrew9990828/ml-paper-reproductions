# =======================================================================
# My GPT from scratch - The model
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 26, 2026
#
# File: model.py
#
# Description:
# We are using the GPT-2 tokenizer, but besides that, we have some
# freedom to choose our own size of this GPT. I wanted to aim for
# Roughly 50million parameters which is under half the size of base
# GPT-2 which has 124million params.
#
# Params / Architecture:
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
# =======================================================================