# ============================================================
# My GPT from scratch - Dataset file
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 22, 2026
#
# File: dataset.py
#
# Description:
# Takes the raw 18million line text corpus and turns it in to
# training examples. This is done with our dataset class inside
# of this file. At the end, the dataloader method turns these
# examples into batches for training.
#
# Mental Model:
#   1. Raw text corpus
#   2. Tokenize into IDs
#   3. Slice into fixed-length windows
#   4. Create input x & create shifted target y
#   5. Dataset returns one (x, y) pair
#   6. Dataloader stacks many pairs into batches
#   7. Train.py feeds those batches into the model
# ============================================================

import tiktoken
import torch

# This is just to run tests as we go through the build
example_text = "One day, a little girl named Lily found a needle in her room. " \
"She knew it was difficult to play with it because it was sharp. Lily wanted to" \
" share the needle with her mom, so she could sew a button on her shirt. Lily went" \
" to her mom and said, "

# For my gpt, I used the gpt2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    ids = tokenizer.encode(example_text)
    print(ids)
    words = tokenizer.decode(ids)
    print(words)
