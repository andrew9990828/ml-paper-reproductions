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
import io
import tiktoken
import torch

# This is just to run tests as we go through the build
example_text = "One day, a little girl named Lily found a needle in her room. " \
"She knew it was difficult to play with it because it was sharp. Lily wanted to" \
" share the needle with her mom, so she could sew a button on her shirt. Lily went" \
" to her mom and said, "

# For my gpt, I used the gpt2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# len = 58 toks
token_ids = tokenizer.encode(example_text)

def get_token_window(
    token_ids: list[int],
    start: int,
    context_length: int,
) -> tuple[list[int], list[int]]:
    """Return an input window and its one-token-shifted target."""

    end = start + context_length
    x = token_ids[start:end]
    y = token_ids[start+1:end+1]

    return x, y


if __name__ == "__main__":
    ids = tokenizer.encode(example_text)
    print(len(ids))
    words = tokenizer.decode(ids)
    print(words)
    print(get_token_window(ids, 2, 8))