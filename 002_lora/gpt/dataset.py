# ============================================================
# My GPT from scratch - Dataset file
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 25, 2026
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
from torch.utils.data import DataLoader, Dataset

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
    context_length: int
) -> tuple[list[int], list[int]]:
    """Return an input window and its one-token-shifted target."""

    end = start + context_length
    x = token_ids[start:end]
    y = token_ids[start+1:end+1]

    return x, y


class GPTDataSet(Dataset):
    """
    This class returns one (x, y) pair using the base Dataset class provided
    by torch documentation. You can find the dataset documentation at this
    link: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial
    """
    def __init__(self, text: str, tokenizer: tiktoken, context_length: int, stride: int):
        self.token_ids = tokenizer.encode(text)
        self.context_length = context_length
        self.stride = stride

    # Count the number of possible windows given by the text corpus
    # and stride & context_length.
    def __len__(self) -> int:
        num_tokens = len(self.token_ids)
        len_count = 0
        idx = 0

        while num_tokens > idx + self.context_length:
            len_count += 1
            idx += self.stride
            
        return len_count

    # Just grab a token window based on stride using our helper
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base, target = get_token_window(self.token_ids, idx * self.stride, self.context_length)
        return torch.tensor(base, dtype=torch.long), torch.tensor(target, dtype=torch.long)



# def dataloader(text: str, 
#     tokenizer: tiktoken, 
#     context_length: int, 
#     stride: int, 
#     batch_size: int
# ) -> tuple[torch.Tensor, torch.Tensor]: 
#     dataset = GPTDataSet(text, tokenizer, context_length, stride)
#     idx = 0
#     base_batches = torch.empty(batch_size,)
#     target_batches = torch.empty(batch_size)

#     while idx + context_length < len(dataset):
#         base, targets = dataset[idx]
#         base_batches = torch.cat((base_batches, base), dim=1)
#         target_batches = torch.cat((base_batches, base), dim=1)
#         idx += stride

#     return base_batches, target_batches



if __name__ == "__main__":
    ids = tokenizer.encode(example_text)
    print(len(ids))
    words = tokenizer.decode(ids)
    print(words)
    print(get_token_window(ids, 2, 8))
    test_dataset = GPTDataSet(example_text, tokenizer, 8, 2)
    data = test_dataset[15]
    base, targets = data
    dataloader = DataLoader(GPTDataSet(example_text, tokenizer, 8, 2), batch_size=8, shuffle=True)
    print(dataloader)
    print(base, targets)