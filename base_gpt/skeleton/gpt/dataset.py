"""Turn stories into input/target pairs. The target is one token ahead."""
import torch
from torch.utils.data import Dataset

EOS_TOKEN = "<|endoftext|>"


def load_stories(path, limit):
    # TODO 1: Read UTF-8 lines until you've counted `limit` EOS markers.
    # Keep the markers in the returned text. Raise an error if there aren't
    # enough stories. With limit=None, read all stories.
    raise NotImplementedError("TODO: return the story text")


def get_token_window(token_ids, start, context_length):
    # TODO 2: Return two slices of length context_length.
    # Input starts at `start`. Target starts one position later.
    # Example: [10, 20, 30, 40] -> x=[10, 20, 30], y=[20, 30, 40].
    raise NotImplementedError("TODO: return x, y")


class GPTDataSet(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        # TODO 3: Store context_length, stride, and the encoded token IDs.
        # Allow the EOS marker when encoding: allowed_special={EOS_TOKEN}.
        # Require positive context_length and stride.
        raise NotImplementedError("TODO: tokenize and store the text")

    def __len__(self):
        # TODO 4: Count valid starting positions, stepping by stride.
        # Each window needs context_length + 1 tokens, including its target.
        # Too little text -> zero windows. Don't return the token count.
        raise NotImplementedError("TODO: return the number of windows")

    def __getitem__(self, idx):
        # TODO 5: Start at idx * stride and call get_token_window.
        # Return x and y as torch.long tensors, both shaped [T].
        # DataLoader will stack them into [B, T].
        raise NotImplementedError("TODO: return input and target tensors")
