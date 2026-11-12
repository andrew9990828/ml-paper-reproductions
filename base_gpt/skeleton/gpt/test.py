"""Evaluate the chosen checkpoint on stories you kept out of training."""
import math
from pathlib import Path
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
from dataset import GPTDataSet, load_stories
from model import AndrewsGPT

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_DIR / "checkpoints" / "best.pt"
NUM_STORIES = 100
BATCH_SIZE = 4


def main():
    # TODO 1: Choose the device. Load CHECKPOINT on cpu with weights_only=True.
    # Read saved["config"] to build the model and dataset. Don't guess the config.
    # Load saved["model_state_dict"] into the model, then move it to device.

    # TODO 2: Load NUM_STORIES from gpt_data/test.txt and use the GPT-2 tokenizer.
    # Build the dataset with stride=context_length. DataLoader shuffle=False.
    # Report that this is a SUBSET if you're not evaluating the whole test set.

    # TODO 3: eval() + no_grad(). Loop through every test batch.
    # Move inputs/targets to device and get logits.
    # Flatten logits to [B*T,V] and targets to [B*T]. Compute cross-entropy.

    # TODO 4: total_loss += loss.item() * tokens_in_THIS_batch.
    # Accumulate total tokens separately. Don't weight by the running count.
    # Guard against zero tokens; test_loss = total_loss / total_tokens.
    # Perplexity is math.exp(test_loss). Lower is better for the same evaluation.

    # TODO 5: Print checkpoint step, story/token counts, loss, and perplexity.
    # Use validation for tuning. Don't repeatedly tune against this test result.
    raise NotImplementedError("TODO: evaluate and report the test results")


if __name__ == "__main__":
    main()
