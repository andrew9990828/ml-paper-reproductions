"""Predict, measure the error, and update the weights. Repeat."""
from pathlib import Path
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
from dataset import GPTDataSet, load_stories
from model import AndrewsGPT, GPT_CONFIGS

PROJECT_DIR = Path(__file__).resolve().parent.parent
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 1
MAX_STEPS = 100  # Across the entire run, not per epoch.
TRAIN_STORIES, VAL_STORIES = 1000, 100
VALIDATE_EVERY = 50


def validate(model, loader, device):
    # TODO 1: Remember the current mode, then use eval() AND no_grad().
    # Move x/y to device. Logits [B,T,V] -> [B*T,V]; targets [B,T] -> [B*T].
    # Compute cross-entropy. Accumulate batch mean loss * batch token count.
    # Divide by TOTAL tokens, guarding against zero. Restore the previous mode.
    raise NotImplementedError("TODO: return token-weighted validation loss")


def save_checkpoint(model, optimizer, config, epoch, global_step, path):
    # TODO 2: Save a dictionary with these exact keys:
    # config, model_state_dict, optimizer_state_dict, epoch, global_step.
    # Make the parent folder. torch.save to a temporary file, then replace path.
    # That keeps an interrupted write from destroying the previous checkpoint.
    raise NotImplementedError("TODO: save the checkpoint")


def main():
    # TODO 3: Choose cuda if available, otherwise cpu. Use the GPT-2 tokenizer.
    # Load train.txt and val.txt from PROJECT_DIR / "gpt_data".
    # Make datasets with stride=context_length and DataLoaders.
    # Shuffle training only. Error if either loader has zero batches.

    # TODO 4: Build AndrewsGPT(GPT_CONFIGS), move to device, create AdamW.
    # Use nn.CrossEntropyLoss(). Start global_step at zero.

    # TODO 5: Set train mode. Loop over epochs, then batches:
    # Move x/y to device -> zero gradients -> forward -> reshape -> loss.
    # backward() -> clip gradient norm to 1.0 -> optimizer.step().
    # Increment global_step. Print loss occasionally. Stop BOTH loops at MAX_STEPS.

    # TODO 6: Validate every VALIDATE_EVERY steps. Save latest.pt periodically.
    # Also save best.pt when validation loss improves. Save latest.pt at the end.
    # Paths belong under PROJECT_DIR / "checkpoints".
    # Start by overfitting ONE batch. Does its loss actually go down?

    # OPTIONAL: Add W&B after the loop works. Never commit local wandb logs.
    raise NotImplementedError("TODO: build the training loop")


if __name__ == "__main__":
    main()
