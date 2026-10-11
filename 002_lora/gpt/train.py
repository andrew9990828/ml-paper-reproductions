
# =======================================================================
# My GPT from scratch - Training the model
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: October 10, 2026
#
# File: train.py
#
# Description:
# This is where we actually train the model. We will use our dataset.py
# class and PyTorch's DataLoader to load the data and feed it into our
# model from model.py.
#
# Mental Model:
#   1. Prepare the data:
#       Read text and tokenize it
#   2. Create the dataset
#   3. Create the dataloader
#   4. Train:
#       Loop over dataloader. For each batch, clear gradients, feed x
#       into the model, calculate loss against y, run backward, and
#       update weights.
#   5. Repeat:
#       Finishing one full pass through dataloader completes ONE epoch
#   6. Validate:
#       Run the model on unseen data without updating weights
#   7. Track and save:
#       Log training metrics to W&B and save checkpoints
#
# NOTE:
# The core GPT training loop was implemented by Andrew Bieber.
#
# The following supporting infrastructure was delegated to ChatGPT:
#   - Weights & Biases experiment tracking
#   - Validation loop and evaluation metrics
#   - Model/optimizer checkpoint saving
#   - Training progress logging and throughput measurements
#   - Loading smaller subsets of the prepared dataset
#   - Gradient clipping for training stability
#
# These additions allow us to monitor and save training runs without
# changing the underlying GPT architecture or learning process.
# =======================================================================

from dataset import GPTDataSet
from torch.utils.data import DataLoader
from model import AndrewsGPT

from pathlib import Path
from time import perf_counter

import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


# Put on GPU if we have one:
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Training on: {device}")


# Globals we can configure to train how we like
epochs = 1
max_steps = 2000
batch_size = 1
learning_rate = 0.0003

# How much data we want to use for this training experiment
train_stories = 10_000
val_stories = 200

# How often we want to log, validate, and save the model
log_every = 20
validate_every = 100
save_every = 250
val_batches = 10

loss_function = nn.CrossEntropyLoss()

GPT_CONFIGS = {
    "context_length": 256,
    "vocab_size": 50257,
    "embed_dim": 384,
    "num_heads": 6,
    "num_layers": 16,
    "dropout": 0.1
}

# File paths relative to this project
project_dir = Path(__file__).resolve().parent.parent

train_path = project_dir / "gpt_data" / "train.txt"
val_path = project_dir / "gpt_data" / "val.txt"

checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / "latest.pt"

tokenizer = tiktoken.get_encoding("gpt2")   # BPE... I chose gpt2
context_length = GPT_CONFIGS["context_length"]


# =======================================================================
# DELEGATED: Loading smaller subsets of the data
#
# Our full TinyStories dataset contains over 2 million stories.
# We don't want to load all of that into RAM for a small experiment.
#
# This reads a specified number of complete stories, stopping at the
# <|endoftext|> token we added during data preparation.
# =======================================================================

def load_stories(path, limit):

    lines = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            lines.append(line)

            if line.strip() == "<|endoftext|>":
                count += 1

                if count >= limit:
                    break

    if count < limit:
        raise ValueError(
            f"Expected {limit} stories but found {count} in {path}"
        )

    return "".join(lines)


# =======================================================================
# Prepare the datasets and dataloaders
# =======================================================================

# Read in the data we want to train and validate on.
train_txt = load_stories(train_path, train_stories)
val_txt = load_stories(val_path, val_stories)

# Create the dataset objects by feeding in the text
train_dataset = GPTDataSet(
    train_txt,
    tokenizer=tokenizer,
    context_length=context_length,
    stride=context_length
)

val_dataset = GPTDataSet(
    val_txt,
    tokenizer=tokenizer,
    context_length=context_length,
    stride=context_length
)

# Create our dataloaders
# Shuffle training data, but validation does not need shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# =======================================================================
# Initialize our GPT and optimizer
# =======================================================================

gpt = AndrewsGPT(GPT_CONFIGS).to(device)

optimizer = optim.AdamW(
    gpt.parameters(),
    lr=learning_rate
)


# =======================================================================
# DELEGATED: Validation
#
# We want to measure whether the model learns patterns that generalize
# beyond the training examples.
#
# Unlike training:
#   - We don't calculate gradients
#   - We don't update the model's weights
#   - We calculate average loss across validation batches
#
# The validation set is separate from the training set.
# =======================================================================

def validate():

    gpt.eval()

    total_loss = 0
    batches = 0

    with torch.no_grad():

        for x, y in val_loader:

            if batches >= val_batches:
                break

            inputs = x.to(device, non_blocking=True)
            targets = y.to(device, non_blocking=True)

            logits = gpt(inputs)

            # Same shapes we use during training
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = loss_function(logits, targets)

            total_loss += loss.item()
            batches += 1

    # Put the GPT back into training mode
    gpt.train()

    return total_loss / batches


# =======================================================================
# DELEGATED: Saving model checkpoints
#
# We don't want to lose hours of training if the program stops.
#
# Save:
#   - Model parameters
#   - Optimizer state
#   - Model configuration
#   - Current epoch and training step
#
# This lets us reload the trained weights later.
#
# NOTE: Exact training resumption is not implemented yet.
# =======================================================================

def save_checkpoint(epoch, step):

    temp_path = checkpoint_dir / "latest.tmp"

    torch.save(
        {
            "model_state_dict": gpt.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": GPT_CONFIGS,
            "epoch": epoch,
            "global_step": step
        },
        temp_path
    )

    # Replace the previous checkpoint only after saving finishes
    temp_path.replace(checkpoint_path)

    print(f"Checkpoint saved at step {step}", flush=True)


# =======================================================================
# DELEGATED: Weights & Biases experiment tracking
#
# W&B lets us monitor training from a browser while the GPT runs.
#
# We want to track:
#   - Training loss
#   - Validation loss
#   - Learning rate
#   - Training throughput (tokens/second)
#   - Number of optimizer steps
#
# This will also help when we compare LoRA experiments later.
# =======================================================================

run = wandb.init(
    project="andrews-gpt-from-scratch",
    name="gpt-initial-pretraining",
    config={
        **GPT_CONFIGS,
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_stories": train_stories,
        "val_stories": val_stories,
        "optimizer": "AdamW"
    }
)

# Tell W&B to use optimizer steps for our graphs
run.define_metric("global_step")
run.define_metric("train/*", step_metric="global_step")
run.define_metric("val/*", step_metric="global_step")


# =======================================================================
# Begin training below
#
# This is my original GPT training loop with a few additions for
# logging, validation, gradient clipping, and checkpoint saving.
#
# Core process:
#   Zero gradients -> Forward -> Loss -> Backward -> Update weights
# =======================================================================

# Put in training mode:
gpt.train()

global_step = 0

# Track average loss instead of only the last batch's loss
running_loss = 0
running_steps = 0
running_tokens = 0

start_time = perf_counter()

for i in range(epochs):

    # x = make prediction off these toks
    # y = targets
    for x, y in train_loader:

        # Clear gradients every batch
        optimizer.zero_grad()

        inputs = x.to(device, non_blocking=True)
        targets = y.to(device, non_blocking=True)

        logits = gpt(inputs)

        # y.shape = [B, T]
        # logits.shape = [B, T, V]
        # We have to reshape logits into -> [B*T, V]
        # We have to reshape y into -> [B*T]
        # So when we call loss we have [B*T, V] compared to [B*T]
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1) # 2D -> 1D

        loss = loss_function(logits, targets)

        loss.backward()

        # DELEGATED: Clip large gradients to improve stability
        nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)

        optimizer.step()

        global_step += 1

        # DELEGATED: Track average training loss and throughput
        running_loss += loss.item()
        running_steps += 1
        running_tokens += inputs.numel()

        # -------------------------------------------------------
        # DELEGATED: Log training progress to W&B
        # -------------------------------------------------------

        if global_step % log_every == 0:

            elapsed = perf_counter() - start_time
            avg_loss = running_loss / running_steps
            tokens_per_sec = running_tokens / max(elapsed, 1e-9)

            run.log({
                "global_step": global_step,
                "train/loss": avg_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/tokens_per_sec": tokens_per_sec
            })

            print(
                f"Epoch {i + 1} | "
                f"Step {global_step} | "
                f"Loss: {avg_loss:.4f} | "
                f"Tokens/sec: {tokens_per_sec:.1f}",
                flush=True
            )

            running_loss = 0
            running_steps = 0
            running_tokens = 0

            start_time = perf_counter()

        # -------------------------------------------------------
        # DELEGATED: Validate every N optimizer steps
        # -------------------------------------------------------

        if global_step % validate_every == 0:

            val_loss = validate()

            run.log({
                "global_step": global_step,
                "val/loss": val_loss
            })

            print(
                f"Step {global_step} | "
                f"Validation loss: {val_loss:.4f}",
                flush=True
            )

            # Reset timing after validation
            start_time = perf_counter()

        # -------------------------------------------------------
        # DELEGATED: Save checkpoints during training
        # -------------------------------------------------------

        if global_step % save_every == 0:
            save_checkpoint(i + 1, global_step)

            # Reset timing after writing a checkpoint
            start_time = perf_counter()

        # Stop once we reach our planned training budget
        if global_step >= max_steps:
            break

    print(f"Epoch: {i + 1}", flush=True)

    if global_step >= max_steps:
        break


# =======================================================================
# DELEGATED: Final experiment logging and checkpoint
#
# Make sure we save the last weights even if training ends before the
# next scheduled checkpoint.
# =======================================================================

# Record any training batches that weren't logged yet
if running_steps > 0:

    elapsed = perf_counter() - start_time

    run.log({
        "global_step": global_step,
        "train/loss": running_loss / running_steps,
        "train/lr": optimizer.param_groups[0]["lr"],
        "train/tokens_per_sec": running_tokens / max(elapsed, 1e-9)
    })

# Run final validation, even if max_steps was below validate_every
final_val_loss = validate()

run.log({
    "global_step": global_step,
    "val/loss": final_val_loss
})

save_checkpoint(i + 1, global_step)

print(f"Final validation loss: {final_val_loss:.4f}")
print(f"Training completed after {global_step} steps!")

run.finish()
