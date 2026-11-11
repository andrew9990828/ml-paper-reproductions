# ============================================================
# My GPT from scratch - Run the Test set
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: November 10th, 2026
#
# File: test.py
#
# Description:
# Running the validation set during training lets us judge 
# potential overfitting. With that, we can tune hyperparams
# and evaluate the model better. In contrast, the test set
# allows us to test generalization. Can this model take its
# learned patterns to structures it's never seen before?
#
# Mental Model:
#   1. Load the checkpoint
#   2. Prepare the test set
#   3. Set eval mode + disable gradients
#   4. Loop through test batches
#   5. Combine the results
# ============================================================
import torch
import tiktoken
from pathlib import Path
from model import AndrewsGPT
from dataset import GPTDataSet
from torch.utils.data import DataLoader
from train import load_stories

# put on gpu if we have one
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Testing on: {device}")

GPT_CONFIGS = {
    "context_length": 256,
    "vocab_size": 50257,
    "embed_dim": 384,
    "num_heads": 6,
    "num_layers": 16,
    "dropout": 0.1
}

# Number of test_stories to test against
num_stories = 1000

# batch size for the dataloader later
batch_size = 16

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt-2")

# File paths relative to this project
project_dir = Path(__file__).resolve().parent.parent

checkpoint = project_dir / "checkpoints" / "latest.pt"

test_stories = load_stories(checkpoint, num_stories)

test_dataset = GPTDataSet(
    text=test_stories,
    tokenizer=tokenizer,
    context_length=GPT_CONFIGS["context_length"],
    stride=GPT_CONFIGS["stride"]
)

# Create our dataloader. Test set does not need shuffling
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)

print(f"Test batches: {len(test_loader)}")

gpt = AndrewsGPT(GPT_CONFIGS)

torch.no_grad()
gpt.eval()


for x, y in test_loader:

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