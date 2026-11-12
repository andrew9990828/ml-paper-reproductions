# ============================================================
# My GPT from scratch - Run the Test set
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: November 11th, 2026
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
import torch.nn as nn
import tiktoken
import math
import wandb
from pathlib import Path
from model import AndrewsGPT
from dataset import GPTDataSet
from torch.utils.data import DataLoader



# put on gpu if we have one
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Testing on: {device}")

# Number of test_stories to test against
num_stories = 1000

# Declare your loss function here
loss_function = nn.CrossEntropyLoss()

# batch size for the dataloader later
batch_size = 16

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# File paths relative to this project
project_dir = Path(__file__).resolve().parent.parent

checkpoint = project_dir / "checkpoints" / "pretrained-final.pt"
test_txt = project_dir / "gpt_data" / "test.txt"

# Both the dataset and model use the saved training architecture.
saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
GPT_CONFIGS = saved["config"]



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


test_stories = load_stories(test_txt, num_stories)

test_dataset = GPTDataSet(
    text=test_stories,
    tokenizer=tokenizer,
    context_length=GPT_CONFIGS["context_length"],
    stride=GPT_CONFIGS["context_length"]
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
gpt.load_state_dict(saved["model_state_dict"])
gpt.to(device)

gpt.eval()

with torch.no_grad():
    total_loss = 0
    num_tokens = 0

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
        targets = targets.reshape(-1)  # 2D -> 1D

        # Average loss
        loss = loss_function(logits, targets)

        batch_total_loss = loss.item() * targets.shape[0]

        total_loss += batch_total_loss
        num_tokens += targets.shape[0]

    if num_tokens == 0:
        raise ValueError("No test tokens were evaluated.")
    
    test_loss = total_loss / num_tokens
    perplexity = math.exp(test_loss)

# --- DELEGATED TO CHATGPT: print results and record them in W&B ---

print("\nTest evaluation complete!")
print(f"Checkpoint: {checkpoint.name}")
print(f"Checkpoint step: {saved['global_step']:,}")
print(f"Test stories: {num_stories:,} (subset of test.txt)")
print(f"Test batches: {len(test_loader):,}")
print(f"Test tokens: {num_tokens:,}")
print(f"Test loss: {test_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

# Record final results after evaluation, so they are printed locally
# even if connecting to W&B fails.
# This creates a new evaluation run in the same project as training.
with wandb.init(
    project="andrews-gpt-from-scratch",
    entity="andrew9990828-milwaukee-school-of-engineering-msoe",
    job_type="test",
    name=f"gpt-test-step-{saved['global_step']}",
    dir=str(project_dir),
    config={
        "model_config": GPT_CONFIGS,
        "checkpoint_file": checkpoint.name,
        "checkpoint_step": saved["global_step"],
        "checkpoint_epoch": saved["epoch"],
        "test_file": test_txt.name,
        "test_stories": num_stories,
        "test_selection": "first 1000 stories in test.txt",
        "batch_size": batch_size,
        "stride": GPT_CONFIGS["context_length"],
        "tokenizer": "gpt2",
        "device": str(device),
    },
) as run:
    metrics = {
        "test/loss": test_loss,
        "test/perplexity": perplexity,
        "test/num_tokens": num_tokens,
        "test/num_stories": num_stories,
        "test/num_batches": len(test_loader),
    }

    run.log(metrics)
    run.summary.update(metrics)

# --- END DELEGATED ---