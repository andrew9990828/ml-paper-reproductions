# ============================================================
# My GPT from scratch - Make GPT talk back!
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: November 12th, 2026
#
# File: generate.py
#
# Description:
# At this point we have trained, validated, and tested our
# GPT. Now it's time for our model to predict the output
# given our input! It was trained on TinyStories without
# finetuning. Given the beginning of a story, it generates
# a continuation using its learned patterns.
#
# Mental Model:
#   1. Load the trained model
#   2. Turn the prompt into token IDs
#   3. Feed the most recent tokens into the model
#   4. Take the final position's logits
#   5. Choose next token:
#       I chose sampling. Sampling turns the logits into
#       probabilities using softmax then randomly selects
#       according to those probabilities.
#   6. Append that token then repeat
#   7. Stop and decode
# ============================================================
import torch
import tiktoken
import torch.nn as nn
from model import AndrewsGPT
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

project_dir = Path(__file__).resolve().parent.parent

trained_checkpoint = project_dir / "checkpoints" / "pretrained-final.pt"

tokenizer = tiktoken.get_encoding("gpt2")

trained_metrics = torch.load(
    trained_checkpoint,
    map_location="cpu",
    weights_only=True
)
GPT_CONFIGS = trained_metrics["config"]

gpt = AndrewsGPT(GPT_CONFIGS)
gpt.load_state_dict(trained_metrics["model_state_dict"])
gpt.to(device)

# --- DELEGATED TO CHATGPT: terminal presentation ---
print("\n" + "=" * 60)
print("                        ANDREW'S GPT")
print("           Give me a beginning. I'll write the story.")
print("=" * 60)

prompt = input("\nYour story prompt:\n> ")
# --- END DELEGATED ---

if not prompt.strip():
    raise ValueError("Please enter a nonempty prompt.")

tokens = tokenizer.encode(prompt)

tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0)
tokens = tokens.to(device, non_blocking=True)

gpt.eval()
context_length = GPT_CONFIGS["context_length"]
max_tokens_output = 200

# --- DELEGATED TO CHATGPT: generation status ---
print("\nWriting your story...", flush=True)
# --- END DELEGATED ---

with torch.no_grad():

    for _ in range(max_tokens_output):
        context = tokens[:, -context_length:]
        logits = gpt(context)

        next_logits = logits[:, -1, :]
        softmax = nn.Softmax(dim=-1)
        probs = softmax(next_logits)

        next_token = torch.multinomial(probs, num_samples=1)

        # If <|endoftext|> token is hit, stop generation.
        if next_token.item() == tokenizer.eot_token:
            break

        tokens = torch.cat((tokens, next_token), dim=-1)

# --- DELEGATED TO CHATGPT: display the complete story ---
story = tokenizer.decode(tokens[0].tolist())

print("\n" + "=" * 60)
print("GPT STORY")
print("=" * 60)
print()
print(story)
print()
print("=" * 60)
# --- END DELEGATED ---