# =======================================================================
# My GPT from scratch - Training the model
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: October 7, 2026
#
# File: train.py
#
# Description:
# This is where we actually train the model. We will use our dataset.py 
# class and PyTorch's Dataloader to load the data and feed it into our
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
# =======================================================================
from dataset import GPTDataSet
from torch.utils.data import DataLoader
from model import AndrewsGPT
import tiktoken
import torch.nn as nn
import torch.optim as optim


# Globals we can configure to train how we like
data_path = "002_lora/gpt_data/tinystories_test_500_lines.txt"
epochs = 3
loss_function = nn.CrossEntropyLoss()
optimizer = optim.AdamW()

GPT_CONFIGS = {
    "context_length" : 256,
    "vocab_size" : 50257,
    "embed_dim" : 384,
    "num_heads" : 6,
    "num_layers" : 16,
    "dropout" : 0.1
}

tokenizer = tiktoken.get_encoding("gpt2")   # BPE... I chose gpt2
context_length = 256                        # Sequence length? I chose 256

# Read in the data you want to train on. This is stored as a string.
with open(data_path, "r", encoding="utf8") as f:
    raw_txt = f.read()

# Create the dataset object by feeding it the data you just loaded
dataset = GPTDataSet(
    raw_txt, 
    tokenizer=tokenizer, 
    context_length=context_length,
    stride=context_length
)

# Create a DataLoader object we will use to 
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

gpt = AndrewsGPT(GPT_CONFIGS)

# Begin training below
for _ in range(epochs):

    # x = make prediction off these toks
    # y = targets
    for x, y in dataloader:

        logits = gpt(x)

        loss = loss_function(logits, y)

        loss.backward()

        optimizer
