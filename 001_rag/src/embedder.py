# ============================================================
# RAG Reproduction — Embeddings assigned
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 11, 2026
#
# File: embedder.py
#
# Description:
# Reads all chunks from each set of different chunk sizes
# and dumps the bdense floating-point vectors into data/embeddings/*pt.
# We use the pretrained weights and BPE encoder of DPR given to
# us by the wonderful people who wrote this paper lol.
# YAY OPEN_SOURCE <3
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import json


tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

model = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

# Switch model from training mode -> inference mode
model.eval()


# First we have to actually read the jsonl file. Its read as a list of strings
def load_lines(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    return lines
    # embeddings = torch.tensor(len(lines), 768)
    # for line in lines

# Helper function that turns a string -> dict from our loaded jsonl lines
def parse_line(line: str) -> dict:
    hashmap = json.loads(line)
    return hashmap

# Uses parse_line to loop through all lines and convert strings -> dicts
# Stored as a list of dicts
def process_all_lines(lines: list[str]) -> list[dict]:
    processed = []
    for line in lines:
        processed.append(parse_line(line))

    return processed

def embed_chunk(file_path: str, save_path: str) -> None:
    lines = load_lines(file_path)
    processed = process_all_lines(lines)

    # Can just declare our embedding shapes here
    embeddings = torch.empty(len(lines), 768)

    for i, line in enumerate(processed):
        tokens = tokenizer(line["text"], return_tensors="pt")
        embed = model(**tokens).pooler_output
        embeddings[i] = embed.squeeze(0)

    torch.save(embeddings, save_path)

if __name__ == "__main__":

    # All chunks are located here
    save_name = "data/embeddings/embeddings_size100.pt"
    embed_chunk("data/chunks/chunks_size100.jsonl", save_name)
