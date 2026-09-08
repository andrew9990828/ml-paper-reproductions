# ============================================================
# RAG Reproduction — Chunk the wikipedia pages
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 7, 2026
#
# File: chunker.py
#
# Description:
# Reads each wikipedia page I have saved locally in this project
# Stores them all as chunks of x size (I'm making it configurable)
# All chunks labeled into a local json file under data/processed/chunks.json
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

# First just set up a method that can process one file, 
# then just create the pipeline

import json

# Global var to keep chunk_id
chunk_id = 0

def chunk_file(file_path: str, chunk_size: int, save_path: str) -> None:

    # First just read the entire contents of the given file
    with open(file_path, "r", encoding="utf8") as f:
        raw_txt = f.read()

    # A list of all the words
    words = raw_txt.split()
    num_words = len(words)

    last_idx = 0

    while num_words > chunk_size:

        chunk = []
        for i in range(chunk_size):
            chunk.append(words[i+last_idx])

        chunk_text = " ".join(chunk)
        chunk_id += 1
        last_idx += 50
        num_words -= 50
        meta_data = {
            "chunk": chunk_id,
            "article": file_path,
            "text": chunk_text
            }

        with open(save_path, "w", encoding="utf8") as f:
            json.dump(meta_data, f, indent=2)

        
