# ============================================================
# RAG Reproduction — Chunk the wikipedia pages
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 8, 2026
#
# File: chunker.py
#
# Description:
# Reads each wikipedia page I have saved locally in this project
# Stores them all as chunks of x size (I'm making it configurable)
# All chunks labeled into a local json file under data/chunks/*.jsonl
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

# First just set up a method that can process one file, 
# then just create the pipeline

import json
from pathlib import Path


# Save_path should be .jsonl file because originally using a json file and json.dump()
# was rewriting the chunks everytime only keeping the final chunk of the last page
def chunk_file(file_path: Path, chunk_size: int, save_path: str, chunk_num: int) -> int:

    # First just read the entire contents of the given file
    with open(file_path, "r", encoding="utf8") as f:
        raw_txt = f.read()

    # A list of all the words
    words = raw_txt.split()
    num_words = len(words)

    last_idx = 0
    chunk_id = chunk_num 

    # Could do this with slicing but I choose a manual loop
    while num_words >= chunk_size:

        chunk = []
        for i in range(chunk_size):
            chunk.append(words[i+last_idx])

        chunk_text = " ".join(chunk)
        chunk_id += 1
        last_idx += chunk_size
        num_words -= chunk_size

        # This just builds the metadata dict
        meta_data = {
            "chunk": chunk_id,
            "article": file_path.name,
            "text": chunk_text
            }

        # NOTE: Using append here and json.dumps() which make a python obj
        # .jsonl makes it stupid easy to read the word chunks later or process
        # them in bulk. 
        with open(save_path, "a", encoding="utf8") as f:
            f.write(json.dumps(meta_data) + "\n")

    # Lets us keep track of the chunk_ids and pass them later each iteration
    return chunk_id


"""
I ran this script 4 seperate times:
    I used chunking of size 100, 50, 25, and 10!
    I did this to later run some experiments to see how that changes retrieval!
    Simply change the chunk_file() argument for size to generate different sizes
"""
if __name__ == "__main__":

    # All raw .txt files are located here
    raw_dir = Path("data/raw")
    # What save_path label I want
    save_jsonl = "data/chunks/chunks_size100.jsonl"
    # Initialized the first id to 0
    next_chunk = 0

    # Looping through all 10 articles and simply passing the previous chunk 
    for file_pa in raw_dir.glob("*.txt"):
        next_chunk = chunk_file(file_pa, 100, save_jsonl, next_chunk)