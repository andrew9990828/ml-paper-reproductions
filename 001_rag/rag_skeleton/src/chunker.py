# ============================================================
# RAG Reproduction — Chunk the Wikipedia pages
#
# Goal:
# Turn raw article text into fixed-size WORD chunks stored as JSONL.
# ============================================================

import json
from pathlib import Path


def chunk_file(
    file_path: Path,
    chunk_size: int,
    save_path: str,
    chunk_num: int,
) -> int:
    """
    Read one article, split it into fixed-size word chunks,
    write each chunk as one JSONL object, and return the final chunk id.

    Output object shape:

        {
            "chunk": <global integer id>,
            "article": <source filename>,
            "text": <chunk text>
        }

    Think before coding:
        - What does raw_txt.split() give you?
        - How can you walk through that list chunk_size words at a time?
        - Why do we want one JSON object per line instead of repeatedly
          json.dump()-ing one giant structure?
    """

    with open(file_path, "r", encoding="utf8") as f:
        raw_txt = f.read()

    words = raw_txt.split()

    # TODO 1:
    # Break `words` into non-overlapping groups of `chunk_size`.
    #
    # You may use slicing or a manual loop.
    #
    # Do not write an incomplete final chunk unless you deliberately
    # decide to change the behavior of the original reproduction.

    # TODO 2:
    # For each chunk:
    #   - increment the global chunk id
    #   - build the metadata dict
    #   - append one JSON object + "\n" to save_path

    # TODO 3:
    # Return the final global chunk id so the next article can continue
    # numbering from where this one stopped.

    raise NotImplementedError


if __name__ == "__main__":
    raw_dir = Path("data/raw")

    # Change these to reproduce the four experiments:
    # 10, 25, 50, 100
    chunk_size = 100

    save_jsonl = f"data/chunks/chunks_size{chunk_size}.jsonl"

    # Avoid accidentally appending duplicate chunks across repeated runs.
    Path(save_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(save_jsonl).write_text("", encoding="utf8")

    next_chunk = 0

    for file_path in raw_dir.glob("*.txt"):
        next_chunk = chunk_file(
            file_path,
            chunk_size,
            save_jsonl,
            next_chunk,
        )
