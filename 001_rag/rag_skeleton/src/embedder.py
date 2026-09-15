# ============================================================
# RAG Reproduction — Build DPR context embeddings
#
# Goal:
# Turn every chunk into one 768-dimensional dense vector.
# ============================================================

import json
from pathlib import Path

import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)


MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"

tokenizer = DPRContextEncoderTokenizer.from_pretrained(MODEL_NAME)
model = DPRContextEncoder.from_pretrained(MODEL_NAME)
model.eval()


def load_lines(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf8") as f:
        return f.readlines()


def parse_line(line: str) -> dict:
    return json.loads(line)


def process_all_lines(lines: list[str]) -> list[dict]:
    return [parse_line(line) for line in lines]


@torch.inference_mode()
def embed_chunk(
    file_path: str,
    save_path: str,
) -> None:
    """
    Embed every chunk in a JSONL file.

    Expected shapes:

        one chunk embedding: [1, 768]
        stored matrix:       [N, 768]

    Important:
        Use the DPR CONTEXT encoder here.
        The question encoder belongs in retriever.py.
    """

    lines = load_lines(file_path)
    processed = process_all_lines(lines)

    # Make sure the output directory exists on a fresh clone.
    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # TODO 1:
    # Allocate a tensor that can store one 768-d vector per chunk.

    # TODO 2:
    # For each chunk:
    #   - tokenize line["text"]
    #   - send the tokens through the DPR context encoder
    #   - inspect pooler_output
    #   - remove only the batch dimension
    #   - place that vector into the correct row

    # TODO 3:
    # torch.save(...) the complete [N, 768] tensor.

    raise NotImplementedError


if __name__ == "__main__":
    chunk_size = 100

    chunk_path = f"data/chunks/chunks_size{chunk_size}.jsonl"
    save_path = f"data/embeddings/embeddings_size{chunk_size}.pt"

    embed_chunk(
        chunk_path,
        save_path,
    )