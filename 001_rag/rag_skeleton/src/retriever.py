# ============================================================
# RAG Reproduction — Dense Retriever
#
# Goal:
# Embed one question and compare it against every stored chunk.
# ============================================================

import torch
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)


MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(MODEL_NAME)
model = DPRQuestionEncoder.from_pretrained(MODEL_NAME)
model.eval()

TOP_K = 10


@torch.inference_mode()
def embed_query(query: str) -> torch.Tensor:
    """
    Turn one question into one DPR question embedding.

    Expected output shape:

        [1, 768]

    Think:
        The context encoder made the stored document vectors.
        What paired DPR model should make the query vector?
    """

    # TODO:
    # tokenize -> model -> pooler_output

    raise NotImplementedError


def load_embeddings(file_path: str) -> torch.Tensor:
    return torch.load(file_path)


def retrieve(
    chunk_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compare one query against all chunks and return top-k scores + indices.

    Shapes:

        query_embedding:  [1, 768]
        chunk_embeddings: [N, 768]

    Goal:

        [1, 768]  ?  [N, 768]
              ↓
           [1, N]

    Ask yourself:
        Which tensor needs to be transposed?

    Returns:

        scores:  [1, k]
        indices: [1, k]
    """

    # TODO 1:
    # Compute one similarity score per chunk using matrix multiplication.

    # TODO 2:
    # Keep only the top-k scores and matching indices.

    raise NotImplementedError


if __name__ == "__main__":
    query = input("Ask a question about baseball: ")

    chunk_size = 50
    embedding_path = f"data/embeddings/embeddings_size{chunk_size}.pt"

    query_embedding = embed_query(query)
    chunk_embeddings = load_embeddings(embedding_path)

    scores, indices = retrieve(
        chunk_embeddings,
        query_embedding,
        TOP_K,
    )

    print("\nScores shape:", scores.shape)
    print("Indices shape:", indices.shape)
    print("\nTop-k indices:")
    print(indices)
