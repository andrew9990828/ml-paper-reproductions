# ============================================================
# RAG Reproduction — Build the retriever
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 12, 2026
#
# File: retriever.py
#
# Description:
# Thinking through this description for this file was easily the biggest
# "OOOOOHHHHHH" moment of my self-ML studies. Here's exactly what were doing:
# RAG-Sequence:
#   1. We take an input/query, we then tokenize that input and embed it.
#   2. In RAG-Sequence, we're comparing this one input against all embeddings.
#   3. This sounds VERY similar to attention in a transformer and the math is
#       somewhat similar. The biggest difference is we obviously don't have QKV
#       and more importantly, there's no quadratic bottleneck like in attention
#       with Q @ K.T. INSTEAD were comparing ONE query against ALL chunked embeddings.
#       This is Big O(n) for time or more precisely O(num_chunks * embedding_dim). 
#       But what's so cool is the math.
#   4. Very similar in math, we are comparing our input or query to every embedding.
#   5. Lets think in shapes:
#       input.shape = [1, 768]    chunked_embeddings.shape = [num_chunks, 768]
#       ****TO COMPARE EMBEDDINGS WE DO MATMUL****
#   6. In matmul would [1, 768] x [num_chunks, 768] work? Obviously not.
#       Instead: input @ chunked_embeddings.T -> [1, 768] x [768, num_chunks] -> [1, num_chunks]
#   7. Sick, we made the comparisons, now get top-k or in this case top 10.
#   8. Return the top-k scores and indices. Boom, this file did its job.
# 
# THIS is why I LOVE studying ML.
#
# NOTE: We keep TOP_K = 10 the same because this allows us to see how chunk_size effects
# retrieval -> generator output best. So TOP_K remains constant with all tests. 
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


# Important to note that for the embedder.py file we used the contextencoder
# Here we want to use the DPRQuestionEncoder and tokenizer!
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)

model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)

# Switch model from training mode -> inference mode
model.eval()

# Were defaulting TOP_K to 10 for all retrievals. Description explains why
TOP_K = 10


# This tokenizes and also embeds the query
# torch.Tensor -> shape [1, 768]
def embed_query(query: str) -> torch.Tensor:
    tokens = tokenizer(query, return_tensors="pt")
    embeddings = model(**tokens).pooler_output
    return embeddings

def load_embeddings(file_path: str) -> torch.Tensor:
    chunk_embeddings = torch.load(file_path)
    return chunk_embeddings

def retrieve(chunk_embeddings: torch.Tensor,
             query_embeddings: torch.Tensor,
             k: int) -> tuple[torch.Tensor, torch.Tensor]:
    
    # Follows the math of the description
    comparison = query_embeddings @ chunk_embeddings.T

    scores, indices = torch.topk(comparison, k, dim=-1)

    return (scores, indices)


if __name__ == "__main__":

    query = input("Ask a question about baseball: ")

    chunk_embedding_path = "data/embeddings/embeddings_size50.pt"
    chunk_text_path = "data/chunks/chunks_size50.jsonl"

    with torch.inference_mode():
        query_embedding = embed_query(query)
        chunk_embeddings = load_embeddings(chunk_embedding_path)

        scores, indices = retrieve(
            chunk_embeddings,
            query_embedding,
            TOP_K
        )

    """
    This is testing we used to verify its actually retrieving.
    It works exactly as intended.
    """
    # # Load the chunks back in the same order they were embedded
    # with open(chunk_text_path, "r", encoding="utf8") as f:
    #     chunks = [json.loads(line) for line in f]

    # # Remove batch dimension: [1, 10] -> [10]
    # scores = scores.squeeze(0)
    # indices = indices.squeeze(0)

    # print("\nTOP RETRIEVED CHUNKS:\n")

    # for score, index in zip(scores, indices):
    #     idx = index.item()
    #     chunk = chunks[idx]

    #     print(f"Score: {score.item():.4f}")
    #     print(f"Article: {chunk['article']}")
    #     print(f"Text: {chunk['text']}")
    #     print("-" * 80)