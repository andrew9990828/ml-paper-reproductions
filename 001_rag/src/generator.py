# ============================================================
# RAG Reproduction — Feed the generator
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 13, 2026
#
# File: generator.py
#
# Description:
# This file includes our generator BART.
# There's a handful of objectives we want to hit here:
#   1. Take top-k scores + indices from retriever
#   2. Use indices to recover actual chunk text
#   3. Pair each retrieved chunk with the query
#   4. Feed those into BART
#   5. Generate candidate answer sequence(s)
#   6. Use retrieval relevance + generation likelihood for RAG-Sequence
# 
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

from transformers import BartTokenizer, BartForConditionalGeneration
from embedder import process_all_lines
import torch


tokenizer = BartTokenizer.from_pretrained(
    "facebook/bart-large"
)

model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large"
)

# Change from training mode to inference mode
model.eval()

def retrieve_chunked_text(
    indices: torch.Tensor,
    chunk_path: str
    ) -> list[str]:

    # topk gives shape [1, k], squeeze -> [k], then convert to Python ints
    idxs = indices.squeeze(0).tolist()

    # Read every chunk from the matching JSONL file
    with open(chunk_path, "r", encoding="utf8") as f:
        raw_lines = f.readlines()

    # Convert each JSON string into a dict
    # Order is preserved, so lines[i] matches embedding row i
    lines = process_all_lines(raw_lines)

    retrieved_chunks = []

    # Use top-k indices to grab the actual retrieved chunk text
    for idx in idxs:
        chunk = lines[idx]
        retrieved_chunks.append(chunk["text"])

    return retrieved_chunks

# ============================================================
# RAG-SEQUENCE — SIMPLE VERSION
#
# 1. Retriever gives us top-k chunks + how relevant each chunk was.
#
# 2. BART uses query + retrieved chunks to propose possible answers.
#
# 3. For ONE possible answer, ask BART:
#       "How likely is this SAME answer given chunk 1?"
#       "How likely is this SAME answer given chunk 2?"
#       ...
#
# 4. For each chunk:
#       chunk relevance * BART answer likelihood
#
#    Then add all of those together:
#
#       RAG score(answer) =
#           sum(retrieval_prob * answer_prob_given_chunk)
#
# 5. Do that for every possible answer.
#    Highest RAG score = final answer.
#
# THAT is RAG-Sequence.
# ============================================================

# generate_candidates takes the query + chunk and generates an output for that chunk
# We do this for all chunks we retrived, returning a list of BARTs output.
def generate_candidates(
    query: str,
    retrieved_chunks: list[str]
    ) -> list[str]:
    candidates = []

    for chunk in retrieved_chunks:
        input_text = f"question: {query} context: {chunk}"

        tokens = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        )

        output_ids = model.generate(**tokens, max_new_tokens=50)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        candidates.append(answer)

    return candidates

# We return log probability of the input_text vs candidate, but we STORE it
# as a tensor(float) for the math later. Easier to keep as a tensor.
def candidate_log_prob(
    query: str,
    chunk: str,
    candidate: str
    ) -> torch.Tensor:

    input_text = f"question: {query} context: {chunk}"

    input_tokens = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    )

    candidate_tokens = tokenizer(
        candidate,
        return_tensors="pt",
        truncation=True
    )

    outputs = model(
        **input_tokens,
        labels=candidate_tokens.input_ids
    )

    loss = outputs.loss
    num_candidate_tokens = candidate_tokens.input_ids.shape[1]
    logp = -(loss * num_candidate_tokens)
    return logp

# Get the summed probabilities
def rag_sequence(
    query: str,
    retriever_scores: torch.Tensor,
    chunks: list[str],
    candidates: list[str]
    ) -> int:

    retriever_log_probs = torch.log_softmax(retriever_scores, dim=-1)
    list_retriever_probs = retriever_log_probs.squeeze(0).tolist()

    cad_scores = []

    for candidate in candidates:
        chunk_scores = []

        for i, chunk in enumerate(chunks):
            logp = candidate_log_prob(query, chunk, candidate)
            combined = list_retriever_probs[i] + logp
            chunk_scores.append(combined)

        cad_score = torch.logsumexp(
            torch.stack(chunk_scores),
            dim=0,
        )

        cad_scores.append(cad_score)

    return torch.argmax(torch.stack(cad_scores)).item()


def generate_no_retrieval(query: str) -> str:
    input_text = f"question: {query}"

    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    )

    output_ids = model.generate(
        **tokens,
        max_new_tokens=50
    )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )