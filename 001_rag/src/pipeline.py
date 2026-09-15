# ============================================================
# RAG Reproduction — Run the machine
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 14, 2026
#
# File: pipeline.py
#
# Description:
# Tests our questions from data/eval/baseball_qa.json in a few diff ways:
#   We test all different chunking sizes against the set of 50 questions
#       (4 x 50) -> 200 outputs.
#   We also test just the generator BART with no retrieval context at all.
#       so another 50 outputs.
#   Total of 250 outputs or results from the pipeline.
#   
# 
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

import json
from retriever import embed_query, load_embeddings, retrieve
from generator import (
    retrieve_chunked_text,
    generate_candidates,
    rag_sequence,
    generate_no_retrieval
)

def return_paths(chunk_size: int) -> tuple[str, str]:
    """
    NOTE: chunk_size must be either 10, 25, 50, 100
    """
    chunks = f"data/chunks/chunks_size{chunk_size}.jsonl"
    chunks_embeddings = f"data/embeddings/embeddings_size{chunk_size}.pt"

    return chunks, chunks_embeddings

def load_baseball_qa(qa_path: str) -> list[dict]:
    """
    Reads in the questions from data/eval/baseball_qa.json,
    and cleans the json into a list of dicts
    """
    with open(qa_path, "r", encoding="utf8") as f:
        lines = f.read()

    return json.loads(lines)


def main() -> None:
    baseball_qa_set_path = "data/eval/baseball_qa.json"
    chunk_sizes = [10, 25, 50, 100]
    top_k = 10

    # load all the questions once
    baseball_qa = load_baseball_qa(baseball_qa_set_path)

    results = []

    for ch in chunk_sizes:
        chunks_path, chunks_embeddings_path = return_paths(ch)

        # Load ONCE outside of the loop of questions
        chunks_embeddings = load_embeddings(chunks_embeddings_path)

        for i, qa in enumerate(baseball_qa):
            qa_query = qa["question"]
            query_embedded = embed_query(qa_query)
            retrieved_chunks = retrieve(chunks_embeddings, query_embedded, top_k)
            scores, indices = retrieved_chunks
            chunked_text = retrieve_chunked_text(indices, chunks_path)
            candidates = generate_candidates(qa_query, chunked_text)
            best_candidate_idx = rag_sequence(qa_query, scores, chunked_text, candidates)

            generated_answer = candidates[best_candidate_idx]

            results.append({
                "question_id": i,
                "chunk_size": ch,
                "top_k": top_k,
                "question": qa_query,
                "expected_answer": qa["answer"],

                "retrieved_indices": indices.squeeze(0).tolist(),
                "retriever_scores": scores.squeeze(0).tolist(),
                "retrieved_chunks": chunked_text,

                "candidates": candidates,
                "best_candidate_idx": best_candidate_idx,
                "generated_answer": generated_answer,

                "method": "rag_sequence",
            })
    
    for i, qa in enumerate(baseball_qa):
        qa_query = qa["question"]

        generated_answer = generate_no_retrieval(qa_query)

        results.append({
            "question_id": i,
            "chunk_size": None,
            "top_k": None,
            "question": qa_query,
            "expected_answer": qa["answer"],
            "generated_answer": generated_answer,
            "method": "bart_only",
        })

    with open("data/eval/results.json", "w", encoding="utf8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()