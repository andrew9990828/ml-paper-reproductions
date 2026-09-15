# ============================================================
# RAG Reproduction — Run the machine
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 15, 2026
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
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

import json
import os
import time

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


def save_results(results: list[dict], results_path: str) -> None:
    # Save after every question so a crash doesn't kill the whole run.
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    baseball_qa_set_path = "data/eval/baseball_qa.json"
    chunk_sizes = [10, 25, 50, 100]
    top_k = 10

    # True = 1 RAG question + 1 BART-only question.
    # False = full 250 output experiment.
    test_mode = False

    # load all the questions once
    baseball_qa = load_baseball_qa(baseball_qa_set_path)

    if test_mode:
        print("\n========== TEST MODE ==========\n")
        chunk_sizes = [10]
        baseball_qa = baseball_qa[:1]
        results_path = "data/eval/results_test.json"
        results = []

    else:
        print("\n========== FULL EXPERIMENT ==========\n")
        results_path = "data/eval/results.json"

        # If the run already started before crashing, load what we have.
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf8") as f:
                results = json.load(f)

            print(f"Found {len(results)} existing results. Resuming...")
        else:
            results = []

    # Keep track of runs we've already finished.
    completed = {
        (result["method"], result["chunk_size"], result["question_id"])
        for result in results
    }

    expected_total = len(chunk_sizes) * len(baseball_qa) + len(baseball_qa)
    experiment_start = time.time()

    # ============================================================
    # RAG-SEQUENCE
    # ============================================================

    for ch in chunk_sizes:
        chunks_path, chunks_embeddings_path = return_paths(ch)

        print(f"\nLoading embeddings for chunk size {ch}...")

        # Load ONCE outside of the loop of questions
        chunks_embeddings = load_embeddings(chunks_embeddings_path)

        for i, qa in enumerate(baseball_qa):
            run_key = ("rag_sequence", ch, i)

            if run_key in completed:
                print(f"[SKIP] RAG | chunk={ch} | question={i + 1}")
                continue

            question_start = time.time()

            print(f"\n[RAG] chunk={ch} | question={i + 1}/{len(baseball_qa)}")

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

            completed.add(run_key)
            save_results(results, results_path)

            elapsed = time.time() - question_start
            print(f"Finished in {elapsed:.2f}s | Saved {len(results)}/{expected_total}")

            if test_mode:
                print(f"Expected:  {qa['answer']}")
                print(f"Generated: {generated_answer}")

    # ============================================================
    # BART-ONLY
    # ============================================================

    for i, qa in enumerate(baseball_qa):
        run_key = ("bart_only", None, i)

        if run_key in completed:
            print(f"[SKIP] BART only | question={i + 1}")
            continue

        question_start = time.time()

        print(f"\n[BART only] question={i + 1}/{len(baseball_qa)}")

        qa_query = qa["question"]
        generated_answer = generate_no_retrieval(qa_query)

        results.append({
            "question_id": i,
            "chunk_size": None,
            "top_k": None,
            "question": qa_query,
            "expected_answer": qa["answer"],
            "generated_answer": generated_answer,
            "method": "generator_only",
        })

        completed.add(run_key)
        save_results(results, results_path)

        elapsed = time.time() - question_start
        print(f"Finished in {elapsed:.2f}s | Saved {len(results)}/{expected_total}")

        if test_mode:
            print(f"Expected:  {qa['answer']}")
            print(f"Generated: {generated_answer}")

    total_time = time.time() - experiment_start

    print("\n====================================")
    print("Experiment complete.")
    print(f"Results saved: {len(results)}")
    print(f"Output: {results_path}")
    print(f"Runtime: {total_time / 60:.2f} minutes")
    print("====================================")


if __name__ == "__main__":
    main()