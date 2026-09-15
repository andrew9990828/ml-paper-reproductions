# ============================================================
# RAG Reproduction — End-to-end experiment pipeline
#
# Goal:
# Wire together:
#
# query
#   -> query embedding
#   -> dense retrieval
#   -> chunk lookup
#   -> candidate generation
#   -> RAG-Sequence selection
#   -> saved result
# ============================================================

import json
import os
import time

from retriever import (
    embed_query,
    load_embeddings,
    retrieve,
)

from generator import (
    retrieve_chunked_text,
    generate_candidates,
    rag_sequence,
    generate_no_retrieval,
)


def return_paths(
    chunk_size: int,
) -> tuple[str, str]:

    chunks = (
        f"data/chunks/"
        f"chunks_size{chunk_size}.jsonl"
    )

    embeddings = (
        f"data/embeddings/"
        f"embeddings_size{chunk_size}.pt"
    )

    return chunks, embeddings


def load_baseball_qa(
    qa_path: str,
) -> list[dict]:

    with open(
        qa_path,
        "r",
        encoding="utf8",
    ) as f:
        return json.load(f)


def save_results(
    results: list[dict],
    results_path: str,
) -> None:

    with open(
        results_path,
        "w",
        encoding="utf8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
        )


def main() -> None:

    baseball_qa_set_path = (
        "data/eval/baseball_qa.json"
    )

    chunk_sizes = [
        10,
        25,
        50,
        100,
    ]

    top_k = 10

    # Start here while debugging.
    #
    # True:
    #   one RAG question
    #   one generator-only question
    #
    # False:
    #   full experiment
    test_mode = True

    baseball_qa = load_baseball_qa(
        baseball_qa_set_path
    )

    if test_mode:

        chunk_sizes = [100]
        baseball_qa = baseball_qa[:1]

        results_path = (
            "data/eval/results_test.json"
        )

        results = []

    else:

        results_path = (
            "data/eval/results.json"
        )

        if os.path.exists(results_path):

            with open(
                results_path,
                "r",
                encoding="utf8",
            ) as f:
                results = json.load(f)

        else:
            results = []

    # Keep track of runs already completed.
    # This lets a full experiment resume after a crash.
    completed = {
        (
            result["method"],
            result["chunk_size"],
            result["question_id"],
        )
        for result in results
    }

    expected_total = (
        len(chunk_sizes)
        * len(baseball_qa)
        + len(baseball_qa)
    )

    experiment_start = time.time()

    # ========================================================
    # RAG-SEQUENCE
    # ========================================================

    for chunk_size in chunk_sizes:

        chunks_path, embeddings_path = (
            return_paths(chunk_size)
        )

        # Load the full embedding matrix ONCE per configuration,
        # not once per question.
        chunk_embeddings = (
            load_embeddings(
                embeddings_path
            )
        )

        for i, qa in enumerate(
            baseball_qa
        ):

            run_key = (
                "rag_sequence",
                chunk_size,
                i,
            )

            if run_key in completed:
                continue

            question_start = time.time()

            query = qa["question"]

            # TODO 1:
            # Embed the query.
            #
            # Expected:
            #     query_embedding -> [1, 768]

            # TODO 2:
            # Retrieve top-k scores + indices.
            #
            # Expected:
            #     scores  -> [1, k]
            #     indices -> [1, k]

            # TODO 3:
            # Turn indices back into top-k chunk text.

            # TODO 4:
            # Generate one candidate per retrieved chunk.

            # TODO 5:
            # Run RAG-Sequence scoring to get the best candidate index.

            # TODO 6:
            # Recover the final generated answer from the candidates list.

            # --------------------------------------------------------
            # Result bookkeeping is already provided.
            #
            # The learning objective is the RAG pipeline above,
            # not writing experiment JSON.
            # --------------------------------------------------------

            results.append({
                "question_id": i,
                "chunk_size": chunk_size,
                "top_k": top_k,
                "question": query,
                "expected_answer": qa["answer"],

                "retrieved_indices":
                    indices.squeeze(0).tolist(),

                "retriever_scores":
                    scores.squeeze(0).tolist(),

                "retrieved_chunks":
                    chunk_text,

                "candidates":
                    candidates,

                "best_candidate_idx":
                    best_candidate_idx,

                "generated_answer":
                    generated_answer,

                "method":
                    "rag_sequence",
            })

            completed.add(run_key)

            save_results(
                results,
                results_path,
            )

            elapsed = (
                time.time()
                - question_start
            )

            print(
                f"RAG chunk={chunk_size} "
                f"question={i + 1} "
                f"finished in {elapsed:.2f}s "
                f"| saved "
                f"{len(results)}/{expected_total}"
            )

    # ========================================================
    # GENERATOR-ONLY BASELINE
    # ========================================================

    for i, qa in enumerate(
        baseball_qa
    ):

        run_key = (
            "generator_only",
            None,
            i,
        )

        if run_key in completed:
            continue

        question_start = time.time()

        query = qa["question"]

        # TODO:
        # Generate an answer using ONLY the generator.
        #
        # No retrieval.
        # No external context.
        #
        # Store the returned string in:
        #
        #     generated_answer

        results.append({
            "question_id": i,
            "chunk_size": None,
            "top_k": None,
            "question": query,
            "expected_answer": qa["answer"],
            "generated_answer": generated_answer,
            "method": "generator_only",
        })

        completed.add(run_key)

        save_results(
            results,
            results_path,
        )

        elapsed = (
            time.time()
            - question_start
        )

        print(
            f"Generator only "
            f"question={i + 1} "
            f"finished in {elapsed:.2f}s "
            f"| saved "
            f"{len(results)}/{expected_total}"
        )

    total_time = (
        time.time()
        - experiment_start
    )

    print("\n==============================")
    print("Experiment complete.")
    print(f"Results: {results_path}")
    print(
        f"Runtime: "
        f"{total_time / 60:.2f} minutes"
    )
    print("==============================")


if __name__ == "__main__":
    main()