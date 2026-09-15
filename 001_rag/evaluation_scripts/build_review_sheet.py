# ============================================================
# RAG Reproduction — Build failure review sheet
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 15, 2026
#
# File: build_review_sheet.py
#
# Description:
# This file creates a CSV that makes the most interesting RAG
# failures easy to manually inspect.
#
# I delegated the first version of this evaluation helper to
# GPT-5.6 Sol.
#
# The rows are sorted by "oracle gap":
#
#   best candidate F1 - selected answer F1
#
# A large gap means the candidate generator produced something
# much closer to the expected answer, but RAG-Sequence selected
# a worse candidate.
#
# This is NOT an automatic failure taxonomy. Retrieval failure,
# generation failure, and selection failure still need human
# judgment if I want to make strong claims in the writeup.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a manual RAG failure-review CSV."
    )

    parser.add_argument(
        "--metrics",
        default="data/eval/metrics/per_question_metrics.json",
        help="Path created by evaluate_results.py",
    )

    parser.add_argument(
        "--results",
        default="data/eval/results.json",
        help="Original results.json",
    )

    parser.add_argument(
        "--output",
        default="data/eval/failure_review.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows to include",
    )

    args = parser.parse_args()

    with Path(args.metrics).open("r", encoding="utf8") as f:
        metrics = json.load(f)

    with Path(args.results).open("r", encoding="utf8") as f:
        results = json.load(f)

    raw_lookup = {
        (
            row["method"],
            row.get("chunk_size"),
            row["question_id"],
        ): row
        for row in results
    }

    rag_rows = [
        row
        for row in metrics
        if row["method"] == "rag_sequence"
    ]

    rag_rows.sort(
        key=lambda row: (
            row["oracle_gap"],
            row["oracle_token_f1"],
        ),
        reverse=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "question_id",
        "chunk_size",
        "question",
        "expected_answer",
        "generated_answer",
        "selected_token_f1",
        "selected_candidate_idx",
        "oracle_candidate",
        "oracle_candidate_idx",
        "oracle_token_f1",
        "oracle_gap",
        "top_retrieved_chunk",
        "second_retrieved_chunk",
        "manual_failure_type",
        "manual_notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for metric_row in rag_rows[:args.limit]:
            key = (
                metric_row["method"],
                metric_row["chunk_size"],
                metric_row["question_id"],
            )

            raw = raw_lookup[key]
            chunks = raw.get(
                "retrieved_chunks",
                [],
            )

            writer.writerow({
                "question_id": metric_row["question_id"],
                "chunk_size": metric_row["chunk_size"],
                "question": metric_row["question"],
                "expected_answer": metric_row["expected_answer"],
                "generated_answer": metric_row["generated_answer"],
                "selected_token_f1": round(
                    metric_row["token_f1"],
                    4,
                ),
                "selected_candidate_idx": metric_row[
                    "selected_candidate_idx"
                ],
                "oracle_candidate": metric_row[
                    "oracle_candidate"
                ],
                "oracle_candidate_idx": metric_row[
                    "oracle_candidate_idx"
                ],
                "oracle_token_f1": round(
                    metric_row["oracle_token_f1"],
                    4,
                ),
                "oracle_gap": round(
                    metric_row["oracle_gap"],
                    4,
                ),
                "top_retrieved_chunk": (
                    chunks[0]
                    if len(chunks) > 0
                    else ""
                ),
                "second_retrieved_chunk": (
                    chunks[1]
                    if len(chunks) > 1
                    else ""
                ),
                "manual_failure_type": "",
                "manual_notes": "",
            })

    print(f"Saved review sheet: {output_path}")


if __name__ == "__main__":
    main()
