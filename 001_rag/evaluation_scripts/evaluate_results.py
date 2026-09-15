# ============================================================
# RAG Reproduction — Evaluate experiment results
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 15, 2026
#
# File: evaluate_results.py
#
# Description:
# This file evaluates the saved experiment outputs in results.json.
#
# I delegated the first version of the evaluation harness to GPT-5.6 Sol.
# The point here is not to hide behind one "accuracy" number, especially
# because my expected answers are usually full sentences while the model
# often gives short answers like "four", "the pitcher", or "yes".
#
# So this script reports a few different things:
#   1. Basic result-file integrity checks
#   2. Normalized Exact Match
#   3. SQuAD-style token F1
#   4. Simple answer containment
#   5. Average generated-answer length
#   6. Oracle candidate F1 for RAG runs
#   7. Gap between the selected candidate and best available candidate
#   8. Which candidate index the RAG scorer selected
#
# The oracle metric is especially useful here:
#
#   if one of the 10 candidates is good but the final selected answer is bad,
#   that points more toward the RAG-Sequence selection/reranking stage.
#
#   if all 10 candidates are bad, that points more toward retrieval and/or
#   generation instead.
#
# This script intentionally does NOT claim retrieval Recall@K because the
# experiment does not contain ground-truth supporting passage IDs.
# ============================================================

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """
    SQuAD-style normalization:
      - lowercase
      - remove punctuation
      - remove articles
      - collapse whitespace
    """
    text = text.lower()
    text = "".join(
        ch if ch not in string.punctuation else " "
        for ch in text
    )
    words = [
        word
        for word in text.split()
        if word not in ARTICLES
    ]
    return " ".join(words)


def exact_match(prediction: str, reference: str) -> float:
    return float(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def containment(prediction: str, reference: str) -> float:
    """
    Gives credit when the normalized short answer appears inside the
    normalized reference, or vice versa.

    This is intentionally simple. It is useful for cases like:
        prediction: "the pitcher"
        reference:  "Traditionally, the designated hitter bats in place
                     of the pitcher."
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)

    if not pred or not ref:
        return 0.0

    return float(
        pred in ref
        or ref in pred
    )


def word_count(text: str) -> int:
    return len(text.strip().split())


def result_key(row: dict) -> tuple:
    return (
        row.get("method"),
        row.get("chunk_size"),
        row.get("question_id"),
    )


def validate_results(rows: list[dict]) -> list[str]:
    """
    Returns human-readable integrity warnings/errors.

    This does not assume one exact experiment size so the evaluation
    script can still be reused if somebody changes the dataset later.
    """
    issues = []

    if not rows:
        return ["results file is empty"]

    seen = set()

    for i, row in enumerate(rows):
        key = result_key(row)

        if key in seen:
            issues.append(
                f"duplicate experiment key at row {i}: {key}"
            )
        seen.add(key)

        required = {
            "question_id",
            "question",
            "expected_answer",
            "generated_answer",
            "method",
        }

        missing = required - row.keys()

        if missing:
            issues.append(
                f"row {i} missing fields: {sorted(missing)}"
            )

        if row.get("method") == "rag_sequence":
            rag_required = {
                "chunk_size",
                "top_k",
                "retrieved_indices",
                "retriever_scores",
                "retrieved_chunks",
                "candidates",
                "best_candidate_idx",
            }

            missing_rag = rag_required - row.keys()

            if missing_rag:
                issues.append(
                    f"RAG row {i} missing fields: {sorted(missing_rag)}"
                )
                continue

            top_k = row["top_k"]
            candidates = row["candidates"]
            best_idx = row["best_candidate_idx"]

            for field in (
                "retrieved_indices",
                "retriever_scores",
                "retrieved_chunks",
                "candidates",
            ):
                values = row.get(field, [])

                if len(values) != top_k:
                    issues.append(
                        f"RAG row {i}: len({field})={len(values)} "
                        f"but top_k={top_k}"
                    )

            if not (0 <= best_idx < len(candidates)):
                issues.append(
                    f"RAG row {i}: invalid best_candidate_idx={best_idx}"
                )
            elif row["generated_answer"].strip() != candidates[best_idx].strip():
                issues.append(
                    f"RAG row {i}: generated_answer does not match "
                    f"candidates[best_candidate_idx]"
                )

    return issues


def build_per_question_metrics(rows: list[dict]) -> list[dict]:
    output = []

    for row in rows:
        generated = row["generated_answer"]
        expected = row["expected_answer"]

        record = {
            "question_id": row["question_id"],
            "method": row["method"],
            "chunk_size": row.get("chunk_size"),
            "question": row["question"],
            "expected_answer": expected,
            "generated_answer": generated,
            "exact_match": exact_match(generated, expected),
            "token_f1": token_f1(generated, expected),
            "containment": containment(generated, expected),
            "answer_words": word_count(generated),
        }

        if row["method"] == "rag_sequence":
            candidate_scores = [
                token_f1(candidate, expected)
                for candidate in row["candidates"]
            ]

            oracle_idx = max(
                range(len(candidate_scores)),
                key=candidate_scores.__getitem__,
            )

            oracle_f1 = candidate_scores[oracle_idx]

            record.update({
                "selected_candidate_idx": row["best_candidate_idx"],
                "oracle_candidate_idx": oracle_idx,
                "oracle_candidate": row["candidates"][oracle_idx],
                "oracle_token_f1": oracle_f1,
                "oracle_gap": oracle_f1 - record["token_f1"],
                "top_retriever_score": row["retriever_scores"][0],
                "retriever_margin_1_2": (
                    row["retriever_scores"][0]
                    - row["retriever_scores"][1]
                    if len(row["retriever_scores"]) > 1
                    else None
                ),
            })

        output.append(record)

    return output


def config_name(row: dict) -> str:
    if row["method"] == "generator_only":
        return "generator_only"

    return f"rag_chunk_{row['chunk_size']}"


def summarize(per_question: list[dict]) -> dict:
    groups = defaultdict(list)

    for row in per_question:
        groups[config_name(row)].append(row)

    summary = {}

    for name, rows in sorted(groups.items()):
        current = {
            "count": len(rows),
            "normalized_exact_match": mean(
                row["exact_match"]
                for row in rows
            ),
            "mean_token_f1": mean(
                row["token_f1"]
                for row in rows
            ),
            "mean_containment": mean(
                row["containment"]
                for row in rows
            ),
            "mean_answer_words": mean(
                row["answer_words"]
                for row in rows
            ),
        }

        rag_rows = [
            row
            for row in rows
            if row["method"] == "rag_sequence"
        ]

        if rag_rows:
            current.update({
                "mean_oracle_token_f1": mean(
                    row["oracle_token_f1"]
                    for row in rag_rows
                ),
                "mean_oracle_gap": mean(
                    row["oracle_gap"]
                    for row in rag_rows
                ),
                "candidate0_selection_rate": mean(
                    row["selected_candidate_idx"] == 0
                    for row in rag_rows
                ),
                "candidate0_or1_selection_rate": mean(
                    row["selected_candidate_idx"] in (0, 1)
                    for row in rag_rows
                ),
                "mean_top_retriever_score": mean(
                    row["top_retriever_score"]
                    for row in rag_rows
                ),
                "mean_retriever_margin_1_2": mean(
                    row["retriever_margin_1_2"]
                    for row in rag_rows
                    if row["retriever_margin_1_2"] is not None
                ),
                "selected_candidate_histogram": dict(
                    sorted(
                        Counter(
                            row["selected_candidate_idx"]
                            for row in rag_rows
                        ).items()
                    )
                ),
            })

        summary[name] = current

    return summary


def print_summary(summary: dict) -> None:
    print("\n=== RESULT SUMMARY ===\n")

    header = (
        f"{'config':<20}"
        f"{'n':>5}"
        f"{'EM':>9}"
        f"{'F1':>9}"
        f"{'contain':>10}"
        f"{'words':>9}"
        f"{'oracle F1':>12}"
        f"{'gap':>9}"
    )

    print(header)
    print("-" * len(header))

    for name, stats in summary.items():
        oracle = stats.get("mean_oracle_token_f1")
        gap = stats.get("mean_oracle_gap")

        print(
            f"{name:<20}"
            f"{stats['count']:>5}"
            f"{stats['normalized_exact_match']:>9.3f}"
            f"{stats['mean_token_f1']:>9.3f}"
            f"{stats['mean_containment']:>10.3f}"
            f"{stats['mean_answer_words']:>9.2f}"
            f"{oracle if oracle is not None else float('nan'):>12.3f}"
            f"{gap if gap is not None else float('nan'):>9.3f}"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved RAG experiment results."
    )

    parser.add_argument(
        "--results",
        default="data/eval/results.json",
        help="Path to results.json",
    )

    parser.add_argument(
        "--output-dir",
        default="data/eval/metrics",
        help="Where evaluation JSON files should be written",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open("r", encoding="utf8") as f:
        rows = json.load(f)

    issues = validate_results(rows)

    print(f"Loaded {len(rows)} experiment rows from {results_path}")

    if issues:
        print("\n=== INTEGRITY WARNINGS ===")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Integrity checks passed.")

    per_question = build_per_question_metrics(rows)
    summary = summarize(per_question)

    print_summary(summary)

    per_question_path = (
        output_dir
        / "per_question_metrics.json"
    )

    summary_path = (
        output_dir
        / "metrics_summary.json"
    )

    with per_question_path.open("w", encoding="utf8") as f:
        json.dump(
            per_question,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with summary_path.open("w", encoding="utf8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {per_question_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
