# ============================================================
# RAG Reproduction — Plot experiment results
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 15, 2026
#
# File: plot_results.py
#
# Description:
# This file makes a few simple graphs from the output of
# evaluate_results.py.
#
# I delegated the first version of this evaluation/plotting code
# to GPT-5.6 Sol.
#
# The graphs are intentionally simple:
#   1. Selected-answer token F1 by configuration
#   2. Selected vs oracle candidate F1
#   3. Which candidate index RAG-Sequence selects
#   4. Average generated-answer length
#
# Run evaluate_results.py first.
# ============================================================

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def config_sort_key(name: str) -> tuple:
    if name == "generator_only":
        return (1, 10**9)

    if name.startswith("rag_chunk_"):
        return (
            0,
            int(name.rsplit("_", 1)[1]),
        )

    return (2, name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot evaluated RAG experiment results."
    )

    parser.add_argument(
        "--metrics-dir",
        default="data/eval/metrics",
        help="Directory created by evaluate_results.py",
    )

    parser.add_argument(
        "--output-dir",
        default="data/eval/plots",
        help="Where PNG plots should be written",
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        metrics_dir
        / "metrics_summary.json"
    ).open("r", encoding="utf8") as f:
        summary = json.load(f)

    with (
        metrics_dir
        / "per_question_metrics.json"
    ).open("r", encoding="utf8") as f:
        rows = json.load(f)

    configs = sorted(
        summary,
        key=config_sort_key,
    )

    labels = [
        (
            "Generator only"
            if name == "generator_only"
            else name.replace("rag_chunk_", "Chunk ")
        )
        for name in configs
    ]

    # --------------------------------------------------------
    # Plot 1: selected-answer token F1
    # --------------------------------------------------------

    values = [
        summary[name]["mean_token_f1"]
        for name in configs
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean token F1")
    plt.xlabel("Configuration")
    plt.title("Answer Quality by Configuration")
    plt.xticks(rotation=20)
    plt.tight_layout()

    path = output_dir / "token_f1_by_configuration.png"
    plt.savefig(path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plot 2: selected answer vs best available candidate
    # --------------------------------------------------------

    rag_configs = [
        name
        for name in configs
        if name.startswith("rag_chunk_")
    ]

    rag_labels = [
        name.replace("rag_chunk_", "Chunk ")
        for name in rag_configs
    ]

    selected = [
        summary[name]["mean_token_f1"]
        for name in rag_configs
    ]

    oracle = [
        summary[name]["mean_oracle_token_f1"]
        for name in rag_configs
    ]

    x = list(range(len(rag_configs)))
    width = 0.38

    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - width / 2 for i in x],
        selected,
        width=width,
        label="Selected answer",
    )
    plt.bar(
        [i + width / 2 for i in x],
        oracle,
        width=width,
        label="Best candidate available",
    )
    plt.xticks(x, rag_labels)
    plt.ylabel("Mean token F1")
    plt.xlabel("Chunk size")
    plt.title("RAG Selection vs Candidate Oracle")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "selected_vs_oracle_f1.png"
    plt.savefig(path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plot 3: winning candidate index by chunk size
    # --------------------------------------------------------

    by_chunk = defaultdict(Counter)

    for row in rows:
        if row["method"] != "rag_sequence":
            continue

        by_chunk[int(row["chunk_size"])][
            int(row["selected_candidate_idx"])
        ] += 1

    chunk_sizes = sorted(by_chunk)

    plt.figure(figsize=(9, 5))

    for chunk_size in chunk_sizes:
        counts = by_chunk[chunk_size]
        indices = sorted(counts)

        plt.plot(
            indices,
            [counts[i] for i in indices],
            marker="o",
            label=f"Chunk {chunk_size}",
        )

    plt.xlabel("Selected candidate index")
    plt.ylabel("Number of selections")
    plt.title("Which Retrieved Candidate Wins?")
    plt.xticks(range(10))
    plt.legend()
    plt.tight_layout()

    path = output_dir / "candidate_index_histogram.png"
    plt.savefig(path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plot 4: generated-answer length
    # --------------------------------------------------------

    lengths = [
        summary[name]["mean_answer_words"]
        for name in configs
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, lengths)
    plt.ylabel("Mean generated words")
    plt.xlabel("Configuration")
    plt.title("Generated Answer Length")
    plt.xticks(rotation=20)
    plt.tight_layout()

    path = output_dir / "answer_length_by_configuration.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
