# Evaluation scripts

These scripts evaluate the saved `data/eval/results.json` experiment.

They are **not unit tests**. They are the evaluation harness for the actual
RAG experiment outputs.

## 1. Run the metrics

From the `001_rag` repo root:

```bash
python evaluation/evaluate_results.py
```

Outputs:

```text
data/eval/metrics/
├── metrics_summary.json
└── per_question_metrics.json
```

The main metrics are:

- normalized exact match
- token F1
- answer containment
- generated answer length
- oracle candidate token F1
- selected-vs-oracle gap
- candidate-index selection distribution

Exact match is expected to be harsh here because the reference answers are
usually full sentences while the model often returns concise answers.

## 2. Make the plots

```bash
python evaluation/plot_results.py
```

Outputs:

```text
data/eval/plots/
├── token_f1_by_configuration.png
├── selected_vs_oracle_f1.png
├── candidate_index_histogram.png
└── answer_length_by_configuration.png
```

## 3. Build a manual failure-review sheet

```bash
python evaluation/build_review_sheet.py
```

This creates:

```text
data/eval/failure_review.csv
```

The CSV is sorted by oracle gap so the most interesting candidate-selection
failures show up first.

For the writeup, `manual_failure_type` can be filled with labels such as:

- retrieval
- generation
- selection
- ambiguous / mixed

Do not call any automatic retrieval metric "Recall@K" unless the dataset has
ground-truth supporting passage IDs.
