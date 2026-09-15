# RAG-Sequence Reproduction — Starter Skeleton

Paper: **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
Lewis et al., 2020

This folder is the learner version of the completed RAG reproduction.

The point is **not** to build a production RAG app or hide everything behind a framework call.

The point is to rebuild the core pipeline yourself:

```text
chunk
  ↓
embed
  ↓
retrieve
  ↓
generate candidates
  ↓
score the same candidate under every retrieved chunk
  ↓
marginalize with RAG-Sequence
  ↓
select the final answer
```

## What Is Already Given to You

You are given:

- file structure
- model names
- function signatures
- I/O plumbing where it is not the learning objective
- tensor-shape hints
- the major RAG-Sequence equations
- strategic comments around the difficult parts
- the raw Wikipedia corpus
- the evaluation question set

You are **not** given the core implementation logic.

The TODOs are intentionally concentrated around the parts worth understanding.

## Recommended Order

Work through the files in this order:

```text
1. chunker.py
2. embedder.py
3. retriever.py
4. generator.py
5. pipeline.py
```

`fetch_wikipedia.py` is intentionally treated as plumbing.

The original project delegated that one-off API work because learning the MediaWiki API was not the objective.

The raw Wikipedia files are already included, so you do not need to run `fetch_wikipedia.py` unless you want to rebuild the corpus yourself.

## Core Models

Context encoder:

```text
facebook/dpr-ctx_encoder-single-nq-base
```

Question encoder:

```text
facebook/dpr-question_encoder-single-nq-base
```

Generator:

```text
google/flan-t5-large
```

The original RAG paper used BART as its generator.

This simplified reproduction instead uses FLAN-T5-large as the standalone seq2seq generator because it is instruction tuned and works reliably for question answering without reproducing the paper's full downstream fine-tuning setup.

The retrieval logic, candidate scoring, and RAG-Sequence marginalization are still implemented manually.

## Shapes You Should Keep Checking

A chunk embedding:

```text
[768]
```

All chunk embeddings:

```text
[N, 768]
```

One query embedding:

```text
[1, 768]
```

Similarity against every chunk:

```text
[1, 768] @ [768, N] -> [1, N]
```

Top-k scores and indices:

```text
[1, k]
```

If the shapes stop making sense, stop coding and trace them again.

## RAG-Sequence Objective

For a complete candidate answer `y`:

```text
p(y | x) = Σ_z p(z | x) p(y | x, z)
```

where:

```text
x = question
z = retrieved chunk
y = complete candidate answer
```

In log-space, the implementation needs the equivalent of:

```text
logsumexp(
    log p(z | x)
    +
    log p(y | x, z)
)
```

The important detail is that the **same complete candidate** is scored against every retrieved chunk before marginalization.

That is what makes this RAG-Sequence rather than simply:

```text
retrieve one chunk
→ generate one answer
→ return it
```

## Rules for Using the Skeleton

Try not to open the completed implementation until you have genuinely attempted the TODO.

If you get stuck:

1. write down the tensor shapes
2. write down what information the function receives
3. write down what shape or information it must return
4. trace the data through the pipeline
5. only then look up syntax

The point of the project is to make the architecture obvious by forcing you to build the important pieces yourself.

## Dependencies

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Running the Project

Run all commands from the root of the skeleton directory.

Do **not** `cd` into `src/` first.

Correct:

```bash
python src/chunker.py
python src/embedder.py
python src/retriever.py
python src/pipeline.py
```

The scripts use paths such as:

```text
data/raw/
data/chunks/
data/embeddings/
data/eval/
```

These paths are relative to the project root.

## Expected Data Layout

The skeleton uses:

```text
data/
├── sources.json
├── raw/
├── chunks/
├── embeddings/
└── eval/
    └── baseball_qa.json
```

The following are already provided:

```text
data/sources.json
data/raw/*
data/eval/baseball_qa.json
```

You generate the chunk files yourself:

```text
data/chunks/chunks_size10.jsonl
data/chunks/chunks_size25.jsonl
data/chunks/chunks_size50.jsonl
data/chunks/chunks_size100.jsonl
```

Then generate the embedding files:

```text
data/embeddings/embeddings_size10.pt
data/embeddings/embeddings_size25.pt
data/embeddings/embeddings_size50.pt
data/embeddings/embeddings_size100.pt
```

The completed project used:

```text
chunk sizes: 10, 25, 50, 100 words
top-k:       10
```

Do not confuse the chunk sizes with tokenizer tokens.

The chunker in this reproduction uses whitespace-separated **word counts**.

## Suggested Target

If you already know basic Python, PyTorch, and transformer fundamentals:

```text
~15–25 focused hours
```

is a reasonable target for the full exercise.

That is not a speedrun requirement.

The value is in understanding why each operation exists.

## Final Check

By the end, you should be able to explain this without code:

> How does one query become a final RAG-Sequence answer?

You should be able to trace:

```text
raw text
  ↓
chunks
  ↓
chunk embeddings
  ↓
query embedding
  ↓
similarity scores
  ↓
top-k retrieval
  ↓
candidate generation
  ↓
candidate log probabilities
  ↓
retrieval + generation scores
  ↓
RAG-Sequence marginalization
  ↓
argmax
  ↓
final answer
```

If you can explain the shapes and probabilities all the way from the query to the final `argmax`, you got the point of the exercise.