# RAG Reproduction

Paper: https://arxiv.org/pdf/2005.11401

## Goal

Build a small proof-of-concept reproduction of the original RAG architecture using a small Wikipedia corpus.

Main experiment:

> Compare open-domain question answering with and without retrieval.

## Architecture

```text
Wikipedia Articles
        ↓
     Chunking
        ↓
   DPR Embeddings
        ↓
  Document Index

Question
   ↓
DPR Question Embedding
   ↓
Similarity Search
   ↓
Top-K Retrieved Chunks
   ↓
BART Generator
   ↓
Answer
```

## Stage 1 — Build the Dataset

- Download around 10 Wikipedia articles as `.txt` files.
- Store each article separately in `data/raw/`.
- Preserve the source article for every chunk created later.

Example:

```text
data/
└── raw/
    ├── article_1.txt
    ├── article_2.txt
    ├── article_3.txt
    └── ...
```

## Stage 2 — Chunk the Documents

Build a configurable chunking pipeline.

Initial setup:

```text
chunk_size = 100 words
overlap = 0
```

Process each document separately and create as many chunks as possible.

Store the result in:

```text
data/processed/chunks.json
```

Each chunk should contain:

- chunk ID
- source document
- chunk text

Example:

```json
{
  "chunk_id": 0,
  "source": "article_1.txt",
  "text": "..."
}
```

Later, experiment with different chunk sizes and overlap.

## Stage 3 — Embed the Chunks

Use the pretrained DPR context encoder.

Pipeline:

```text
Chunk Text
    ↓
DPR Tokenizer
    ↓
Token IDs
    ↓
DPR Context Encoder
    ↓
Embedding Vector
```

Store all chunk embeddings as a PyTorch tensor.

Conceptually:

```text
num_chunks × embedding_dimension
```

## Stage 4 — Build the Retriever

Given a question:

```text
Question
    ↓
DPR Question Tokenizer
    ↓
DPR Question Encoder
    ↓
Query Embedding
```

Compare the query embedding against every stored chunk embedding.

Use PyTorch to:

1. Calculate similarity scores.
2. Rank the chunks.
3. Return the top-k most relevant chunks.

Conceptually:

```python
scores = query_embedding @ document_embeddings.T
```

Then:

```python
top_scores, top_indices = torch.topk(scores, k)
```

### First Major Milestone

Given a question, print:

- top-k retrieved chunks
- source document
- similarity score
- chunk text

Do **not** add generation until retrieval works properly.

## Stage 5 — Add the Generator

Load pretrained BART weights.

Feed the generator the question and retrieved context.

Basic pipeline:

```text
Question
   ↓
Retriever
   ↓
Top-K Chunks
   ↓
Question + Retrieved Context
   ↓
BART
   ↓
Answer
```

For the first proof of concept, focus on getting retrieval-conditioned generation working before worrying about reproducing every decoding detail from the paper.

## Stage 6 — Open-Domain QA Experiment

Create a small question dataset with known answers.

Example:

```json
{
  "question": "Who created ...?",
  "answer": "...",
  "expected_source": "article_3.txt"
}
```

Compare:

```text
BART Alone
vs.
BART + RAG
```

Track:

- answer accuracy
- whether the correct passage was retrieved
- retrieval ranking
- retrieved similarity score

This lets us separate:

```text
Retriever Failure
vs.
Generator Failure
```

## Stage 7 — Chunking Experiments

Keep the models and questions the same.

Change only the chunking strategy.

Example chunk sizes:

```text
50 words
100 words
200 words
```

Optionally test overlap:

```text
0 words
20 words
50 words
```

Measure how chunking affects:

- retrieval accuracy
- retrieval ranking
- final QA accuracy

Main question:

> How much does the way we split external knowledge affect RAG performance?

## Stage 8 — Top-K Experiments

Test different numbers of retrieved passages.

Example:

```text
k = 1
k = 3
k = 5
```

Measure whether retrieving more context:

- improves retrieval coverage
- improves answer accuracy
- introduces irrelevant context
- hurts generation quality

## Stage 9 — Results and Write-Up

Document:

- architecture
- dataset
- chunking pipeline
- retriever implementation
- generator integration
- retrieval examples
- RAG vs. no-RAG results
- chunk-size results
- top-k results
- failure cases
- differences from the original paper

## Core Mental Model

```text
External Knowledge
      ↓
Retrieve Relevant Information
      ↓
Give It To The Generator
      ↓
Generate A Better Answer
```

The retriever and external document index provide the **non-parametric memory**.

The generator's learned weights provide the **parametric memory**.

## Build Order

```text
Raw Text
   ↓
Chunks
   ↓
Embeddings
   ↓
Retriever
   ↓
Generation
   ↓
Experiments
   ↓
Results
```

Do not move to the next stage until the current stage works and makes sense.