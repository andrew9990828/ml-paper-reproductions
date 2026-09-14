# RAG Reproduction

Paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)  
Lewis et al., 2020

## Goal

Reproduce the core RAG-Sequence architecture from the original paper using a small baseball Wikipedia corpus.

The point of this project is not to build a production RAG system.

The goals are to:

- understand the original architecture from the paper
- implement dense retrieval with DPR
- understand and implement RAG-Sequence generation
- compare generation with and without retrieval
- experiment with how chunk size changes retrieval and final QA performance

A second version of this repo will include an empty implementation skeleton so other people can reproduce the project themselves.

---

## Architecture

```text
Wikipedia Articles
        ↓
      Chunking
        ↓
DPR Context Encoder
        ↓
Stored Chunk Embeddings
        ↓

Question
   ↓
DPR Question Encoder
   ↓
Query Embedding
   ↓
Query @ Chunk_Embeddings.T
   ↓
Top-K Chunks
   ↓
RAG-Sequence + BART
   ↓
Answer
```

---

## Dataset

The corpus contains 10 Wikipedia pages about baseball rules.

Examples include:

- Rules of baseball
- Strike zone
- Strikeout
- Base on balls
- Balk
- Force play
- Tag out
- Infield fly rule
- Designated hitter
- Inning

Each article is downloaded and stored locally as raw text.

```text
data/
├── raw/
├── chunks/
└── embeddings/
```

---

## Chunking

The same corpus is chunked using four different fixed word counts:

```text
10 words
25 words
50 words
100 words
```

No overlap is used.

Each chunk is stored as JSONL with:

```json
{
  "chunk": 1,
  "article": "strike_zone.txt",
  "text": "..."
}
```

This gives us four versions of the exact same knowledge base while changing only chunk size.

---

## DPR Embeddings

Each chunk is embedded using the pretrained DPR context encoder:

```text
facebook/dpr-ctx_encoder-single-nq-base
```

Every chunk becomes one vector in:

```text
R^768
```

For `N` chunks, the stored embedding matrix has shape:

```text
[N, 768]
```

The embeddings are saved as PyTorch `.pt` files.

---

## Retriever

Questions are embedded using the paired DPR question encoder:

```text
facebook/dpr-question_encoder-single-nq-base
```

For:

```text
query_embedding  -> [1, 768]
chunk_embeddings -> [N, 768]
```

retrieval is:

```python
scores = query_embedding @ chunk_embeddings.T
```

which produces:

```text
[1, N]
```

or one relevance score for every chunk.

The top chunks are then selected with:

```python
scores, indices = torch.topk(scores, k=TOP_K)
```

For the main chunk-size experiment:

```python
TOP_K = 10
```

is kept constant.

This lets us change chunk size without also changing retrieval depth.

---

## RAG-Sequence Generator

The generator uses pretrained BART:

```text
facebook/bart-large
```

For each retrieved chunk, BART receives:

```text
question + retrieved chunk
```

and generates candidate answer sequences.

RAG-Sequence then scores each candidate using all retrieved documents.

The core equation is:

```text
p(y | x) = Σ p(z | x) * p(y | x, z)
```

Dumbed down:

```text
final answer score =
SUM(
    how relevant the chunk was
    *
    how likely BART thinks the answer is using that chunk
)
```

The candidate with the highest RAG-Sequence score becomes the final answer.

---

## Main Experiments

### 1. RAG vs No Retrieval

Compare:

```text
BART alone
vs.
BART + DPR retrieval
```

Main question:

> Does external retrieved knowledge improve answer quality compared with the generator alone?

---

### 2. Chunk Size

Keep everything else constant:

```text
same corpus
same questions
same DPR models
same BART model
TOP_K = 10
```

Change only:

```text
chunk_size = 10
chunk_size = 25
chunk_size = 50
chunk_size = 100
```

Measure:

- whether the correct information is retrieved
- retrieval ranking
- final answer correctness
- failure cases

Main question:

> How does the amount of information stored in each chunk affect retrieval and generation?

---

### 3. Optional Top-K Experiment

After the chunk-size experiment, hold chunk size constant and vary:

```text
k = 1
k = 3
k = 5
k = 10
k = 20
```

This is a separate experiment so chunk size and retrieval depth are not changed at the same time.

---

## Evaluation

A small baseball QA dataset will contain:

```json
{
  "question": "What is a strike?",
  "answer": "...",
  "expected_source": "strike_zone.txt"
}
```

For each question, we can record:

- expected answer
- expected source
- retrieved chunks
- retrieval rank
- generated answer
- whether retrieval succeeded
- whether generation succeeded

This helps separate:

```text
Retriever Failure
        vs.
Generator Failure
```

---

## Build Progress

```text
Wikipedia Fetching       ✅
Chunking                 ✅
DPR Context Embeddings   ✅
DPR Question Encoder     ✅
Dense Retriever          ✅
Top-K Retrieval          ✅
BART Generation          🚧
RAG-Sequence Scoring     🚧
End-to-End Pipeline      ⬜
Evaluation               ⬜
Experiments              ⬜
Write-Up                 ⬜
```

---

## Core Mental Model

RAG sounds much more complicated than it is.

```text
Question
   ↓
Find useful information
   ↓
Use that information while generating
   ↓
Score the possible answers
   ↓
Return the best one
```

The retriever answers:

> "Where should I look?"

The generator answers:

> "Given what I found, what should I say?"

RAG-Sequence combines both.
````
