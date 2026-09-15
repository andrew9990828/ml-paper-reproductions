# RAG Reproduction

Paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)  
Lewis et al., 2020

## Goal

Reproduce the core RAG-Sequence architecture from the original paper using a small baseball Wikipedia corpus.

The point of this project is not to build a production RAG system. The goal is to understand the architecture by implementing the major pieces myself and testing how retrieval changes generation.

Main goals:

- implement dense retrieval with DPR
- implement RAG-Sequence generation with BART
- compare generation with and without retrieval
- test how chunk size affects retrieval and final QA performance
- separate retrieval failures from generator failures

A second version of the repo will include a blank implementation skeleton so someone else can reproduce the project themselves.

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
Top-K Retrieved Chunks
   ↓
BART Candidate Generation
   ↓
RAG-Sequence Scoring
   ↓
Final Answer
```

---

## Dataset

The knowledge base contains 10 Wikipedia articles about baseball rules:

- Balk
- Base on balls
- Designated hitter
- Force play
- Infield fly rule
- Inning
- Rules of baseball
- Strike zone
- Strikeout
- Tag out

The raw articles are stored as `.txt` files.

```text
data/
├── raw/
├── chunks/
├── embeddings/
└── eval/
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

Each chunk is stored as JSONL:

```json
{
  "chunk": 1,
  "article": "strike_zone.txt",
  "text": "..."
}
```

This creates four versions of the same knowledge base while changing only chunk size.

The chunk files are:

```text
chunks_size10.jsonl
chunks_size25.jsonl
chunks_size50.jsonl
chunks_size100.jsonl
```

---

## DPR Context Embeddings

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

The four embedding matrices are saved as PyTorch `.pt` files:

```text
embeddings_size10.pt
embeddings_size25.pt
embeddings_size50.pt
embeddings_size100.pt
```

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

retrieval is performed using a matrix multiplication:

```python
scores = query_embedding @ chunk_embeddings.T
```

which produces:

```text
[1, N]
```

or one similarity score for every stored chunk.

The most relevant chunks are selected with:

```python
scores, indices = torch.topk(scores, k=TOP_K)
```

For the main experiment:

```python
TOP_K = 10
```

is kept constant.

This lets chunk size change while retrieval depth stays the same.

---

## BART Generator

The generator uses:

```text
facebook/bart-large
```

For each retrieved chunk, BART receives:

```text
question + retrieved chunk
```

and generates a candidate answer.

With `k = 10`, this initially produces 10 candidate answers.

```text
query + chunk 1  -> candidate 1
query + chunk 2  -> candidate 2
...
query + chunk 10 -> candidate 10
```

---

## RAG-Sequence

Each candidate is then scored against every retrieved chunk.

For one candidate:

```text
candidate
    ↓
score using chunk 1
score using chunk 2
score using chunk 3
...
score using chunk 10
```

The original RAG-Sequence equation is:

```text
p(y | x) = Σ p(z | x) * p(y | x, z)
```

where:

```text
x = question
z = retrieved chunk
y = candidate answer
```

The DPR retrieval scores are converted to retrieval log probabilities using:

```python
torch.log_softmax(...)
```

BART provides:

```text
log p(candidate | question, chunk)
```

For each chunk:

```text
retrieval log probability
+
candidate log probability
```

The chunk scores are combined using:

```python
torch.logsumexp(...)
```

This produces one final RAG-Sequence score for each candidate.

```text
candidate 1 -> one RAG score
candidate 2 -> one RAG score
...
candidate 10 -> one RAG score
```

The candidate with the highest score is selected with:

```python
torch.argmax(...)
```

---

## Evaluation Dataset

The evaluation set contains 50 baseball questions:

```text
10 source articles
×
5 questions per article
=
50 questions
```

Stored in:

```text
data/eval/baseball_qa.json
```

Example:

```json
{
  "id": 1,
  "question": "What is a balk?",
  "answer": "A balk is...",
  "expected_source": "balk.txt"
}
```

The `answer` field is the gold/reference answer.

Later experiment results will also contain:

```text
baseline_answer
rag_answer
```

These are model outputs and are separate from the gold answer.

---

## Main Experiments

### 1. BART Baseline vs RAG

Compare:

```text
BART alone
vs.
BART + retrieval
```

The baseline is run once across all 50 questions.

The RAG system is tested across all four chunk sizes.

This gives:

```text
50 baseline runs

50 questions × 4 chunk sizes
=
200 RAG runs
```

Total:

```text
250 QA outputs
```

Main question:

> Does retrieval improve answer quality compared with BART alone?

---

### 2. Chunk Size

The same 50 questions are run against:

```text
10-word chunks
25-word chunks
50-word chunks
100-word chunks
```

Everything else stays constant:

```text
same corpus
same questions
same DPR models
same BART model
TOP_K = 10
```

Main question:

> How does chunk size affect retrieval and final answer quality?

---

## Metrics

The experiment will record enough raw information to separate retrieval performance from generation performance.

Possible metrics include:

- retrieval Hit@10
- expected-source rank
- mean reciprocal rank
- RAG answer accuracy
- BART baseline accuracy
- RAG improvement over baseline
- retrieval failures
- generator failures

This lets us distinguish:

```text
Retriever Failure
        vs.
Generator Failure
```

For example:

```text
correct article not retrieved
        ↓
retriever failure
```

versus:

```text
correct information retrieved
but final answer is wrong
        ↓
generator failure
```

---

## Experiment Results

All experiment outputs will be stored in one structured results file:

```text
results/experiment_results.json
```

Each question can store:

- gold answer
- expected source
- baseline answer
- retrieved sources
- retrieved chunk text
- retrieval scores
- expected-source rank
- RAG answer
- winning candidate
- RAG score
- chunk size

This allows analysis and graphing without rerunning the models.

The same results file can later be filtered by:

```text
chunk_size
expected_source
retrieval_success
answer_correct
retrieval_rank
```

---

## Planned Visualizations

The main result will be a bar graph comparing QA accuracy across:

```text
BART Baseline
10-word RAG
25-word RAG
50-word RAG
100-word RAG
```

Separate retrieval graphs can compare the four RAG configurations using metrics such as:

```text
Hit@10
MRR
Expected-source rank
```

---

## Optional Top-K Experiment

After the chunk-size experiment, chunk size can be held constant while changing:

```text
k = 1
k = 3
k = 5
k = 10
k = 20
```

This would test retrieval depth separately from chunk size.

---

## Build Progress

```text
Wikipedia Fetching       ✅
Chunking                 ✅
DPR Context Embeddings   ✅
DPR Question Encoder     ✅
Dense Retriever          ✅
Top-K Retrieval          ✅
BART Candidate Generation ✅
RAG-Sequence Scoring     ✅
Evaluation Dataset       ✅
End-to-End Pipeline      ✅
Automated Evaluation     ✅
Experiments              ✅
Graphs / Analysis        ✅
Final Write-Up           ✅
```

---

## Core Mental Model

RAG sounds more complicated than it really is.

```text
Question
   ↓
Find useful information
   ↓
Generate possible answers using that information
   ↓
Score those answers using retrieval + generation
   ↓
Return the best answer
```

The retriever answers:

> "Where should I look?"

The generator answers:

> "Given what I found, what should I say?"

RAG-Sequence combines both.