# Data

The skeleton expects the same general structure as the completed project:

```text
data/
├── sources.json
├── raw/
├── chunks/
├── embeddings/
└── eval/
    └── baseball_qa.json
```

The source list and QA set are dataset inputs rather than implementation solutions.

Generated files should include:

```text
data/chunks/chunks_size10.jsonl
data/chunks/chunks_size25.jsonl
data/chunks/chunks_size50.jsonl
data/chunks/chunks_size100.jsonl

data/embeddings/embeddings_size10.pt
data/embeddings/embeddings_size25.pt
data/embeddings/embeddings_size50.pt
data/embeddings/embeddings_size100.pt
```

The original chunker uses **word counts**, not tokenizer-token counts.
