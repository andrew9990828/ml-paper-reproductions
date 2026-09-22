# LoRA From Scratch

## Goal

Build a small GPT-style language model from scratch, train it end-to-end, then implement LoRA manually and compare parameter-efficient fine-tuning against standard training.

The point of this project is not to use a pretrained transformer or hide the architecture behind a high-level framework.

The goals are to understand:

- how raw text becomes training examples
- tokenization
- token and positional embeddings
- causal self-attention
- multi-head attention
- transformer blocks
- next-token prediction
- cross-entropy loss
- training and autoregressive generation
- what LoRA changes inside a transformer
- why low-rank adaptation drastically reduces the number of trainable parameters

---

## Project Plan

### Phase 1 - GPT From Scratch

Build the entire language model pipeline:

```text
raw text
   ↓
tokenization
   ↓
input / target sequences
   ↓
token embeddings
   +
positional embeddings
   ↓
transformer blocks
   ↓
language-model logits
   ↓
cross-entropy loss
   ↓
training
   ↓
autoregressive generation
```

The base model will initially be trained on the TinyStories dataset.

The GPT implementation is being written from scratch so that every major part of the architecture is understood and owned directly.

---

### Phase 2 - LoRA

Once the base GPT is working:

1. Train and save the base GPT.
2. Load the trained model.
3. Freeze the original model parameters.
4. Implement LoRA manually.
5. Attach low-rank adapters to selected transformer projections.
6. Fine-tune the model on a smaller or specialized dataset.
7. Compare LoRA fine-tuning against standard full-model training.

The comparison will include:

- trainable parameter count
- memory usage
- training behavior
- training loss
- generated output
- model behavior after fine-tuning

---

## Current Structure

```text
002_lora/
│
├── gpt/
│   ├── attention.py
│   ├── dataset.py
│   ├── generate.py
│   ├── model.py
│   └── train.py
│
├── gpt_data/
│   └── tiny_stories.txt
│
└── README.md
```

---

## Files

### `dataset.py`

Responsible for:

- loading raw text
- GPT-2 tokenization using `tiktoken`
- creating input and target token sequences
- building the PyTorch dataset
- creating dataloaders for training

### `attention.py`

Responsible for:

- query, key, and value projections
- scaled dot-product attention
- causal masking
- multi-head self-attention

### `model.py`

Responsible for defining the GPT architecture, including:

- token embeddings
- positional embeddings
- transformer blocks
- residual connections
- normalization
- feed-forward networks
- final language-model output projection

### `train.py`

Responsible for:

- initializing the model
- loading training data
- calculating cross-entropy loss
- backpropagation
- optimization
- training progress
- checkpointing trained weights

### `generate.py`

Responsible for autoregressive text generation using a trained GPT model.

Generation follows the standard next-token process:

```text
prompt
   ↓
model
   ↓
next-token logits
   ↓
choose next token
   ↓
append token
   ↓
repeat
```

---

## Dataset

The initial GPT model is trained using the TinyStories dataset:

https://huggingface.co/datasets/roneneldan/TinyStories

TinyStories provides a large corpus of simple natural-language stories that works well for experimenting with smaller language models.

The raw dataset is excluded from Git because of its size.

---

## Training Objective

GPT is trained using next-token prediction.

For a token sequence:

```text
[t0, t1, t2, t3, t4]
```

the model receives:

```text
input:
[t0, t1, t2, t3]

target:
[t1, t2, t3, t4]
```

The model therefore learns:

```text
t0 -> predict t1
t1 -> predict t2
t2 -> predict t3
t3 -> predict t4
```

The loss is calculated across all token positions in the sequence.

---

## LoRA

A normal transformer contains large trainable weight matrices.

For a weight matrix:

```text
W
```

standard fine-tuning updates the entire matrix.

LoRA instead freezes the original matrix and learns a low-rank update:

```text
W' = W + BA
```

where:

```text
A ∈ R^(r × d_in)
B ∈ R^(d_out × r)
```

and:

```text
r << d_in
r << d_out
```

The original matrix remains frozen while only the much smaller matrices `A` and `B` are trained.

This allows the model to adapt while training far fewer parameters.

---

## Long-Term GPT Reuse

The GPT model is being developed inside `002_lora/gpt/` first because LoRA is the initial use case.

Once the implementation is stable and tested, the reusable GPT code will be moved into the repository-level `common/` directory.

The intended future structure is roughly:

```text
ml-paper-reproductions/
│
├── common/
│   └── gpt/
│       ├── attention.py
│       ├── model.py
│       ├── dataset.py
│       └── generation.py
│
├── 001_rag/
│
├── 002_lora/
│
├── 003_...
│
└── ...
```

This allows the same GPT implementation to be reused across future paper reproductions and experiments instead of rebuilding the model every time.

Future projects may reuse or modify the base GPT for:

- LoRA
- linear attention
- FlashAttention
- alternative normalization methods
- different positional encoding methods
- transformer architecture experiments
- fine-tuning experiments
- inference experiments

The long-term goal is for the GPT implementation to become reusable transformer infrastructure for the rest of the repository.

---

## Development Philosophy

The GPT implementation is intentionally being built mostly from first principles.

Instead of copying an existing implementation, each component is being implemented and debugged individually.

The focus is on understanding questions such as:

```text
What shape should this tensor have?

Why is this operation necessary?

What information is flowing through this layer?

What parameters are actually being trained?

How does this component affect the full model?
```

The objective is not only to produce a working GPT.

The objective is to understand the model well enough that future transformer architectures and papers can be implemented by modifying known components rather than treating the transformer as a black box.

---

## Status

### GPT

- [x] TinyStories dataset downloaded
- [x] GPT-2 tokenizer selected
- [ ] Dataset implementation
- [ ] Input / target sequence generation
- [ ] Dataloader
- [ ] Single-head causal attention
- [ ] Multi-head causal attention
- [ ] Feed-forward network
- [ ] Transformer block
- [ ] Full GPT model
- [ ] Training loop
- [ ] Model checkpointing
- [ ] Autoregressive generation
- [ ] Train GPT on TinyStories
- [ ] Evaluate generated output

### LoRA

- [ ] Study LoRA paper
- [ ] Implement low-rank adapter layer
- [ ] Freeze base GPT
- [ ] Insert LoRA adapters
- [ ] Verify trainable parameter count
- [ ] Fine-tune using LoRA
- [ ] Compare against full fine-tuning
- [ ] Record results

### Repository Infrastructure

- [ ] Stabilize GPT implementation
- [ ] Move reusable GPT code into `common/gpt/`
- [ ] Reuse the base GPT implementation in future paper reproductions