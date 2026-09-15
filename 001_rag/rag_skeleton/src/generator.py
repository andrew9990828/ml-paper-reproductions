# ============================================================
# RAG Reproduction — Candidate generation + RAG-Sequence scoring
#
# This is the hardest file in the skeleton.
#
# Goal:
#   1. Recover the top-k chunk text.
#   2. Generate one candidate answer per chunk.
#   3. Score the SAME complete candidate under every retrieved chunk.
#   4. Combine generation likelihood with retrieval probability.
#   5. Marginalize across chunks.
#   6. Pick the highest-scoring candidate.
# ============================================================

import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


MODEL_NAME = "google/flan-t5-large"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(device)

# FP16 cuts GPU memory use roughly in half during inference.
# This is still floating-point inference, not quantization.
if device.type == "cuda":
    model = model.half()

model.eval()


def build_prompt(
    query: str,
    chunk: str | None = None,
) -> str:
    """
    Build the instruction-style prompt used by FLAN-T5.

    Generator-only baseline:
        question only

    RAG path:
        question + retrieved context
    """

    if chunk is None:

        # TODO:
        # Build a prompt that asks the model to answer the question
        # in one complete sentence.

        raise NotImplementedError

    # TODO:
    # Build a prompt that:
    #
    #   - asks the model to answer the question
    #   - provides the retrieved chunk as context
    #   - tells the model to use only that context
    #   - requests a complete sentence
    #
    # Keep the format consistent because this exact prompt will also
    # be reused when scoring candidate likelihoods.

    raise NotImplementedError


def retrieve_chunked_text(
    indices: torch.Tensor,
    chunk_path: str,
) -> list[str]:
    """
    Convert top-k embedding-row indices back into actual chunk text.

    Why this works:
        JSONL line i and embedding row i were created in the same order.
    """

    idxs = indices.squeeze(0).tolist()

    with open(
        chunk_path,
        "r",
        encoding="utf8",
    ) as f:
        lines = [
            json.loads(line)
            for line in f
        ]

    # TODO:
    # Use idxs to recover only the matching chunk["text"] values,
    # preserving retrieval order.

    raise NotImplementedError


@torch.inference_mode()
def generate_candidates(
    query: str,
    retrieved_chunks: list[str],
) -> list[str]:
    """
    Generate one candidate answer per retrieved chunk.

    With k=10:

        query + chunk 0 -> candidate 0
        query + chunk 1 -> candidate 1
        ...
        query + chunk 9 -> candidate 9

    Batch the work instead of calling the model separately 10 times.
    """

    # TODO 1:
    # Build one prompt per retrieved chunk.

    # TODO 2:
    # Tokenize the entire list as one batch.
    #
    # You will need:
    #   return_tensors="pt"
    #   padding=True
    #   truncation=True
    #
    # Move the batch onto `device`.

    # TODO 3:
    # Generate candidate token sequences.
    #
    # The original reproduction uses:
    #
    #   max_new_tokens=64

    # TODO 4:
    # Decode the entire generated batch into list[str].

    raise NotImplementedError


@torch.inference_mode()
def candidate_log_probs(
    query: str,
    chunks: list[str],
    candidate: str,
) -> torch.Tensor:
    """
    Score ONE COMPLETE candidate against EVERY retrieved chunk.

    For k=10, return:

        [10]

    where element i is:

        log p(candidate | query, chunk_i)

    Important:

    We are NOT asking only how likely the candidate was under the
    chunk that originally generated it.

    We are asking:

        how likely is this SAME candidate under chunk 0?
        how likely is this SAME candidate under chunk 1?
        ...
        how likely is this SAME candidate under chunk k-1?
    """

    # TODO 1:
    # Build one prompt per chunk using the SAME query.

    # TODO 2:
    # Tokenize those prompts as one padded/truncated batch
    # and move it to `device`.

    # TODO 3:
    # Repeat the SAME candidate once per chunk.
    #
    # Example:
    #
    #   candidate_texts = [candidate] * len(chunks)
    #
    # Tokenize that batch too.

    # TODO 4:
    # Build labels from the candidate token ids.
    #
    # Padding must not contribute to sequence probability.
    #
    # Which label value does Hugging Face use to mean:
    #
    #   "ignore this token position"?

    # TODO 5:
    # Run the seq2seq model with:
    #
    #   model(**input_tokens, labels=labels)
    #
    # The output logits contain one vocabulary distribution for
    # every candidate-token position.

    # TODO 6:
    # Convert logits into LOG probabilities across the vocabulary.

    # TODO 7:
    # For every sequence position, retrieve the log probability
    # assigned to the token that ACTUALLY appears in the candidate.
    #
    # Hint:
    #
    #   torch.gather(...)
    #
    # Be careful:
    # your ignored padding value is not a valid vocabulary index.

    # TODO 8:
    # Mask padding back out.

    # TODO 9:
    # SUM token log probabilities across the sequence dimension.
    #
    # Do NOT length-normalize.
    #
    # RAG-Sequence uses the probability of the complete sequence.

    raise NotImplementedError


@torch.inference_mode()
def rag_sequence(
    query: str,
    retriever_scores: torch.Tensor,
    chunks: list[str],
    candidates: list[str],
) -> int:
    """
    Return the index of the candidate with the highest
    RAG-Sequence score.

    Core equation:

        p(y | x)
        =
        sum_z p(z | x) p(y | x, z)

    Work in log-space.

    For each chunk:

        log p(z | x)
        +
        log p(y | x, z)

    Then marginalize over retrieved chunks with:

        torch.logsumexp(...)

    Repeat this for every candidate and return the index of
    the highest-scoring candidate.
    """

    # TODO 1:
    # Convert raw retriever scores into retrieval LOG probabilities.
    #
    # Expected shape after removing the batch dimension:
    #
    #   [k]
    #
    # Hardware note:
    # Make sure this tensor is on the same `device` as the
    # generation log probabilities.

    # TODO 2:
    # Create a place to store one marginalized score
    # for each candidate.

    # TODO 3:
    # For every candidate:
    #
    #   - call candidate_log_probs(...)
    #   - combine retrieval + generation log probabilities
    #   - marginalize over the k chunks using logsumexp
    #   - save the resulting scalar candidate score

    # TODO 4:
    # Stack the candidate scores and return the index
    # of the maximum.

    raise NotImplementedError


@torch.inference_mode()
def generate_no_retrieval(
    query: str,
) -> str:
    """
    Generator-only baseline.

    No retrieved context is used.
    """

    # TODO 1:
    # Build the no-context prompt.

    # TODO 2:
    # Tokenize it and move it to `device`.

    # TODO 3:
    # Generate with max_new_tokens=64.

    # TODO 4:
    # Decode and return the generated string.

    raise NotImplementedError