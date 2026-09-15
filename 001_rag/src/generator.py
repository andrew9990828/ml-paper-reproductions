# ============================================================
# RAG Reproduction — Feed the generator
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 15, 2026
#
# File: generator.py
#
# Description:
# This file includes our generator BART.
# There's a handful of objectives we want to hit here:
#   1. Take top-k scores + indices from retriever
#   2. Use indices to recover actual chunk text
#   3. Pair each retrieved chunk with the query
#   4. Feed those into BART
#   5. Generate candidate answer sequence(s)
#   6. Use retrieval relevance + generation likelihood for RAG-Sequence
#
# Later, I loaded this repo on my actual PC and added CUDA optimization
# with the help of gpt-5.6-sol. If I ran the unoptimized code on my CPU
# on my laptop, it would've taken 15-30hrs. While I only have a NVIDIA 2070
# GeForce RTX, it will take only 1-3hrs on my PC.
# This isn't my strongsuit, so I'm not going to sit here and act like I'm
# a pytorch CUDA wizard when it's my first real exposure to a real inference issue.
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import json


# Use the GPU if we have one, otherwise just fall back to the CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Keep the tokenization on the CPU
tokenizer = BartTokenizer.from_pretrained(
    "facebook/bart-large"
)

# Throw the ~400M parameter model on the GPU.
model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large"
).to(device)

# BART weights are normally FP32, so about 4 bytes per number.
# For inference we can use FP16 instead, cutting that roughly in half.
# This is NOT quantization. We're still using floating point numbers.
if device.type == "cuda":
    model = model.half()

# disable training mode
model.eval()


def retrieve_chunked_text(
    indices: torch.Tensor,
    chunk_path: str
    ) -> list[str]:

    # topk gives shape [1, k], squeeze -> [k], then convert to Python ints
    idxs = indices.squeeze(0).tolist()

    # Read every chunk from the matching JSONL file
    with open(chunk_path, "r", encoding="utf8") as f:
        raw_lines = f.readlines()

    # Convert each JSON string into a dict
    # Order is preserved, so lines[i] matches embedding row i
    lines = [json.loads(line) for line in raw_lines]

    retrieved_chunks = []

    # Use top-k indices to grab the actual retrieved chunk text
    for idx in idxs:
        chunk = lines[idx]
        retrieved_chunks.append(chunk["text"])

    return retrieved_chunks


# ============================================================
# RAG-SEQUENCE — SIMPLE VERSION
#
# 1. Retriever gives us top-k chunks + how relevant each chunk was.
#
# 2. BART uses query + retrieved chunks to propose possible answers.
#
# 3. For ONE possible answer, ask BART:
#       "How likely is this SAME answer given chunk 1?"
#       "How likely is this SAME answer given chunk 2?"
#       ...
#
# 4. For each chunk:
#       chunk relevance * BART answer likelihood
#
#    Then add all of those together:
#
#       RAG score(answer) =
#           sum(retrieval_prob * answer_prob_given_chunk)
#
# 5. Do that for every possible answer.
#    Highest RAG score = final answer.
#
# THAT is RAG-Sequence.
# ============================================================


# generate_candidates takes the query + chunk and generates an output for that chunk
# We do this for all chunks we retrived, returning a list of BARTs output.
@torch.inference_mode()
def generate_candidates(
    query: str,
    retrieved_chunks: list[str]
    ) -> list[str]:

    input_texts = []

    for chunk in retrieved_chunks:
        input_text = f"question: {query} context: {chunk}"
        input_texts.append(input_text)

    # Instead of sending all 10 chunks through BART one at a time,
    # make one batch and let the GPU do the annoying math all at once.
    # padding=True just makes every row the same length so it fits in one tensor.
    tokens = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    output_ids = model.generate(
        **tokens,
        max_new_tokens=50
    )

    # We generated a whole batch, so decode the whole batch too.
    candidates = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )

    return candidates


# We return one log probability for this SAME candidate against every chunk.
# Instead of doing 10 BART calls, we make 10 pairs and batch all the math.
@torch.inference_mode()
def candidate_log_probs(
    query: str,
    chunks: list[str],
    candidate: str
    ) -> torch.Tensor:

    input_texts = []

    for chunk in chunks:
        input_text = f"question: {query} context: {chunk}"
        input_texts.append(input_text)

    # All 10 different question + chunk inputs go in one batch.
    input_tokens = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Same candidate 10 times because we want to ask:
    # "How likely is THIS answer for chunk 1, chunk 2, chunk 3...?"
    candidate_texts = [candidate] * len(chunks)

    candidate_tokens = tokenizer(
        candidate_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Padding is fake data, so -100 tells BART to ignore those spots.
    labels = candidate_tokens.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(
        **input_tokens,
        labels=labels
    )

    # BART gives us scores over the whole vocabulary for every output token.
    # Turn those scores into log probabilities.
    log_probs = torch.log_softmax(
        outputs.logits.float(),
        dim=-1
    )

    # Real candidate token = True
    # Padding = False
    mask = labels != -100

    # gather() can't use -100 as an index, so temporarily replace
    # the padding spots with 0. We throw them away with the mask anyway.
    safe_labels = labels.clone()
    safe_labels[~mask] = 0

    # For every token position, grab the log probability BART gave
    # to the token that was ACTUALLY in our candidate.
    token_log_probs = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Kill the fake padding positions.
    token_log_probs = token_log_probs * mask

    # Add the token log probs together for each row.
    #
    # Output shape:
    # [10]
    #
    # One full candidate log probability for each retrieved chunk.
    return token_log_probs.sum(dim=1)


# Get the summed probabilities
@torch.inference_mode()
def rag_sequence(
    query: str,
    retriever_scores: torch.Tensor,
    chunks: list[str],
    candidates: list[str]
    ) -> int:

    # Turn our raw retriever scores into log probabilities.
    # Keep this as a tensor now because we can do the math on all 10 at once.
    retriever_log_probs = torch.log_softmax(
        retriever_scores.float(),
        dim=-1
    ).squeeze(0).to(device)

    cad_scores = []

    for candidate in candidates:

        # This used to loop through all 10 chunks and call BART 10 times.
        # Now this ONE call gives us all 10 generation scores.
        generation_log_probs = candidate_log_probs(
            query,
            chunks,
            candidate
        )

        # Same idea as before:
        #
        # retrieval score chunk 1 + generation score chunk 1
        # retrieval score chunk 2 + generation score chunk 2
        # ...
        #
        # PyTorch just does all 10 additions at once.
        combined_scores = (
            retriever_log_probs
            + generation_log_probs
        )

        # Combine this candidate's score across all retrieved chunks.
        cad_score = torch.logsumexp(
            combined_scores,
            dim=0
        )

        cad_scores.append(cad_score)

    # Return the index of whichever candidate got the highest RAG score.
    return torch.argmax(
        torch.stack(cad_scores)
    ).item()


@torch.inference_mode()
def generate_no_retrieval(query: str) -> str:

    input_text = f"question: {query}"

    tokens = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    output_ids = model.generate(
        **tokens,
        max_new_tokens=50
    )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )