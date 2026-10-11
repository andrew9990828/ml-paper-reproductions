# =======================================================================
# My GPT from scratch - Split the data
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: October 10, 2026
#
# File: prepare_data.py
#
# Description:
# This is where we split the data for training/validation/testing.
#
# NOTE: This was delegated to ChatGPT 5.6 Sol. Not really vital to
# write this script manually. It isn't fundamental to my understanding 
# of my GPT. 
# =======================================================================

import random
from pathlib import Path
from datasets import load_dataset

# -------------------- CONFIGURATION --------------------

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
MAX_STORIES = None

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "gpt_data"
EOS_TOKEN = "<|endoftext|>"

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# -------------------- LOAD DATA --------------------

print("Loading TinyStories...")

dataset = load_dataset(
    "skeskinen/TinyStories-hf",
    split="train",
    streaming=True
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED)

counts = {
    "train": 0,
    "val": 0,
    "test": 0
}

paths = {
    name: OUTPUT_DIR / f"{name}.txt"
    for name in counts
}

# -------------------- SPLIT DATA --------------------

print("Splitting stories...")

with (
    paths["train"].open("w", encoding="utf-8") as train_file,
    paths["val"].open("w", encoding="utf-8") as val_file,
    paths["test"].open("w", encoding="utf-8") as test_file
):

    files = {
        "train": train_file,
        "val": val_file,
        "test": test_file
    }

    for index, row in enumerate(dataset):

        if MAX_STORIES is not None and index >= MAX_STORIES:
            break

        story = row["text"].strip()

        if not story:
            continue

        # Randomly assign the entire story to one dataset
        p = rng.random()

        if p < TRAIN_RATIO:
            split = "train"

        elif p < TRAIN_RATIO + VAL_RATIO:
            split = "val"

        else:
            split = "test"

        # Preserve story boundary using GPT-2 EOS
        files[split].write(story)
        files[split].write(f"\n{EOS_TOKEN}\n")

        counts[split] += 1

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1:,} records")

# -------------------- VERIFY --------------------

total = sum(counts.values())

assert total > 0, "No stories were written!"

print("\nData preparation complete!")
print(f"Total stories: {total:,}\n")

for name, count in counts.items():
    percentage = (count / total) * 100

    print(
        f"{name:5s}: {count:>10,} stories "
        f"({percentage:.2f}%)"
    )

    print(f"       Saved to: {paths[name]}")

assert sum(counts.values()) == total

print("\nAll splits successfully created.")