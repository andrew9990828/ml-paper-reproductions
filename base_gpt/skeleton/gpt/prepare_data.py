"""Split the stories before the model ever sees them."""
import random
from pathlib import Path
from datasets import load_dataset

SEED = 42
MAX_STORIES = 10_000  # Start small. Use None for the whole source.
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
EOS_TOKEN = "<|endoftext|>"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "gpt_data"

def main():
    # TODO 1: Load skeskinen/TinyStories-hf, split="train", streaming=True.
    # We're making our own three splits from this source.

    # TODO 2: Make OUTPUT_DIR and open train.txt, val.txt, and test.txt.
    # Opening with "w" overwrites files. Don't rerun over data you want to keep.

    # TODO 3: Use random.Random(SEED). For each nonempty story, draw a number.
    # Below 0.8 -> train; below 0.9 -> val; otherwise -> test.
    # Keep the WHOLE story in one split. Splitting token windows can leak data.

    # TODO 4: Write the story, then a newline, EOS_TOKEN, and another newline.
    # Stop at MAX_STORIES if set. Count stories and print the split sizes.
    # Check: do the three counts add up to the number of stories written?
    raise NotImplementedError("TODO: split and save the stories")


if __name__ == "__main__":
    main()
