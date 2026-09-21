import tiktoken
from datasets import load_dataset

# ds -> dictionary
# This was used once to load the 18million lines of text
ds = load_dataset("roneneldan/TinyStories")

# Write into gpt_data so we have the dataset to train later.
with open("002_lora/gpt_data/tiny_stories.txt", "w", encoding="utf8") as f:
    for example in ds["train"]:
        f.write(example["text"])


tokenizer = tiktoken.encoding_for_model("gpt2")

if __name__ == "__main__":

