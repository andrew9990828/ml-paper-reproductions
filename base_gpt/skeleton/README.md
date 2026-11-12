# Build your own GPT

This is the blank version of Andrew's GPT. The files and mental model are here;
you fill in the actual implementation. Search for `TODO` and remove each
`NotImplementedError` when that section is done.

## What each file does

| File | What you're building |
| --- | --- |
| `gpt/prepare_data.py` | Split whole stories into training, validation, and test files. |
| `gpt/dataset.py` | Turn text into tokens. Inputs predict the next token. |
| `gpt/attention.py` | Let tokens look at earlier tokens and decide what matters. |
| `gpt/model.py` | Put embeddings, attention, and feed-forward layers together. |
| `gpt/train.py` | Make predictions, measure the error, and update the weights. |
| `gpt/test.py` | Measure how the finished model does on held-out stories. |
| `gpt/generate.py` | Predict one token, append it, and repeat. |

Build them in that order. Training uses validation to guide your choices;
save the test set for evaluating the checkpoint you finally choose.

## Start small

Use Python 3.10+ and a virtual environment with PyTorch, tiktoken, and datasets.
Install a PyTorch build appropriate for your CPU or GPU.

Extract this folder beside `base_gpt`. It has its own data and checkpoints.
From inside `gpt_skeleton`, run scripts like this after filling them in:

```bash
python gpt/prepare_data.py
python gpt/train.py
python gpt/test.py
python gpt/generate.py
```

The default model is deliberately small. Get one batch working first, then
check that training can drive the loss down on that same batch. Scale up later.

## Keep these straight

- **Shapes:** `B` = batch size, `T` = token count, `C` = embedding size, `V` = vocabulary size.
- **Targets:** shift the input by one token. Don't shift it again in training.
- **Loss:** cross-entropy takes raw logits, not probabilities.
- **Evaluation:** `eval()` disables dropout; `no_grad()` disables gradient tracking. Use both.
- **Checkpoint:** save the config with the weights so testing and generation rebuild the same model.

`load_stories` lives in `dataset.py` here so training and testing can share it.
W&B is optional; get the model working before adding logging. No weights, data,
or account information are included.

Learning scaffold based on Andrew Bieber's GPT project. Scaffold and TODOs
prepared with ChatGPT; the core implementation is left for the learner.
