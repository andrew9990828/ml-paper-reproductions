"""Give the GPT a story beginning. Let it predict what comes next."""
from pathlib import Path
import torch
import tiktoken
from model import AndrewsGPT

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_DIR / "checkpoints" / "best.pt"
MAX_NEW_TOKENS = 200


def main():
    # TODO 1: Choose the device. Load the checkpoint on cpu with weights_only=True.
    # Rebuild AndrewsGPT using saved["config"], then load its model_state_dict.
    # Move the model to device. Use eval() and no_grad() for generation.

    # TODO 2: Ask for a prompt. Reject empty/whitespace-only input.
    # Encode with the GPT-2 tokenizer -> long tensor -> add a batch dimension.
    # Shape is [1,T]. Assign the result of .to(device) back to the tensor.

    # TODO 3: Repeat at most MAX_NEW_TOKENS times:
    # Take the LAST context_length tokens inside the loop; keep the full history.
    # Feed that tensor to the model. Logits are [1,T,V].
    # Select the final position's logits -> [1,V].

    # TODO 4: Softmax over vocabulary, then torch.multinomial for one token.
    # Result: [1,1]. Sampling picks according to probabilities, not always argmax.
    # If its ID equals tokenizer.eot_token, break before appending it.

    # TODO 5: Append the new token to the FULL history along the time dimension.
    # The next iteration sees the token you just generated.

    # TODO 6: Decode the full sequence once, using a flat list of Python IDs.
    # Print it under a clear "GPT STORY" heading. Individual tokens aren't words.
    # No loss, backward pass, or weight updates happen here.
    raise NotImplementedError("TODO: generate the story")


if __name__ == "__main__":
    main()
