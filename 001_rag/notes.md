# RAG Notes

## What I Used to Think RAG Was

I used to think RAG was just grabbing vectors of numbers, matching them to something, then taking a wild guess from a chunk.

## RAG Is Two Things

1. **A Retriever**
2. **A Pretrained Generator**
   - Seq2seq NLP model
   - BART in this paper

## Parametric vs. Non-Parametric Memory

### Parametric Memory

Stores knowledge directly inside a model's internal weights.

In this paper, this is mainly the pretrained generator.

### Non-Parametric Memory

Stores knowledge externally and retrieves it when needed.

In this paper, the indexed Wikipedia passages are the non-parametric memory.

## Lewis et al. and the Team's Finding

> "Finally, we demonstrate that the non-parametric memory can be replaced to update the model's knowledge as the world changes."

This means the external knowledge source can be updated without needing to retrain the entire model.

## How RAG Works

RAG uses the input/query to retrieve relevant documents. Those documents provide additional context to the generator, which then produces the target sequence.

Most importantly, RAG's non-parametric memory is easy to update. The information available to the model can be changed by updating the external documents instead of retraining or swapping the entire model.

## RAG-Sequence vs. RAG-Token

### RAG-Sequence

RAG-Sequence treats a retrieved document as the context for the entire generated sequence.

At a high level:

> Try several retrieved documents, measure how well each one explains the answer, weight each by the retriever's confidence, then combine the results.

Important nuance: RAG-Sequence does **not** just choose one document and forget the rest. It considers the top-k retrieved documents and marginalizes over them, but each candidate document conditions the **entire output sequence**.

The equation is:

$$
p_{\text{RAG-Sequence}}(y \mid x)
\approx
\sum_{z \in \text{top-}k(p(\cdot \mid x))}
p_\eta(z \mid x)
p_\theta(y \mid x, z)
$$

Expanding the probability of generating the full sequence:

$$
p_{\text{RAG-Sequence}}(y \mid x)
\approx
\sum_{z \in \text{top-}k(p(\cdot \mid x))}
p_\eta(z \mid x)
\prod_{i=1}^{N}
p_\theta(y_i \mid x, z, y_{1:i-1})
$$

### RAG-Token

RAG-Token allows different retrieved documents to influence different target tokens.

Instead of marginalizing over documents once at the sequence level, the model combines retrieval probabilities again at every generated token.

The equation is:

$$
p_{\text{RAG-Token}}(y \mid x)
\approx
\prod_{i=1}^{N}
\sum_{z \in \text{top-}k(p(\cdot \mid x))}
p_\eta(z \mid x)
p_\theta(y_i \mid x, z, y_{1:i-1})
$$

### Mental Model

- **RAG-Sequence:** each candidate document tries to explain the whole answer.
- **RAG-Token:** different documents can contribute to different tokens of the answer.

## Retriever and Generator

The original paper uses DPR as the retriever, which follows a bi-encoder architecture.

The indexed Wikipedia passages are the **non-parametric memory** in action.

For the generator, the research group chose BART, although the broader RAG idea is not fundamentally tied to BART.

BART's learned parameters are the **parametric memory**.

Both components start from pretrained models. RAG is then fine-tuned on downstream tasks rather than trained completely from scratch.

During RAG training, the query encoder and generator are updated while the document encoder/index remains fixed.

They use Adam for optimization.

## Decoding

RAG-Sequence and RAG-Token also differ in how decoding works.

### RAG-Sequence Decoder

RAG-Sequence is conceptually pretty simple, but faithful decoding is actually more annoying.

Beam search is run for each retrieved document, producing candidate sequences. Those hypotheses are then rescored using the full RAG-Sequence probability across the retrieved documents.

Basically:

> Generate possible answers using each document, then ask which full answer is best after considering the retriever's confidence too.

### RAG-Token Decoder

RAG-Token marginalizes over the retrieved documents at every generated token.

Because that happens token-by-token, it works more naturally with standard autoregressive seq2seq beam decoding.

Funny enough:

> RAG-Sequence is easier to understand, but RAG-Token has the cleaner token-by-token decoder.

## How the Experiment Was Carried Out

They first embedded and chunked Wikipedia.

In their setup, Wikipedia was split into roughly **100-word passages**, creating around **21 million retrievable passages**.

So when I say "document" in the RAG equations, I should really think of something closer to:

> one retrievable passage/chunk from a larger Wikipedia article.

They tested RAG across four main task types:

1. Open-domain Question Answering
2. Abstractive Question Answering
3. Jeopardy Question Generation
4. Fact Verification

## What We Are Reproducing

For our reproduction, I picked **Open-domain Question Answering**.

Why?

Because it's the easiest experiment to understand and it connects directly to real-world RAG use.

The whole idea is that the model should be able to answer questions better because the non-parametric retriever gives the generator relevant external knowledge instead of forcing the generator to rely entirely on what is stored in its weights.

For our mini version:

> query → retrieve relevant passages → use those passages as context → generate answer

Then we can compare:

- generator without retrieval
- generator with retrieval

That should give us a clean proof of concept.

## Results / Important Takeaways

The rest of the paper goes pretty deep into results, but the main point is that the architecture worked.

RAG achieved state-of-the-art results on several knowledge-intensive tasks at the time and showed that combining parametric and non-parametric memory could improve factual generation.

More importantly to me, the knowledge source can be updated without retraining the entire generator.

A few interesting results from the paper:

> "Performance peaks for RAG-Token at 10 retrieved documents."

> "Retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence."

> "We found that people prefer RAG's generation over purely parametric BART, finding RAG more factual and specific."

I mean the human evaluation makes that preference pretty pronounced lol.

Good paper. Very thorough.

The scary math also wasn't that scary once I actually read what every variable represented. The paper literally explains what each part of the probability equation is doing.

Big lesson from reading it:

> Don't stare at an ML equation like it's alien language. Figure out what each variable represents and follow the data through the system.

Time to implement.