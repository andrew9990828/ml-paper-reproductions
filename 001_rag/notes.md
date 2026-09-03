# I used to think RAG was just grabbing vectors of numbers, matching them to something, then taking a wild guess from a chunk.

**RAG is TWO things:**
    1. A Retriever 
    2. A Pretrained Generator (seq2seq NLP -> BART **In this case**)

# Parametric vs Non-Parametric:

**Parametric** -> Stores knowledge directly inside a models internal weights (This is the pretrained generator)

**Non-Parametric** -> Stores raw data in an external database

# Lewis et al. and the Teams Finding

**"Finally, we demonstrate that the non-parametric memory can be replaced to update the model's knowledge as the world changes."**