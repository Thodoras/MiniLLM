# MiniLLM

A light LLM model based on S. Raschka's book Build A Large Language Model.

## Steps - Based on schema.png

### 1. Data preparation and sampling.

- Preparing text for model training
- Spliting text into word and subword tokens.his can be in a naive manner.  Alternatively apply advanced techniques such as byte pairing with the use of a third party library, namely tiktoken.
- Sampling training examples with sliding window approach.
- Convert tokens into vectors that feed the model.

### Schema:
Text -> Tokens -> TokenIds -> (Initially random) Embeddings + Positional Embeddings -> Input Embeddings