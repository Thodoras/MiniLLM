import torch

from data_preparation.gpt_dataloader_v1 import create_dataloader_v1

with open("source_data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
vocab_size = 50257
output_dim = 256
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs)
print(targets)

token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)

#Create positional embedding layer
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# Needed to get tensor from embedding.
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings