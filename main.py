from data_preparation.vocab_extractor import naive_token_extraction_from_path
from data_preparation.simple_tokenizer_v1 import SimpleTokenizerV1

vocab = naive_token_extraction_from_path("source-data/the-verdict.txt")
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know, " Mrs. Gisburn said with pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))