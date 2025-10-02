from typing import Dict, List
from data_preparation.token_utils import extract_tokens
        
def naive_token_extraction_from_path(path: str) -> Dict[str, int]:
    with open(path, 'r', encoding="utf-8") as file:
        raw_text = file.read()
    tokens = extract_tokens(raw_text)
    return _map_tokens_to_ids(tokens)

def _map_tokens_to_ids(tokens: List[str]) -> Dict[str, int]:
    all_words = sorted(set(tokens))
    vocab = {word: idx for idx, word in enumerate(all_words)}
    return vocab

tokens = naive_token_extraction_from_path("source-data/the-verdict.txt")
_map_tokens_to_ids(tokens)