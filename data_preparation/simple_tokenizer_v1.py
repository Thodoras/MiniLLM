import re
from typing import Dict, List

class SimpleTokenizerV1:
    def __init__(self, vocab) -> Dict[str, int]:
        self._str_to_int = vocab
        self._int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text: str) -> List[int]:
        preprocessed_tokens_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_tokens = [token.strip() for token in preprocessed_tokens_with_spaces if token.strip()]
        token_ids = [self._str_to_int[token] for token in preprocessed_tokens]
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        text = ' '.join([self._int_to_str[token_id] for token_id in token_ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text
        