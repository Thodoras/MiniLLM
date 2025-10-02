import re
from typing import List


def extract_tokens(text: str) -> List[str]:
    preprocessed_tokens_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed_tokens = [token.strip() for token in preprocessed_tokens_with_spaces if token.strip()]
    return preprocessed_tokens