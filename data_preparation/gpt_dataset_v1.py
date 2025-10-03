from typing import Tuple
import torch
from torch.utils.data import Dataset
from tiktoken import Encoding

class GPTDatasetV1(Dataset):
    
    def __init__(self, text: str, tokenizer: Encoding, max_length: int, stride: int):
        self._input_ids = []
        self._target_ids = []
        token_ids = tokenizer.encode(text)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_vec = token_ids[i: i + max_length]
            target_vec = token_ids[i+1: i + max_length + 1]
            self._input_ids.append(torch.tensor(input_vec))
            self._target_ids.append(torch.tensor(target_vec))
            
    def __len__(self) -> int:
        return len(self._input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._input_ids[idx], self._target_ids[idx]