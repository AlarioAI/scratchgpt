import torch
from torch.utils.data import Dataset
from typing import Literal

from .tokenizer.base_tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, 
                 text: str, 
                 tokenizer: Tokenizer, 
                 block_size: int,
                 split: Literal['train', 'validation', 'test'],
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
        ):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        total_size = len(self.data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        if split == 'train':
            self.data = self.data[:train_size]
        elif split == 'validation':
            self.data = self.data[train_size:train_size+val_size]
        elif split == 'test':
            self.data = self.data[train_size+val_size:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'.")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block = self.data[idx:idx + self.block_size]
        target = self.data[idx + 1:idx + self.block_size + 1]
        return block, target
