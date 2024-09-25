from typing import override
from .base_tokenizer import Tokenizer


def get_vocab(text: str) -> list[str]:
    chars = sorted(list(set(text)))
    return chars


def str_to_int(chars: list[str]) -> dict[str, int]:
    return {char:idx for idx, char in enumerate(chars)}


def int_to_str(chars: list[str]) -> dict[int, str]:
    return {idx: char for idx, char in enumerate(chars)}


class CharTokenizer(Tokenizer):

    def __init__(self, text: str) -> None:
        self._vocabulary = get_vocab(text)
        self._encoding_mapping = str_to_int(self._vocabulary)
        self._decoding_mapping = int_to_str(self._vocabulary)

    @property
    @override
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    @override
    def vocabulary(self) -> list[str]:
        return self._vocabulary

    @override
    def encode(self, text: str) -> list[int]:
        return [self._encoding_mapping[char] for char in text]

    @override
    def decode(self, encoding: list[int],) -> str:
        return ''.join(self._decoding_mapping[v] for v in encoding)
