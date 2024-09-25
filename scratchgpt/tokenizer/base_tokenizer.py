from abc import abstractmethod, ABC


class Tokenizer(ABC):

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string into a sequence of token IDs."""

    @abstractmethod
    def decode(self, encoding: list[int]) -> str:
        """Convert a sequence of token IDs back into a string."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""

    @property
    @abstractmethod
    def vocabulary(self) -> list[str]:
        """Return the learned vocabulary"""
