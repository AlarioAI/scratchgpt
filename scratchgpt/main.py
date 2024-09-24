import argparse
import io

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from tqdm import tqdm
from tqdm.utils import sys

torch.set_rng_state(torch.manual_seed(1337).get_state())

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
MAX_ITERS = 3000
BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-2
EVAL_INTERVAL = 300
N_EMBED = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--train_file",
                        help="The file you want to train on",
                        required=True,
                        type=argparse.FileType("r"))
    return parser.parse_args()


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self._lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, indices: Tensor, targets: Tensor|None = None) -> tuple[Tensor, Tensor]:
        tok_emb = self._token_embedding_table(indices) # B, T, C
        logits = self._lm_head(tok_emb) # (B, T, vocab_size)

        if targets is None:
            loss = torch.empty(0)
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idxs: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            logits, _loss = self(idxs)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idxs = torch.cat((idxs, idx_next), dim=1)
        return idxs


def load_dataset(path: io.TextIOWrapper) -> str:
    return path.read()


def get_vocab(text: str) -> list[str]:
    chars = sorted(list(set(text)))
    return chars


def str_to_int(chars: list[str]) -> dict[str, int]:
    return {char:idx for idx, char in enumerate(chars)}


def int_to_str(chars: list[str]) -> dict[int, str]:
    return {idx: char for idx, char in enumerate(chars)}


def encode(text: str, mapping: dict[str, int]) -> list[int]:
    return [mapping[char] for char in text]


def decode(encoding: list[int], mapping: dict[int, str]) -> str:
    return ''.join(mapping[v] for v in encoding)


def get_batch(block_size: int, batch_size: int, data: Tensor) -> tuple[Tensor, Tensor]:
    indices = torch.randint(len(data) - block_size, (batch_size,))
    batch = torch.stack([data[i:i+block_size] for i in indices])
    targets = torch.stack([data[i+1:i+block_size+1] for i in indices])
    return batch, targets


def main():
    args = parse_args()
    print(f"Using the device: {DEVICE}")

    text = load_dataset(args.train_file)

    chars = get_vocab(text)

    vocab_size = len(chars)
    print(f"{chars=}\n{vocab_size=}")

    encoding_mapping = str_to_int(chars)
    decoding_mapping = int_to_str(chars)

    data = torch.tensor(encode(text, encoding_mapping), dtype=torch.long).to(DEVICE)

    train_split: float = 0.9
    train_size = int(train_split * len(text))

    train_data = data[:train_size]
    val_data = data[train_size:]

    model = BigramLanguageModel(vocab_size, N_EMBED)

    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    loss = Tensor()
    pbar = tqdm(total=MAX_ITERS, desc="Training", file=sys.stdout)#, bar_format="{l_bar}{bar}| Loss: {loss:.4f}")

    for _ in range(MAX_ITERS):
        train_batch, train_targets = get_batch(BLOCK_SIZE, BATCH_SIZE, train_data)
        logits, loss = model(train_batch, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Training | Loss: {loss.item():.4f}")
        pbar.update(1)

    print(f"{loss.item()=}")

    context = torch.zeros((1,1), dtype=torch.long).to(DEVICE)
    generated = model.generate(context, max_new_tokens=500)
    first_batch_trained = decode(generated[0].tolist(), decoding_mapping)
    print(first_batch_trained)


if __name__ == "__main__":
    main()
