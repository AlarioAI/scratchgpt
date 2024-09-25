import argparse
import io
import math
import sys

import torch
from torch import Tensor, dropout, nn
from torch.nn import functional as F
from torch.optim.adamw import AdamW
from tqdm import tqdm
from ptflops import get_model_complexity_info

from scratchgpt.tokenizer.char_tokenizer import CharTokenizer


from .metering import AverageValueMeter


torch.manual_seed(1337)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BLOCK_SIZE = 32
MAX_ITERS = 5000
LEARNING_RATE = 3e-3
EVAL_INTERVAL = 500
N_EMBED = 48
NUM_HEADS = 6
NUM_BLOCKS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--train_file",
                        help="The file you want to train on",
                        required=True,
                        type=argparse.FileType("r"))
    return parser.parse_args()


def print_model_complexity(model: nn.Module):
    input_shape = (BLOCK_SIZE,)

    flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=True, as_strings=True)

    print(flops)
    print(params)


class Head(nn.Module):
    def __init__(self, embedding_size: int, block_size: int, head_size: int) -> None:
        super().__init__()

        self._key = nn.Linear(embedding_size, head_size, bias=False)
        self._query = nn.Linear(embedding_size, head_size, bias=False)
        self._value = nn.Linear(embedding_size, head_size, bias=False)
        self._dropout_factor = .4
        self._dropout = nn.Dropout(self._dropout_factor)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, context: Tensor) -> Tensor:
        B, T, C = context.shape
        key = self._key(context)
        query = self._query(context)

        normalization_term: float = 1.0 / math.sqrt(C)
        attention_scores = query @ key.transpose(-2, -1) * normalization_term
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_scores = self._dropout(attention_scores)

        value = self._value(context)

        out = attention_scores @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, block_size: int, head_size: int) -> None:
        super().__init__()
        self._dropout_factor = .4
        self._heads = nn.ModuleList(Head(embedding_size, block_size, head_size) for _ in range(num_heads))
        self._proj = nn.Linear(embedding_size, embedding_size)
        self._dropout = nn.Dropout(self._dropout_factor)

    def forward(self, context: Tensor) -> Tensor:
        out = torch.cat([head(context) for head in self._heads], dim=-1)
        out = self._proj(out)
        out = self._dropout(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self._ffwd_multipler = 4
        self._dropout = .4

        self._net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * self._ffwd_multipler),
            nn.ReLU(),
            nn.Linear(self._ffwd_multipler * embedding_size, embedding_size),
            nn.Dropout(self._dropout)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self._net(tensor)


class Block(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int, block_size: int,) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self._self_attn_heads = MultiHeadAttention(num_heads, embedding_size, block_size, head_size)
        self._ffwd = FeedFoward(embedding_size)
        self._layer_norm_attention = nn.LayerNorm(embedding_size)
        self._layer_norm_ffwd = nn.LayerNorm(embedding_size)

    def forward(self, tensor: Tensor) -> Tensor:
        normal_tensor = self._layer_norm_attention(tensor)

        tensor = tensor + self._self_attn_heads(normal_tensor)

        normal_tensor = self._layer_norm_ffwd(tensor)
        tensor = tensor + self._ffwd(normal_tensor)
        return tensor


class BigramLanguageModel(nn.Module):

    def __init__(self, num_heads: int, vocab_size: int, embedding_size: int, block_size: int, num_blocks: int,) -> None:
        super().__init__()
        self._block_size = block_size
        self._token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self._position_embedding_table = nn.Embedding(block_size, embedding_size)
        self._blocks = nn.Sequential(*[Block(num_heads, embedding_size, block_size) for _ in range(num_blocks)])
        self._block_norm = nn.LayerNorm(embedding_size)
        self._lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, context: Tensor, targets: Tensor|None = None) -> tuple[Tensor, Tensor]:
        B, T = context.shape

        tok_emb = self._token_embedding_table(context) # B, T, C
        pos_emb = self._position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
        x = tok_emb + pos_emb # B, T, C
        x = self._blocks(x)
        x = self._block_norm(x)
        logits = self._lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = torch.empty(0)
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            cropped_context = context[:, -self._block_size:]
            logits, _loss = self(cropped_context)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
        return context


def load_dataset(path: io.TextIOWrapper) -> str:
    return path.read()


def get_batch(block_size: int, batch_size: int, data: Tensor) -> tuple[Tensor, Tensor]:
    indices = torch.randint(len(data) - block_size, (batch_size,))
    batch = torch.stack([data[i:i+block_size] for i in indices])
    targets = torch.stack([data[i+1:i+block_size+1] for i in indices])
    return batch, targets


def main():
    args = parse_args()
    print(f"Using the device: {DEVICE}")

    text = load_dataset(args.train_file)

    tokenizer = CharTokenizer(text)

    print(f"{tokenizer.vocabulary=}\n{tokenizer.vocab_size=}")

    tokenized_data = tokenizer.encode(text)
    data = torch.tensor(tokenized_data, dtype=torch.long).to(DEVICE)

    train_split: float = 0.9
    train_size = int(train_split * len(text))

    train_data = data[:train_size]
    val_data = data[train_size:]

    model = BigramLanguageModel(NUM_HEADS, tokenizer.vocab_size, N_EMBED, BLOCK_SIZE, NUM_BLOCKS)

    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    loss = Tensor()
    pbar = tqdm(total=MAX_ITERS, desc="Training", file=sys.stdout)#, bar_format="{l_bar}{bar}| Loss: {loss:.4f}")

    average_loss = AverageValueMeter()
    val_average_loss = AverageValueMeter()
    for step in range(MAX_ITERS):
        model.train()
        train_batch, train_targets = get_batch(BLOCK_SIZE, BATCH_SIZE, train_data)
        logits, loss = model(train_batch, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        average_loss.add(loss.item())
        mean, std = average_loss.value()

        model.eval()
        with torch.no_grad():
            val_batch, val_targets = get_batch(BLOCK_SIZE, BATCH_SIZE, val_data)
            _, val_loss = model(val_batch, val_targets)
            val_average_loss.add(val_loss.item())

        val_mean, val_std = val_average_loss.value()
        pbar.set_description(f"Training | Loss: {mean:.4f} {std:.4f}, Validation | Loss: {val_mean:.4f} {val_std:.4f}")
        pbar.update(1)

        if step % EVAL_INTERVAL == 0:
            average_loss.reset()
            val_average_loss.reset()

    context = torch.zeros((1,1), dtype=torch.long).to(DEVICE)
    generated = model.generate(context, max_new_tokens=500)
    first_batch_trained = tokenizer.decode(generated[0].tolist())
    print(first_batch_trained)


if __name__ == "__main__":
    main()
