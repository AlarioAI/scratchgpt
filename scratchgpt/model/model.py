import math
from collections.abc import Callable
from typing import Any

import torch
from ptflops import get_model_complexity_info
from torch import Tensor, nn
from torch.nn import functional as F

from scratchgpt.config import ScratchGPTConfig
from scratchgpt.model.sampling import SamplingConfig, sample_next_token


class Head(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        block_size: int,
        head_size: int,
        dropout_rate: float,
        attention_scale: float,
    ) -> None:
        super().__init__()

        self._key = nn.Linear(embedding_size, head_size, bias=False)
        self._query = nn.Linear(embedding_size, head_size, bias=False)
        self._value = nn.Linear(embedding_size, head_size, bias=False)
        self._dropout = nn.Dropout(dropout_rate)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self._attention_scale = attention_scale

    def forward(self, context: Tensor) -> Tensor:
        B, T, _ = context.shape
        key = self._key(context)
        query = self._query(context)

        normalization_term: float = self._attention_scale
        attention_scores = query @ key.transpose(-2, -1) * normalization_term
        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,  # type: ignore
            float("-inf"),
        )
        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_scores = self._dropout(attention_scores)

        value = self._value(context)

        out: Tensor = attention_scores @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        block_size: int,
        head_size: int,
        dropout_rate: float,
        attention_scale: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self._heads = nn.ModuleList(
            Head(embedding_size, block_size, head_size, dropout_rate, attention_scale) for _ in range(num_heads)
        )
        self._proj = nn.Linear(embedding_size, embedding_size, bias=use_bias)
        self._proj._is_residual_projection = True  # type: ignore[assignment]
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, context: Tensor) -> Tensor:
        out: Tensor = torch.cat([head(context) for head in self._heads], dim=-1)
        out = self._proj(out)
        out = self._dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        dropout_rate: float,
        activation: nn.Module,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self._ffwd_multipler = 4

        self._net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * self._ffwd_multipler, bias=use_bias),
            activation,
            nn.Linear(self._ffwd_multipler * embedding_size, embedding_size, bias=use_bias),
            nn.Dropout(dropout_rate),
        )
        self._net[2]._is_residual_projection = True  # type: ignore[assignment]

    def forward(self, tensor: Tensor) -> Tensor:
        out: Tensor = self._net(tensor)
        return out


class Block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        block_size: int,
        dropout_rate: float,
        attention_scale: float,
        activation: nn.Module,
        use_bias: bool,
    ) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self._self_attn_heads = MultiHeadAttention(
            num_heads,
            embedding_size,
            block_size,
            head_size,
            dropout_rate,
            attention_scale=attention_scale,
            use_bias=use_bias,
        )
        self._ffwd = FeedForward(embedding_size, dropout_rate, activation, use_bias)
        self._layer_norm_attention = nn.LayerNorm(embedding_size, bias=use_bias)
        self._layer_norm_ffwd = nn.LayerNorm(embedding_size, bias=use_bias)

    def forward(self, tensor: Tensor) -> Tensor:
        normal_tensor = self._layer_norm_attention(tensor)

        tensor = tensor + self._self_attn_heads(normal_tensor)

        normal_tensor = self._layer_norm_ffwd(tensor)
        tensor = tensor + self._ffwd(normal_tensor)
        return tensor


def _init_default(model: nn.Module, num_blocks: int) -> None:
    """No-op: leaves torch's default initialization intact."""


def _init_gpt2(model: nn.Module, num_blocks: int) -> None:
    """GPT-2 init: N(0, 0.02) on Linear/Embedding weights, zero bias on Linear,
    then scale residual projections (tagged _is_residual_projection) by 1/sqrt(2*num_blocks)."""

    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_module)
    scale = 1.0 / math.sqrt(2 * num_blocks)
    for module in model.modules():
        if isinstance(module, nn.Linear) and getattr(module, "_is_residual_projection", False):
            module.weight.data.mul_(scale)


INIT_SCHEMES: dict[str, Callable[[nn.Module, int], None]] = {
    "default": _init_default,
    "gpt2": _init_gpt2,
}


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        config: ScratchGPTConfig,
    ) -> None:
        super().__init__()
        arch = config.architecture
        training = config.training
        assert arch.vocab_size is not None, "Must supply vocabulary size"

        self._block_size = arch.block_size
        self._token_embedding_table = nn.Embedding(arch.vocab_size, arch.embedding_size)
        self._position_embedding_table = nn.Embedding(
            arch.block_size,
            arch.embedding_size,
        )

        head_size = arch.embedding_size // arch.num_heads
        attention_scale = arch.attention_scale_for(head_size)

        self._blocks = nn.Sequential(
            *[
                Block(
                    num_heads=arch.num_heads,
                    embedding_size=arch.embedding_size,
                    block_size=arch.block_size,
                    dropout_rate=training.dropout_rate,
                    attention_scale=attention_scale,
                    activation=arch.make_activation(),
                    use_bias=arch.use_bias,
                )
                for _ in range(arch.num_blocks)
            ]
        )
        self._block_norm = nn.LayerNorm(arch.embedding_size, bias=arch.use_bias)
        self._lm_head = nn.Linear(arch.embedding_size, arch.vocab_size, bias=arch.use_bias)

        # Post-build transforms.
        if arch.tie_weights:
            self._lm_head.weight = self._token_embedding_table.weight
        INIT_SCHEMES[arch.init_scheme](self, arch.num_blocks)

    def forward(self, context: Tensor) -> Tensor:
        context = context.long()
        B, T = context.shape

        tok_emb = self._token_embedding_table(context)  # B, T, C
        pos_emb = self._position_embedding_table(torch.arange(T, device=context.device))  # (T, C)
        x = tok_emb + pos_emb  # B, T, C
        x = self._blocks(x)
        x = self._block_norm(x)
        logits: Tensor = self._lm_head(x)  # (B, T, vocab_size)
        return logits

    def generate(
        self,
        context: Tensor,
        max_new_tokens: int,
        stop_token: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        sampling = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        for _ in range(max_new_tokens):
            cropped_context = context[:, -self._block_size :]
            logits = self(cropped_context)[:, -1, :]
            idx_next = sample_next_token(logits, context, sampling)
            context = torch.cat((context, idx_next), dim=1)

            if stop_token is not None and idx_next == stop_token:
                return context

        return context


def print_model_complexity(model: TransformerLanguageModel, config: ScratchGPTConfig, device: torch.device) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== MODEL COMPLEXITY ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    input_shape = (config.architecture.block_size,)

    def input_constructor(input_shape: Any) -> Tensor:
        return torch.randint(
            0,
            model._token_embedding_table.num_embeddings,
            (1,) + input_shape,
            device=device,
        )

    flops, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_constructor,
        print_per_layer_stat=False,
        as_strings=False,
    )

    print(f" FLOPs per forward pass: {flops:,}")
    print(f" Params: {params}")

    print("=========================")
