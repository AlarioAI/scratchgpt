"""Parallel research model path for modern dense-decoder ablations."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scratchgpt.config import ScratchGPTConfig
from scratchgpt.model.model import INIT_SCHEMES, FeedForward, TransformerLanguageModel


class SwiGLUFeedForward(nn.Module):
    def __init__(self, embedding_size: int, dropout_rate: float, use_bias: bool) -> None:
        super().__init__()
        hidden_size = round((8 * embedding_size) / 3)
        self._gate = nn.Linear(embedding_size, hidden_size, bias=use_bias)
        self._value = nn.Linear(embedding_size, hidden_size, bias=use_bias)
        self._proj = nn.Linear(hidden_size, embedding_size, bias=use_bias)
        self._proj._is_residual_projection = True  # type: ignore[assignment]
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, tensor: Tensor) -> Tensor:
        out: Tensor = F.silu(self._gate(tensor)) * self._value(tensor)
        out = self._proj(out)
        out = self._dropout(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, embedding_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.ones(embedding_size))
        self._eps = eps

    def forward(self, tensor: Tensor) -> Tensor:
        variance = tensor.pow(2).mean(dim=-1, keepdim=True)
        out: Tensor = tensor * torch.rsqrt(variance + self._eps)
        return out * self._weight


def _make_norm(embedding_size: int, use_bias: bool, normalization: str) -> nn.Module:
    if normalization == "rmsnorm":
        return RMSNorm(embedding_size)
    return nn.LayerNorm(embedding_size, bias=use_bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer("_inv_freq", inv_freq, persistent=False)

    def forward(self, tensor: Tensor) -> Tensor:
        _, T, head_size = tensor.shape
        inv_freq = self._inv_freq
        assert isinstance(inv_freq, Tensor)
        positions = torch.arange(T, device=tensor.device, dtype=inv_freq.dtype)
        freqs = torch.outer(positions, inv_freq.to(tensor.device))
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(tensor.dtype).view(1, T, head_size)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(tensor.dtype).view(1, T, head_size)
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)
        return (tensor * cos) + (rotated * sin)


class ModernHead(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        block_size: int,
        head_size: int,
        dropout_rate: float,
        attention_scale: float,
        use_rope: bool,
        use_qk_norm: bool,
    ) -> None:
        super().__init__()

        self._key = nn.Linear(embedding_size, head_size, bias=False)
        self._query = nn.Linear(embedding_size, head_size, bias=False)
        self._value = nn.Linear(embedding_size, head_size, bias=False)
        self._dropout = nn.Dropout(dropout_rate)
        self._key_norm = RMSNorm(head_size) if use_qk_norm else None
        self._query_norm = RMSNorm(head_size) if use_qk_norm else None
        self._rotary = RotaryEmbedding(head_size) if use_rope else None
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self._attention_scale = attention_scale

    def forward(self, context: Tensor) -> Tensor:
        _, T, _ = context.shape
        key = self._key(context)
        query = self._query(context)
        if self._key_norm is not None and self._query_norm is not None:
            key = self._key_norm(key)
            query = self._query_norm(query)
        if self._rotary is not None:
            key = self._rotary(key)
            query = self._rotary(query)

        attention_scores = query @ key.transpose(-2, -1) * self._attention_scale
        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,  # type: ignore
            float("-inf"),
        )
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self._dropout(attention_scores)

        value = self._value(context)
        out: Tensor = attention_scores @ value
        return out


class ModernMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        block_size: int,
        head_size: int,
        dropout_rate: float,
        attention_scale: float,
        use_bias: bool,
        use_rope: bool,
        use_qk_norm: bool,
    ) -> None:
        super().__init__()
        self._heads = nn.ModuleList(
            ModernHead(
                embedding_size,
                block_size,
                head_size,
                dropout_rate,
                attention_scale,
                use_rope,
                use_qk_norm,
            )
            for _ in range(num_heads)
        )
        self._proj = nn.Linear(embedding_size, embedding_size, bias=use_bias)
        self._proj._is_residual_projection = True  # type: ignore[assignment]
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, context: Tensor) -> Tensor:
        out: Tensor = torch.cat([head(context) for head in self._heads], dim=-1)
        out = self._proj(out)
        out = self._dropout(out)
        return out


class ModernBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        block_size: int,
        dropout_rate: float,
        attention_scale: float,
        activation: nn.Module,
        use_bias: bool,
        normalization: str,
        ffn_variant: str,
        use_rope: bool,
        use_qk_norm: bool,
    ) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self._self_attn_heads = ModernMultiHeadAttention(
            num_heads,
            embedding_size,
            block_size,
            head_size,
            dropout_rate,
            attention_scale=attention_scale,
            use_bias=use_bias,
            use_rope=use_rope,
            use_qk_norm=use_qk_norm,
        )
        self._ffwd: FeedForward | SwiGLUFeedForward
        if ffn_variant == "swiglu":
            self._ffwd = SwiGLUFeedForward(embedding_size, dropout_rate, use_bias)
        else:
            self._ffwd = FeedForward(embedding_size, dropout_rate, activation, use_bias)
        self._layer_norm_attention = _make_norm(embedding_size, use_bias, normalization)
        self._layer_norm_ffwd = _make_norm(embedding_size, use_bias, normalization)

    def forward(self, tensor: Tensor) -> Tensor:
        normal_tensor = self._layer_norm_attention(tensor)
        tensor = tensor + self._self_attn_heads(normal_tensor)

        normal_tensor = self._layer_norm_ffwd(tensor)
        tensor = tensor + self._ffwd(normal_tensor)
        return tensor


class ModernTransformerLanguageModel(TransformerLanguageModel):
    """Phase 4 research path.

    The default learned-position LayerNorm path delegates to the classic
    implementation for exact parity. Non-default Phase 4 flags run through this
    parallel implementation.
    """

    def __init__(self, config: ScratchGPTConfig) -> None:
        arch = config.architecture
        if (
            arch.position_encoding == "learned"
            and arch.normalization == "layernorm"
            and arch.ffn_variant == "mlp"
            and not arch.qk_norm
        ):
            super().__init__(config)
            self._uses_classic_delegate = True
            self._position_encoding = "learned"
            return

        nn.Module.__init__(self)
        training = config.training
        assert arch.vocab_size is not None, "Must supply vocabulary size"

        self._uses_classic_delegate = False
        self._position_encoding = arch.position_encoding
        self._block_size = arch.block_size
        self._token_embedding_table = nn.Embedding(arch.vocab_size, arch.embedding_size)
        if arch.position_encoding == "learned":
            self._position_embedding_table = nn.Embedding(arch.block_size, arch.embedding_size)

        head_size = arch.embedding_size // arch.num_heads
        attention_scale = arch.attention_scale_for(head_size)
        use_rope = arch.position_encoding == "rope"
        use_qk_norm = arch.qk_norm
        self._blocks = nn.Sequential(
            *[
                ModernBlock(
                    num_heads=arch.num_heads,
                    embedding_size=arch.embedding_size,
                    block_size=arch.block_size,
                    dropout_rate=training.dropout_rate,
                    attention_scale=attention_scale,
                    activation=arch.make_activation(),
                    use_bias=arch.use_bias,
                    normalization=arch.normalization,
                    ffn_variant=arch.ffn_variant,
                    use_rope=use_rope,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(arch.num_blocks)
            ]
        )
        self._modern_block_norm = _make_norm(arch.embedding_size, arch.use_bias, arch.normalization)
        self._lm_head = nn.Linear(arch.embedding_size, arch.vocab_size, bias=arch.use_bias)

        if arch.tie_weights:
            self._lm_head.weight = self._token_embedding_table.weight
        INIT_SCHEMES[arch.init_scheme](self, arch.num_blocks)

    def forward(self, context: Tensor) -> Tensor:
        if self._uses_classic_delegate:
            return super().forward(context)

        context = context.long()
        _, T = context.shape
        x = self._token_embedding_table(context)
        if self._position_encoding == "learned":
            x = x + self._position_embedding_table(torch.arange(T, device=context.device))
        x = self._blocks(x)
        x = self._modern_block_norm(x)
        logits: Tensor = self._lm_head(x)
        return logits
