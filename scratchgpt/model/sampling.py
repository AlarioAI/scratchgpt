from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


def sample_next_token(logits: Tensor, context: Tensor, config: SamplingConfig) -> Tensor:
    logits = logits.clone()
    if config.repetition_penalty != 1.0:
        for batch_idx, seen_tokens in enumerate(context):
            for token_id in seen_tokens.unique():
                token_logit = logits[batch_idx, token_id]
                if token_logit < 0:
                    logits[batch_idx, token_id] = token_logit * config.repetition_penalty
                else:
                    logits[batch_idx, token_id] = token_logit / config.repetition_penalty

    logits = logits / config.temperature

    if config.top_k is not None:
        k = min(config.top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if config.top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        sorted_remove = sorted_probs.cumsum(dim=-1) > config.top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
