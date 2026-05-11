import pytest
import torch

from scratchgpt.model.sampling import SamplingConfig, sample_next_token


def test_top_k_keeps_only_the_highest_k_logits() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    context = torch.tensor([[0, 1]])

    next_token = sample_next_token(logits, context, SamplingConfig(top_k=1))

    assert next_token.item() == 3


def test_top_p_keeps_the_smallest_high_probability_prefix() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    context = torch.tensor([[0, 1]])

    next_token = sample_next_token(logits, context, SamplingConfig(top_p=0.6))

    assert next_token.item() == 0


def test_repetition_penalty_lowers_seen_token_logits_before_sampling() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    context = torch.tensor([[0, 0]])

    next_token = sample_next_token(logits, context, SamplingConfig(top_k=1, repetition_penalty=3.0))

    assert next_token.item() == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
    ],
)
def test_sampling_config_rejects_invalid_values(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingConfig(**kwargs)
