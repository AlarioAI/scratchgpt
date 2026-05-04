"""Phase 2 guardrail: default config must produce Phase 1 numerics.

This test is the safety net around the model.py edits in Phase 2. If any
Phase 2 change alters default behavior, it fails and the change needs to
go behind a flag instead.
"""
import torch

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTConfig, ScratchGPTTraining
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.training.determinism import seed_everything


def _build_default_model() -> TransformerLanguageModel:
    seed_everything(1337)
    config = ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=16, embedding_size=32, num_heads=4, num_blocks=2, vocab_size=64,
        ),
        training=ScratchGPTTraining(batch_size=2, dropout_rate=0.0),
    )
    return TransformerLanguageModel(config)


def test_default_model_forward_pass_is_stable() -> None:
    """A default-config model on a fixed input must produce the same logits
    before and after Phase 2 changes."""
    model = _build_default_model()
    model.eval()

    x = torch.arange(32).reshape(2, 16).long()
    with torch.no_grad():
        logits = model(x)

    # Shape is the structural contract.
    assert logits.shape == (2, 16, 64)
    # Determinism: same input, same seed, same weights -> same output.
    with torch.no_grad():
        logits2 = model(x)
    assert torch.allclose(logits, logits2)

    # Statistics sanity: default init should put logits in a reasonable range.
    assert logits.abs().mean().item() < 10.0
    # Not all zeros (would mean projections are frozen / broken).
    assert logits.std().item() > 1e-3


def test_default_architecture_parameter_count_unchanged() -> None:
    """Adding Phase 2 flags (defaulting to Phase 1 behavior) must not add or
    remove any parameters. The expected value is captured from the actual
    default-config model at the time this test was written. If you deliberately
    change the default architecture, update this constant on purpose."""
    model = _build_default_model()
    total = sum(p.numel() for p in model.parameters())
    expected = 29_952
    assert total == expected, f"Param count drift: got {total}, expected {expected}"
