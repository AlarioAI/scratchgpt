import pytest

from scratchgpt.config import ScratchGPTTraining


def test_training_defaults_include_step_based_fields() -> None:
    cfg = ScratchGPTTraining()
    # Backwards-compat: epoch-based still works
    assert cfg.max_epochs == 50
    # New step-based fields
    assert cfg.max_steps is None  # None means "use max_epochs"
    assert cfg.eval_every_steps == 500
    assert cfg.log_every_steps == 50
    assert cfg.warmup_steps == 0


def test_training_accepts_max_steps() -> None:
    cfg = ScratchGPTTraining(max_steps=5000, eval_every_steps=250, log_every_steps=25)
    assert cfg.max_steps == 5000
    assert cfg.eval_every_steps == 250


def test_training_rejects_zero_eval_every() -> None:
    with pytest.raises(ValueError):
        ScratchGPTTraining(eval_every_steps=0)
