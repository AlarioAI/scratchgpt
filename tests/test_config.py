import pytest

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTTraining


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


def test_architecture_defaults_reflect_phase2_decisions() -> None:
    arch = ScratchGPTArchitecture(vocab_size=256)
    # Phase 2 Exp-1 flipped attention_scale_mode default to "head" (textbook
    # 1/sqrt(head_size)). The "embedding" variant remains available for
    # reproducing Phase 1 baselines. See memory/experiments/2026-05-04-exp1-*.md.
    assert arch.attention_scale_mode == "head"
    # Remaining flags: defaults still preserve Phase 1 numerics pending their
    # own ablation experiments.
    assert arch.ffn_activation == "relu"
    assert arch.tie_weights is False
    assert arch.init_scheme == "default"
    assert arch.use_bias is True


def test_architecture_accepts_phase2_improvements() -> None:
    arch = ScratchGPTArchitecture(
        vocab_size=256,
        attention_scale_mode="head",
        ffn_activation="gelu",
        tie_weights=True,
        init_scheme="gpt2",
        use_bias=False,
    )
    assert arch.attention_scale_mode == "head"
    assert arch.ffn_activation == "gelu"
    assert arch.tie_weights is True
    assert arch.init_scheme == "gpt2"
    assert arch.use_bias is False


def test_architecture_rejects_invalid_attention_scale_mode() -> None:
    with pytest.raises(ValueError):
        ScratchGPTArchitecture(vocab_size=256, attention_scale_mode="not-a-mode")


def test_architecture_rejects_invalid_activation() -> None:
    with pytest.raises(ValueError):
        ScratchGPTArchitecture(vocab_size=256, ffn_activation="swiglu")  # Phase 4, not here
