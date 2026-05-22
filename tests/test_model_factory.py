from typing import Literal

import pytest
import torch

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTConfig, ScratchGPTTraining
from scratchgpt.model.factory import build_language_model
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.model.model_modern import ModernTransformerLanguageModel
from scratchgpt.training.determinism import seed_everything


def _config(model_variant: Literal["classic", "modern"]) -> ScratchGPTConfig:
    return ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=8,
            embedding_size=16,
            num_heads=4,
            num_blocks=1,
            vocab_size=32,
            model_variant=model_variant,
        ),
        training=ScratchGPTTraining(batch_size=2, dropout_rate=0.0),
    )


def _rope_config(model_variant: Literal["classic", "modern"] = "modern") -> ScratchGPTConfig:
    return ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=8,
            embedding_size=16,
            num_heads=4,
            num_blocks=1,
            vocab_size=32,
            model_variant=model_variant,
            position_encoding="rope",
        ),
        training=ScratchGPTTraining(batch_size=2, dropout_rate=0.0),
    )


def _rmsnorm_config(model_variant: Literal["classic", "modern"] = "modern") -> ScratchGPTConfig:
    return ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=8,
            embedding_size=16,
            num_heads=4,
            num_blocks=1,
            vocab_size=32,
            model_variant=model_variant,
            normalization="rmsnorm",
        ),
        training=ScratchGPTTraining(batch_size=2, dropout_rate=0.0),
    )


def _swiglu_config(model_variant: Literal["classic", "modern"] = "modern") -> ScratchGPTConfig:
    return ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=8,
            embedding_size=16,
            num_heads=4,
            num_blocks=1,
            vocab_size=32,
            model_variant=model_variant,
            ffn_variant="swiglu",
        ),
        training=ScratchGPTTraining(batch_size=2, dropout_rate=0.0),
    )


def _qk_norm_config(model_variant: Literal["classic", "modern"] = "modern") -> ScratchGPTConfig:
    return ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=8,
            embedding_size=16,
            num_heads=4,
            num_blocks=1,
            vocab_size=32,
            model_variant=model_variant,
            qk_norm=True,
        ),
        training=ScratchGPTTraining(batch_size=2, dropout_rate=0.0),
    )


def test_factory_builds_classic_model_by_default() -> None:
    model = build_language_model(_config("classic"))

    assert isinstance(model, TransformerLanguageModel)
    assert not isinstance(model, ModernTransformerLanguageModel)


def test_factory_builds_modern_model_variant() -> None:
    model = build_language_model(_config("modern"))

    assert isinstance(model, ModernTransformerLanguageModel)


def test_modern_path_matches_classic_logits_before_ablation_changes() -> None:
    seed_everything(1337)
    classic = build_language_model(_config("classic"))
    seed_everything(1337)
    modern = build_language_model(_config("modern"))

    classic.eval()
    modern.eval()
    x = torch.arange(16).reshape(2, 8).long()

    with torch.no_grad():
        classic_logits = classic(x)
        modern_logits = modern(x)

    assert torch.equal(classic_logits, modern_logits)
    assert sum(p.numel() for p in classic.parameters()) == sum(p.numel() for p in modern.parameters())


def test_rope_requires_modern_model_variant() -> None:
    with pytest.raises(ValueError, match="requires model_variant='modern'"):
        build_language_model(_rope_config("classic"))


def test_rmsnorm_requires_modern_model_variant() -> None:
    with pytest.raises(ValueError, match="requires model_variant='modern'"):
        build_language_model(_rmsnorm_config("classic"))


def test_swiglu_requires_modern_model_variant() -> None:
    with pytest.raises(ValueError, match="requires model_variant='modern'"):
        build_language_model(_swiglu_config("classic"))


def test_qk_norm_requires_modern_model_variant() -> None:
    with pytest.raises(ValueError, match="requires model_variant='modern'"):
        build_language_model(_qk_norm_config("classic"))


def test_modern_rope_forward_shape_and_parameter_delta() -> None:
    seed_everything(1337)
    learned = build_language_model(_config("modern"))
    seed_everything(1337)
    rope = build_language_model(_rope_config())

    x = torch.arange(16).reshape(2, 8).long()
    with torch.no_grad():
        logits = rope(x)

    learned_params = sum(p.numel() for p in learned.parameters())
    rope_params = sum(p.numel() for p in rope.parameters())
    assert logits.shape == (2, 8, 32)
    assert learned_params - rope_params == 8 * 16


def test_modern_rmsnorm_forward_shape_and_parameter_delta() -> None:
    seed_everything(1337)
    learned = build_language_model(_config("modern"))
    seed_everything(1337)
    rmsnorm = build_language_model(_rmsnorm_config())

    x = torch.arange(16).reshape(2, 8).long()
    with torch.no_grad():
        logits = rmsnorm(x)

    learned_params = sum(p.numel() for p in learned.parameters())
    rmsnorm_params = sum(p.numel() for p in rmsnorm.parameters())
    assert logits.shape == (2, 8, 32)
    assert learned_params - rmsnorm_params == 3 * 16


def test_modern_swiglu_forward_shape_and_parameter_delta() -> None:
    seed_everything(1337)
    learned = build_language_model(_config("modern"))
    seed_everything(1337)
    swiglu = build_language_model(_swiglu_config())

    x = torch.arange(16).reshape(2, 8).long()
    with torch.no_grad():
        logits = swiglu(x)

    learned_params = sum(p.numel() for p in learned.parameters())
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    assert logits.shape == (2, 8, 32)
    assert swiglu_params - learned_params == 38


def test_modern_qk_norm_forward_shape_and_parameter_delta() -> None:
    seed_everything(1337)
    learned = build_language_model(_config("modern"))
    seed_everything(1337)
    qk_norm = build_language_model(_qk_norm_config())

    x = torch.arange(16).reshape(2, 8).long()
    with torch.no_grad():
        logits = qk_norm(x)

    learned_params = sum(p.numel() for p in learned.parameters())
    qk_norm_params = sum(p.numel() for p in qk_norm.parameters())
    assert logits.shape == (2, 8, 32)
    assert qk_norm_params - learned_params == 2 * 4 * 4
