"""Model construction helpers."""

from scratchgpt.config import ScratchGPTConfig
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.model.model_modern import ModernTransformerLanguageModel


def build_language_model(config: ScratchGPTConfig) -> TransformerLanguageModel:
    """Build the configured model implementation."""
    if config.architecture.model_variant == "classic":
        if config.architecture.position_encoding != "learned":
            raise ValueError("position_encoding='rope' requires model_variant='modern'")
        if config.architecture.normalization != "layernorm":
            raise ValueError("normalization='rmsnorm' requires model_variant='modern'")
        if config.architecture.ffn_variant != "mlp":
            raise ValueError("ffn_variant='swiglu' requires model_variant='modern'")
        if config.architecture.qk_norm:
            raise ValueError("qk_norm=True requires model_variant='modern'")
    if config.architecture.model_variant == "modern":
        return ModernTransformerLanguageModel(config)
    return TransformerLanguageModel(config)
