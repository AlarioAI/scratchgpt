import math
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from torch import nn


def ensure_split_is_valid(v: tuple[float, float]) -> tuple[float, float]:
    """
    Validates the data split contains only 2 values and they add to 1.0
    """
    splits_sum = sum(v)
    is_valid_split = math.isclose(splits_sum, 1.0)
    if not is_valid_split:
        raise ValueError("Invalid data 'split'")

    val_split = v[1]
    if val_split == 0.0:
        raise ValueError("You can't have 0 sized validation split.")
    return v


SplitType = Annotated[tuple[float, float], AfterValidator(ensure_split_is_valid)]


# Module-level dispatch table. Adding a new activation is one entry here.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class ScratchGPTArchitecture(BaseSettings):
    """
    All settings for training the model.
    """

    block_size: int = 256
    embedding_size: int = 384
    """ Size of the individual embeddings vector """
    num_heads: int = 6
    num_blocks: int = 6
    vocab_size: int | None = None

    model_variant: Literal["classic", "modern"] = "classic"
    """
    Architecture implementation path. 'classic' keeps the pedagogical
    scratchgpt/model/model.py implementation. 'modern' is the parallel research
    path for Phase 4+ ablations.
    """

    position_encoding: Literal["learned", "rope"] = "learned"
    """
    Position encoding strategy. 'learned' is the classic GPT-style learned
    embedding table. 'rope' is available on the modern model path.
    """

    normalization: Literal["layernorm", "rmsnorm"] = "layernorm"
    """
    Residual-stream normalization. 'layernorm' preserves the pedagogical
    baseline. 'rmsnorm' is available on the modern model path.
    """

    ffn_variant: Literal["mlp", "swiglu"] = "mlp"
    """
    Feed-forward block type. 'mlp' is the classic two-layer FFN. 'swiglu' is
    available on the modern model path.
    """

    qk_norm: bool = False
    """Apply per-head RMSNorm to query and key vectors on the modern model path."""

    # --- Phase 2 flags. Defaults preserve Phase 1 baseline numerics. ---
    attention_scale_mode: Literal["embedding", "head"] = "head"
    """
    Attention softmax temperature. 'head' uses the textbook 1/sqrt(head_size);
    'embedding' uses the Phase 1 bug 1/sqrt(embedding_size) and is kept opt-in
    for reproducing pre-Phase-2 baselines.
    """
    ffn_activation: Literal["relu", "gelu"] = "gelu"
    """
    FFN nonlinearity. 'gelu' is the modern default (GPT-2+); 'relu' is kept
    opt-in for reproducing pre-Phase-2 baselines.
    """
    tie_weights: bool = False
    """Share the token embedding matrix with the lm_head (weight tying)."""
    init_scheme: Literal["default", "gpt2"] = "gpt2"
    """
    Parameter initialization. 'gpt2' applies N(0, 0.02) init plus
    1/sqrt(2*num_blocks) scaling on residual projections. 'default' falls
    back to PyTorch's built-in init (kept opt-in for reproducing pre-Phase-2
    baselines).
    """
    use_bias: bool = True
    """If False, removes bias from Linear and LayerNorm modules in the transformer blocks."""
    # --- end Phase 2 flags ---

    @model_validator(mode="after")
    def validate_embedding_and_heads(self) -> Self:
        """
        Ensures that the embedding_size is perfectly divisible by the number of attention heads.
        """
        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                f"Incompatible model architecture: embedding_size ({self.embedding_size}) "
                f"must be divisible by num_heads ({self.num_heads})."
            )
        head_size = self.embedding_size // self.num_heads
        if self.position_encoding == "rope" and head_size % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension")
        return self

    def make_activation(self) -> nn.Module:
        """Instantiate the FFN activation module selected by ffn_activation."""
        return _ACTIVATIONS[self.ffn_activation]()

    def attention_scale_for(self, head_size: int) -> float:
        """Compute the per-head attention scale scalar based on attention_scale_mode.

        'head' (textbook): 1/sqrt(head_size). 'embedding' (Phase 1 bug, opt-in
        for reproducibility): 1/sqrt(embedding_size).
        """
        if self.attention_scale_mode == "head":
            return 1.0 / math.sqrt(head_size)
        return 1.0 / math.sqrt(self.embedding_size)

    model_config = SettingsConfigDict(
        env_prefix="ARCHITECTURE_",
        extra="allow",
    )


class ScratchGPTTraining(BaseSettings):
    """
    All training related parameters
    """

    max_epochs: int = 50
    max_steps: int | None = None
    """If set, overrides max_epochs. Training stops after this many optimizer steps."""
    eval_every_steps: int = Field(default=500, gt=0)
    log_every_steps: int = Field(default=50, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    """Reserved for Phase 3 LR scheduling; landed here so configs stay forward-compatible."""

    learning_rate: float = 3e-4
    batch_size: int = 32
    dropout_rate: float = 0.2
    random_seed: int = 1337
    splits: SplitType = (0.8, 0.2)
    iteration_type: Literal["chunking", "sliding"] = "chunking"

    model_config = SettingsConfigDict(
        env_prefix="TRAINING_",
        extra="allow",
    )


class ScratchGPTConfig(BaseSettings):
    """
    Full model config
    """

    architecture: ScratchGPTArchitecture = Field(default_factory=ScratchGPTArchitecture)
    training: ScratchGPTTraining = Field(default_factory=ScratchGPTTraining)

    model_config = SettingsConfigDict(
        env_prefix="SCRATCH_GPT_",
        extra="allow",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file="scratch_gpt.yaml"),
        )
