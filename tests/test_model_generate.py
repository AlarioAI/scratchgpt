import torch

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTConfig, ScratchGPTTraining
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.training.determinism import seed_everything


def test_generate_accepts_sampling_kwargs_and_preserves_output_shape() -> None:
    seed_everything(1337)
    config = ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=8,
            embedding_size=16,
            num_heads=4,
            num_blocks=1,
            vocab_size=32,
        ),
        training=ScratchGPTTraining(batch_size=1, dropout_rate=0.0),
    )
    model = TransformerLanguageModel(config)
    model.eval()

    context = torch.tensor([[1, 2, 3]])
    generated = model.generate(
        context,
        max_new_tokens=2,
        temperature=1.0,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    assert generated.shape == (1, 5)
