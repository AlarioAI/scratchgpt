"""One-off script: run current-main training for 200 steps on TinyStories to confirm baseline reproducibility."""
import tempfile
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from scratchgpt import (
    CharTokenizer,
    ScratchGPTArchitecture,
    ScratchGPTConfig,
    ScratchGPTTraining,
    Trainer,
    TransformerLanguageModel,
)
from scratchgpt.data import create_data_source

TEXT = "hello world " * 10_000  # ~100kB smoke corpus


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_file = tmp_path / "smoke.txt"
        data_file.write_text(TEXT)

        tokenizer = CharTokenizer(text=TEXT)
        config = ScratchGPTConfig(
            architecture=ScratchGPTArchitecture(
                block_size=64, embedding_size=64, num_heads=4, num_blocks=2,
                vocab_size=tokenizer.vocab_size,
            ),
            training=ScratchGPTTraining(
                max_epochs=1, learning_rate=3e-4, batch_size=16,
                dropout_rate=0.1, random_seed=1337,
            ),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1337)
        model = TransformerLanguageModel(config).to(device)
        optimizer = AdamW(model.parameters(), lr=3e-4)
        data_source = create_data_source(str(data_file))

        trainer = Trainer(model, config.training, optimizer, tmp_path / "exp", device)
        t0 = time.time()
        trainer.train(data_source, tokenizer)
        print(f"DONE in {time.time() - t0:.1f}s on {device}")


if __name__ == "__main__":
    main()
