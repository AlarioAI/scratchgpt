"""Tests for step-based training loop.

We use a tiny synthetic dataset + tiny model so each test runs in < 5s.
"""
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTConfig, ScratchGPTTraining
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.training.run_recorder import RunRecorder
from scratchgpt.training.trainer import Trainer


class _ToyLMDataset(Dataset[dict[str, torch.Tensor]]):
    """Random token ids, for shape/flow tests only -- loss won't converge."""

    def __init__(self, n: int, block_size: int, vocab_size: int) -> None:
        torch.manual_seed(0)
        self._data = torch.randint(0, vocab_size, (n, block_size + 1))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        block = self._data[idx]
        return {"input_ids": block[:-1], "labels": block[1:]}


class _StaticDataSource:
    """DataSource double that returns pre-built loaders, bypassing tokenization."""

    def __init__(
        self,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        self._train, self._val = train_loader, val_loader

    def get_dataloaders(self, **_kwargs: object) -> tuple[
        DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]
    ]:
        return self._train, self._val


def _make_setup(tmp_path: Path, max_steps: int, eval_every_steps: int) -> tuple[
    Trainer, _StaticDataSource, RunRecorder
]:
    vocab_size, block_size = 32, 16
    arch = ScratchGPTArchitecture(
        block_size=block_size, embedding_size=16, num_heads=2, num_blocks=1, vocab_size=vocab_size
    )
    training = ScratchGPTTraining(
        max_epochs=9999,
        max_steps=max_steps,
        eval_every_steps=eval_every_steps,
        log_every_steps=1,
        batch_size=4,
        dropout_rate=0.0,
    )
    config = ScratchGPTConfig(architecture=arch, training=training)

    device = torch.device("cpu")
    model = TransformerLanguageModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_loader = DataLoader(_ToyLMDataset(n=64, block_size=block_size, vocab_size=vocab_size), batch_size=4)
    val_loader = DataLoader(_ToyLMDataset(n=16, block_size=block_size, vocab_size=vocab_size), batch_size=4)
    data_source = _StaticDataSource(train_loader, val_loader)

    recorder = RunRecorder(base_dir=tmp_path / "runs", run_slug="trainer-test")
    trainer = Trainer(
        model=model,
        config=config.training,
        optimizer=optimizer,
        experiment_path=tmp_path / "exp",
        device=device,
        recorder=recorder,
    )
    return trainer, data_source, recorder


def test_trainer_stops_at_max_steps(tmp_path: Path) -> None:
    trainer, data_source, recorder = _make_setup(tmp_path, max_steps=10, eval_every_steps=5)
    trainer.train(data_source=data_source, tokenizer=None)
    last_metric = recorder._metric_history[-1]  # noqa: SLF001
    assert last_metric["step"] == 10


def test_trainer_logs_at_eval_cadence(tmp_path: Path) -> None:
    trainer, data_source, recorder = _make_setup(tmp_path, max_steps=20, eval_every_steps=5)
    trainer.train(data_source=data_source, tokenizer=None)
    val_steps = [m["step"] for m in recorder._metric_history if "val_loss" in m]  # noqa: SLF001
    assert val_steps == [5, 10, 15, 20]


def test_trainer_saves_best_checkpoint(tmp_path: Path) -> None:
    trainer, data_source, recorder = _make_setup(tmp_path, max_steps=10, eval_every_steps=5)
    trainer.train(data_source=data_source, tokenizer=None)
    assert recorder.checkpoint_path("best.pt").exists()
    assert recorder.checkpoint_path("last.pt").exists()


def test_trainer_falls_back_to_epochs_when_max_steps_none(tmp_path: Path) -> None:
    """With max_steps=None, trainer should run max_epochs (old behavior)."""
    vocab_size, block_size = 32, 16
    arch = ScratchGPTArchitecture(
        block_size=block_size, embedding_size=16, num_heads=2, num_blocks=1, vocab_size=vocab_size
    )
    training = ScratchGPTTraining(
        max_epochs=2, max_steps=None, eval_every_steps=500, log_every_steps=1,
        batch_size=4, dropout_rate=0.0,
    )
    config = ScratchGPTConfig(architecture=arch, training=training)
    model = TransformerLanguageModel(config).to(torch.device("cpu"))
    optimizer = AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(_ToyLMDataset(n=8, block_size=block_size, vocab_size=vocab_size), batch_size=4)
    val_loader = DataLoader(_ToyLMDataset(n=4, block_size=block_size, vocab_size=vocab_size), batch_size=4)
    data_source = _StaticDataSource(train_loader, val_loader)
    recorder = RunRecorder(base_dir=tmp_path / "runs", run_slug="epoch-fallback")

    trainer = Trainer(model, config.training, optimizer, tmp_path / "exp", torch.device("cpu"), recorder=recorder)
    trainer.train(data_source=data_source, tokenizer=None)
    # 2 epochs * 2 steps/epoch = 4 optimizer steps
    assert recorder._metric_history[-1]["step"] == 4  # noqa: SLF001
