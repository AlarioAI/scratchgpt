"""Step-based training loop with eval cadence and RunRecorder integration.

Backwards-compat: if config.max_steps is None, falls back to epoch-based training.
"""
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm

from scratchgpt.config import ScratchGPTTraining
from scratchgpt.core.types import DictTensorLoader
from scratchgpt.data.datasource import DataSource
from scratchgpt.metering import AverageValueMeter
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer.base_tokenizer import Tokenizer
from scratchgpt.training.run_recorder import RunRecorder


class Trainer:
    """Step-based trainer. Logs every `log_every_steps`, evaluates every `eval_every_steps`."""

    def __init__(
        self,
        model: TransformerLanguageModel,
        config: ScratchGPTTraining,
        optimizer: Optimizer,
        experiment_path: Path,
        device: torch.device,
        recorder: RunRecorder | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.experiment_path = experiment_path
        self.device = device
        self.recorder = recorder
        self.experiment_path.mkdir(exist_ok=True, parents=True)

    def _forward_loss(self, batch: dict[str, Tensor]) -> Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        logits = self.model(input_ids)
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

    def _evaluate(self, val_loader: DictTensorLoader) -> float:
        self.model.eval()
        meter = AverageValueMeter()
        with torch.no_grad():
            for batch in val_loader:
                loss = self._forward_loss(batch)
                meter.add(loss.item())
        self.model.train()
        return float(meter.value()[0])

    def train(self, data_source: DataSource, tokenizer: Tokenizer) -> None:
        train_loader, val_loader = data_source.get_dataloaders(
            tokenizer=tokenizer,
            block_size=self.model._block_size,
            batch_size=self.config.batch_size,
            splits=self.config.splits,
            random_seed=self.config.random_seed,
            iteration_type=self.config.iteration_type,
        )

        if self.config.max_steps is not None:
            self._train_by_steps(train_loader, val_loader)
        else:
            self._train_by_epochs(train_loader, val_loader)

    def _train_by_steps(
        self,
        train_loader: DictTensorLoader,
        val_loader: DictTensorLoader | None,
    ) -> None:
        assert self.config.max_steps is not None
        max_steps = self.config.max_steps
        eval_every = self.config.eval_every_steps
        log_every = self.config.log_every_steps

        best_val_loss = float("inf")
        step = 0
        meter = AverageValueMeter()
        t_start = time.time()

        self.model.train()
        pbar = tqdm(total=max_steps, desc="Train", file=sys.stdout)
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break

                self.optimizer.zero_grad(set_to_none=True)
                loss = self._forward_loss(batch)
                loss.backward()  # type: ignore[no-untyped-call]
                self.optimizer.step()

                step += 1
                meter.add(loss.item())
                pbar.update(1)

                if step % log_every == 0 and self.recorder is not None:
                    mean, _ = meter.value()
                    self.recorder.log_metric(step, {"train_loss": mean})
                    meter.reset()

                if step % eval_every == 0 or step == max_steps:
                    if val_loader is not None:
                        val_loss = self._evaluate(val_loader)
                        if self.recorder is not None:
                            self.recorder.log_metric(step, {"val_loss": val_loss})
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            if self.recorder is not None:
                                torch.save(self.model.state_dict(), self.recorder.checkpoint_path("best.pt"))
                    if self.recorder is not None:
                        torch.save(self.model.state_dict(), self.recorder.checkpoint_path("last.pt"))
        pbar.close()

        elapsed = time.time() - t_start
        tokens_per_step = self.config.batch_size * self.model._block_size
        if self.recorder is not None:
            self.recorder.finalize(
                extra={
                    "total_steps": step,
                    "tokens_per_sec": tokens_per_step * step / max(elapsed, 1e-9),
                    "total_wallclock_sec": elapsed,
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None
                    ),
                    "benchmark_contract": {
                        "max_steps": self.config.max_steps,
                        "block_size": self.model._block_size,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "random_seed": self.config.random_seed,
                        "dropout_rate": self.config.dropout_rate,
                        "iteration_type": self.config.iteration_type,
                        "dataset_key": getattr(self, "_dataset_key", "unspecified"),
                    },
                }
            )

    def _train_by_epochs(
        self,
        train_loader: DictTensorLoader,
        val_loader: DictTensorLoader | None,
    ) -> None:
        """Legacy epoch-based loop. Preserves checkpoint filenames and the epoch print loop, but replaces
        the per-batch running-std meter with _evaluate (mean only) and drops the no-val-dataset emoji
        message. Acceptable because epoch mode is a fallback for back-compat; new work uses step mode."""
        best_val_loss = float("inf")
        latest_model_path = self.experiment_path / "latest_model_weights.pth"
        best_model_path = self.experiment_path / "best_model_weights.pth"
        step = 0

        for epoch in range(self.config.max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.max_epochs} ---")
            self.model.train()
            meter = AverageValueMeter()
            for batch in tqdm(train_loader, desc="Train", file=sys.stdout):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._forward_loss(batch)
                loss.backward()  # type: ignore[no-untyped-call]
                self.optimizer.step()
                step += 1
                meter.add(loss.item())
            mean_train, _ = meter.value()
            print(f"Train Loss: {mean_train:.4f}")
            torch.save(self.model.state_dict(), latest_model_path)

            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
                if self.recorder is not None:
                    self.recorder.log_metric(step, {"train_loss": mean_train, "val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                    torch.save(self.model.state_dict(), best_model_path)
            else:
                if self.recorder is not None:
                    self.recorder.log_metric(step, {"train_loss": mean_train})
                torch.save(self.model.state_dict(), best_model_path)

        if self.recorder is not None:
            self.recorder.finalize()
