"""RunRecorder: writes one self-describing directory per experiment run."""
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def _collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            != ""
        )
        env["git_sha"] = sha
        env["git_dirty"] = dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        env["git_sha"] = None
        env["git_dirty"] = None
    return env


class RunRecorder:
    """Owns one timestamped run directory; appends metrics, writes summary on finalize."""

    def __init__(self, base_dir: Path, run_slug: str) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.run_dir = base_dir / f"{stamp}-{run_slug}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "checkpoints").mkdir()
        self._metrics_file = self.run_dir / "metrics.jsonl"
        self._finalized = False
        self._metric_history: list[dict[str, Any]] = []
        (self.run_dir / "env.json").write_text(json.dumps(_collect_env(), indent=2))

    def save_config(self, config_dict: dict[str, Any]) -> None:
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(config_dict, sort_keys=False))

    def log_metric(self, step: int, metrics: dict[str, float]) -> None:
        if self._finalized:
            raise RuntimeError("Cannot log to a finalized recorder")
        record = {"step": step, **metrics}
        self._metric_history.append(record)
        with self._metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def checkpoint_path(self, name: str) -> Path:
        return self.run_dir / "checkpoints" / name

    def finalize(self, extra: dict[str, Any] | None = None) -> None:
        """Write summary.json. Trainer should pass the true step count as extra['total_steps']."""
        if self._finalized:
            raise RuntimeError("Recorder already finalized")
        self._finalized = True

        last_logged_step = self._metric_history[-1]["step"] if self._metric_history else 0
        summary: dict[str, Any] = {
            "last_logged_step": last_logged_step,
            # Fallback; trainer overrides via extra for accuracy.
            "total_steps": last_logged_step,
        }
        val_entries = [m for m in self._metric_history if "val_loss" in m]
        if val_entries:
            best = min(val_entries, key=lambda m: m["val_loss"])
            summary["best_val_loss"] = best["val_loss"]
            summary["best_val_step"] = best["step"]
        if extra:
            summary.update(extra)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
