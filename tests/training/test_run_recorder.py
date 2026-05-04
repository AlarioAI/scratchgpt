import json
from pathlib import Path

import pytest

from scratchgpt.training.run_recorder import RunRecorder


def test_recorder_creates_run_directory(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="unit-test")
    assert recorder.run_dir.exists()
    assert recorder.run_dir.parent == tmp_path
    assert recorder.run_dir.name.endswith("-unit-test")


def test_recorder_writes_env_json(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="env-test")
    env_file = recorder.run_dir / "env.json"
    assert env_file.exists()
    env = json.loads(env_file.read_text())
    assert "torch_version" in env
    assert "python_version" in env
    assert "git_sha" in env


def test_recorder_writes_config_yaml(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="cfg-test")
    recorder.save_config({"architecture": {"block_size": 256}})
    cfg_file = recorder.run_dir / "config.yaml"
    assert cfg_file.exists()
    assert "block_size" in cfg_file.read_text()


def test_recorder_appends_metrics_jsonl(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="metrics-test")
    recorder.log_metric(step=0, metrics={"train_loss": 2.5})
    recorder.log_metric(step=10, metrics={"train_loss": 2.3, "val_loss": 2.4})
    lines = (recorder.run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 0
    assert first["train_loss"] == 2.5
    second = json.loads(lines[1])
    assert second["val_loss"] == 2.4


def test_recorder_finalize_writes_summary(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="summary-test")
    recorder.log_metric(step=0, metrics={"val_loss": 3.0})
    recorder.log_metric(step=100, metrics={"val_loss": 2.0})
    recorder.log_metric(step=200, metrics={"val_loss": 2.1})
    # Trainer passes true step count via extra; recorder must NOT overwrite it
    # with the last-logged step.
    recorder.finalize(extra={"tokens_per_sec": 1234.5, "total_steps": 237})
    summary = json.loads((recorder.run_dir / "summary.json").read_text())
    assert summary["best_val_loss"] == 2.0
    assert summary["best_val_step"] == 100
    assert summary["total_steps"] == 237
    assert summary["last_logged_step"] == 200
    assert summary["tokens_per_sec"] == 1234.5


def test_recorder_finalize_falls_back_to_last_logged_step(tmp_path: Path) -> None:
    """If trainer doesn't pass total_steps, fall back to last-logged step."""
    recorder = RunRecorder(base_dir=tmp_path, run_slug="fallback-test")
    recorder.log_metric(step=42, metrics={"val_loss": 1.0})
    recorder.finalize()
    summary = json.loads((recorder.run_dir / "summary.json").read_text())
    assert summary["last_logged_step"] == 42
    assert summary["total_steps"] == 42


def test_recorder_writes_benchmark_contract(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="contract-test")
    recorder.finalize(extra={
        "benchmark_contract": {
            "max_steps": 5000, "block_size": 256, "batch_size": 32,
            "learning_rate": 3e-4, "random_seed": 1337,
        },
    })
    summary = json.loads((recorder.run_dir / "summary.json").read_text())
    assert summary["benchmark_contract"]["max_steps"] == 5000


def test_recorder_rejects_double_finalize(tmp_path: Path) -> None:
    recorder = RunRecorder(base_dir=tmp_path, run_slug="double-final")
    recorder.finalize()
    with pytest.raises(RuntimeError):
        recorder.finalize()
