import json
import subprocess
import sys
from pathlib import Path

STANDARD_CONTRACT = {
    "max_steps": 5000, "block_size": 256, "batch_size": 32,
    "learning_rate": 3e-4, "random_seed": 1337,
    "dropout_rate": 0.1, "iteration_type": "chunking",
    "dataset_key": "tinystories-500000-rows",
}


def _make_run(dir_: Path, best_val_loss: float, tokens_per_sec: float, contract: dict | None = None) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "summary.json").write_text(json.dumps({
        "best_val_loss": best_val_loss,
        "best_val_step": 100,
        "total_steps": 5000,
        "tokens_per_sec": tokens_per_sec,
        "benchmark_contract": contract if contract is not None else STANDARD_CONTRACT,
    }))
    (dir_ / "config.yaml").write_text("architecture:\n  block_size: 256\n")


def test_compare_prints_delta_table(tmp_path: Path) -> None:
    _make_run(tmp_path / "baseline", 2.0, 1000.0)
    _make_run(tmp_path / "variant-a", 1.8, 950.0)
    _make_run(tmp_path / "variant-b", 1.7, 900.0)

    result = subprocess.run(
        [
            sys.executable, "scripts/compare.py",
            str(tmp_path / "baseline"),
            str(tmp_path / "variant-a"),
            str(tmp_path / "variant-b"),
        ],
        capture_output=True, text=True, check=True,
    )
    out = result.stdout
    assert "baseline" in out
    assert "variant-a" in out
    assert "variant-b" in out
    assert "best_val_loss" in out
    assert "tokens_per_sec" in out
    assert "|" in out


def test_compare_errors_on_contract_mismatch(tmp_path: Path) -> None:
    """If two runs disagree on the benchmark contract, comparison is meaningless — hard error."""
    _make_run(tmp_path / "baseline", 2.0, 1000.0)
    _make_run(
        tmp_path / "bad-variant", 1.5, 950.0,
        contract={**STANDARD_CONTRACT, "max_steps": 1000},
    )
    result = subprocess.run(
        [
            sys.executable, "scripts/compare.py",
            str(tmp_path / "baseline"),
            str(tmp_path / "bad-variant"),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "contract mismatch" in (result.stdout + result.stderr).lower()
    assert "max_steps" in (result.stdout + result.stderr)


def test_compare_allows_contract_mismatch_with_flag(tmp_path: Path) -> None:
    """Opt-out for when you deliberately want to compare across budgets."""
    _make_run(tmp_path / "baseline", 2.0, 1000.0)
    _make_run(
        tmp_path / "short-variant", 2.5, 950.0,
        contract={**STANDARD_CONTRACT, "max_steps": 1000},
    )
    result = subprocess.run(
        [
            sys.executable, "scripts/compare.py",
            "--allow-contract-mismatch",
            str(tmp_path / "baseline"),
            str(tmp_path / "short-variant"),
        ],
        capture_output=True, text=True, check=True,
    )
    assert "|" in result.stdout
