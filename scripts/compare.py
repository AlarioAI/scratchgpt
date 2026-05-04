"""Print a markdown delta table across N experiment run directories.

Each run dir is expected to have a summary.json produced by RunRecorder.finalize().

Runs are required to share the same benchmark_contract (max_steps, block_size,
batch_size, learning_rate, random_seed, dropout_rate, iteration_type,
dataset_key) -- otherwise the comparison is meaningless. Override with
--allow-contract-mismatch for deliberate cross-budget comparisons.

Usage:
  python scripts/compare.py runs/baseline-b1 runs/20260504-*-gelu
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

METRICS_OF_INTEREST: list[str] = [
    "best_val_loss",
    "best_val_step",
    "tokens_per_sec",
    "peak_vram_bytes",
    "total_steps",
    "total_wallclock_sec",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", type=Path, nargs="+", help="Run directories to compare")
    p.add_argument(
        "--allow-contract-mismatch", action="store_true",
        help="Skip benchmark_contract equality check (use only when deliberate)",
    )
    return p.parse_args()


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        return {"_missing": True}
    return dict(json.loads(summary_file.read_text()))


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}" if abs(val) < 1000 else f"{val:,.1f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _check_contracts(summaries: list[tuple[str, dict[str, Any]]]) -> None:
    """Raise SystemExit if runs disagree on the benchmark contract."""
    contracts = [(name, s.get("benchmark_contract")) for name, s in summaries]
    missing = [name for name, c in contracts if c is None]
    if missing:
        print(f"ERROR: runs missing benchmark_contract: {missing}", file=sys.stderr)
        sys.exit(2)
    reference = contracts[0][1]
    mismatches: list[str] = []
    for name, contract in contracts[1:]:
        for key, ref_val in reference.items():
            if contract.get(key) != ref_val:
                mismatches.append(
                    f"  {name}.{key} = {contract.get(key)} != {ref_val} (from {contracts[0][0]})"
                )
    if mismatches:
        print(
            "ERROR: benchmark contract mismatch (pass --allow-contract-mismatch to override):",
            file=sys.stderr,
        )
        for m in mismatches:
            print(m, file=sys.stderr)
        sys.exit(2)


def main() -> None:
    args = parse_args()
    summaries = [(run_dir.name, _load_summary(run_dir)) for run_dir in args.run_dirs]

    if not args.allow_contract_mismatch:
        _check_contracts(summaries)

    header = "| metric | " + " | ".join(name for name, _ in summaries) + " |"
    sep = "|" + "|".join(["---"] * (len(summaries) + 1)) + "|"
    print(header)
    print(sep)

    for metric in METRICS_OF_INTEREST:
        row = f"| {metric} | " + " | ".join(_fmt(s.get(metric)) for _, s in summaries) + " |"
        print(row)


if __name__ == "__main__":
    main()
