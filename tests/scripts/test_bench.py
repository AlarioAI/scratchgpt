import subprocess
import sys
from pathlib import Path


def test_bench_script_reports_tokens_per_sec() -> None:
    result = subprocess.run(
        [
            sys.executable, "scripts/bench.py",
            "--block-size", "32",
            "--embedding-size", "32",
            "--num-heads", "2",
            "--num-blocks", "1",
            "--batch-size", "4",
            "--vocab-size", "64",
            "--iters", "5",
            "--device", "cpu",
        ],
        capture_output=True, text=True, check=True, cwd=Path.cwd(),
    )
    assert "tokens_per_sec" in result.stdout
    assert "peak_vram_bytes" in result.stdout
    assert "forward_ms" in result.stdout
