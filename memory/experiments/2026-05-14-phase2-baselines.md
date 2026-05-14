# Generation-2 baselines after Phase 2 defaults

**Date:** 2026-05-14
**Phase:** 2
**Branch / commit:** `feature/sota-upgrade-roadmap` @ `fb86c14`

## Hypothesis

This is a baseline-establishing run, not a new model hypothesis. The expected result is that the accepted Phase 2 defaults should preserve the Exp-4 quality gains when rerun as canonical `baseline-p2-*` artifacts. These runs become the comparison target for the next model-first work until a deliberate benchmark refresh creates a new generation.

## Setup

The benchmark contract matches the Phase 1 baselines: 5000 steps, block size 256, batch size 32, learning rate 3e-4, seed 1337, dropout 0.1, and `iteration_type="chunking"`.

The architecture size also matches Phase 1: `embedding_size=384`, `num_heads=6`, `num_blocks=6`. The Phase 2 defaults are:

```text
attention_scale_mode = "head"
ffn_activation = "gelu"
init_scheme = "gpt2"
tie_weights = false
use_bias = true
```

Runs were executed sequentially on a single NVIDIA GeForce RTX 5090 with torch 2.8.0 + CUDA 12.8 and Python 3.12.9. B2 used the cached Lichess corpus and required the `examples-dependencies` extra so `zstandard` was present for the chess example import.

Commands:

```bash
TMPDIR=/home/ayeganov/tmp/scratchgpt uv run python benchmarks/b1_tinystories.py \
  --slug baseline-p2-b1-tinystories --device cuda

TMPDIR=/home/ayeganov/tmp/scratchgpt uv run --extra examples-dependencies python benchmarks/b2_chess.py \
  --slug baseline-p2-b2-chess --device cuda

TMPDIR=/home/ayeganov/tmp/scratchgpt uv run python benchmarks/b3_chemistry.py \
  --slug baseline-p2-b3-chemistry --device cuda
```

The timestamped run directories were canonicalized to:

- `runs/baseline-p2-b1-tinystories/summary.json`
- `runs/baseline-p2-b2-chess/summary.json`
- `runs/baseline-p2-b3-chemistry/summary.json`

The original Phase 1 baselines were renamed from `runs/baseline-*` to `runs/baseline-p1-*` so generation labels are explicit.

## Result

Same-dataset comparisons against the renamed Phase 1 baselines:

```bash
uv run python scripts/compare.py runs/baseline-p1-b1-tinystories runs/baseline-p2-b1-tinystories
```

| metric | baseline-p1-b1-tinystories | baseline-p2-b1-tinystories |
|---|---|---|
| best_val_loss | 2.2261 | 1.9778 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 19,411.3 | 46,745.2 |
| peak_vram_bytes | 7,224,604,160 | 7,524,103,680 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 2,110.1 | 876.2405 |

```bash
uv run python scripts/compare.py runs/baseline-p1-b2-chess runs/baseline-p2-b2-chess
```

| metric | baseline-p1-b2-chess | baseline-p2-b2-chess |
|---|---|---|
| best_val_loss | 2.1694 | 2.0188 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 83,015.7 | 142,864.8 |
| peak_vram_bytes | 3,146,654,208 | 3,448,250,880 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 493.4007 | 286.7046 |

```bash
uv run python scripts/compare.py runs/baseline-p1-b3-chemistry runs/baseline-p2-b3-chemistry
```

| metric | baseline-p1-b3-chemistry | baseline-p2-b3-chemistry |
|---|---|---|
| best_val_loss | 0.2772 | 0.2515 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 86,728.4 | 141,625.0 |
| peak_vram_bytes | 1,906,259,968 | 2,196,191,744 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 472.2792 | 289.2146 |

P1 to P2 loss deltas:

| benchmark | delta | delta % |
|-----------|-------|---------|
| B1 TinyStories | -0.2483 | -11.15% |
| B2 Chess | -0.1505 | -6.94% |
| B3 Chemistry | -0.0257 | -9.27% |

## Interpretation

The Phase 2 defaults reproduced the expected quality lift and are now accepted as generation-2 baselines. The exact numbers differ slightly from the earlier Exp-4 runs, especially B2, because these were fresh canonical runs on different hardware. Loss, not throughput, is the algorithmic claim here.

Throughput improved substantially relative to P1, but the comparison is confounded by GPU change from RTX A6000 to RTX 5090. The throughput numbers are still useful for planning local runtime, not for claiming that Phase 2 was faster algorithmically.

## Conclusion

Generation-2 baselines are accepted. Future Phase 3/model-first experiments should compare against `runs/baseline-p2-*` unless the benchmark suite is intentionally refreshed and a new baseline generation is created.

## Open questions

The B2 benchmark currently needs `uv run --extra examples-dependencies` in a clean uv environment because `examples.chess` imports `zstandard`. That is documented here for reproducibility; it can be cleaned up later as benchmark ergonomics work.
