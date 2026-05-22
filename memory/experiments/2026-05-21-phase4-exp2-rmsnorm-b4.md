# Phase 4 Exp-2: RMSNorm on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

RMSNorm removes mean-centering and norm bias parameters while keeping residual-stream scale control. In larger modern decoder stacks it is a common replacement for LayerNorm, but this model is small and the B4 context length is only 256, so the expected standalone quality lift is smaller than RoPE.

Prediction: B4 validation loss should be flat to modestly better than `baseline-p3a-b4-fineweb-edu-100m`, with a plausible range of +0.5% to -1.5%. Throughput may be flat to modestly faster, but the current simple RMSNorm implementation is not optimized, so speed is secondary to validation loss for this ablation.

Decision rule: accept RMSNorm as a Phase 4 candidate if it is quality-neutral or better on B4 without a meaningful throughput penalty. Reject it as a standalone candidate if it clearly worsens validation loss (>+0.5%) without compensating speed.

## Setup

Code delta:

- `ScratchGPTArchitecture.normalization` selects `"layernorm"` or `"rmsnorm"`.
- The classic model path rejects `normalization="rmsnorm"` so RMSNorm cannot accidentally land in `model.py`.
- The modern path still delegates to the classic model for the default learned-position LayerNorm configuration.
- RMSNorm uses learned absolute positions for this ablation. RoPE is intentionally off so Exp-2 measures RMSNorm in isolation against the B4 baseline, not against Exp-1.

Training delta from the B4 baseline:

```text
model_variant = modern
normalization = rmsnorm
position_encoding = learned
```

Preflight:

```text
uv run pytest tests/test_config.py tests/test_model_factory.py -q
18 passed

uv run ruff check .
All checks passed!

uv run mypy scratchgpt
Success: no issues found in 22 source files

uv run python scripts/bench.py --block-size 16 --embedding-size 16 --num-heads 4 \
  --num-blocks 1 --batch-size 2 --vocab-size 32 --iters 2 --warmup 1 \
  --device cpu --model-variant modern --normalization rmsnorm
tokens_per_sec: 14059.629638055894
params: 4528
```

Benchmark command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp2-rmsnorm-b4 \
  --arch-override model_variant=modern \
  --arch-override normalization=rmsnorm \
  --device cuda
```

## Result

Run:

```text
runs/20260521-155833-exp2-rmsnorm-b4
```

Summary:

```text
best_val_loss:       4.744145674966336
best_val_step:       5000
tokens_per_sec:      23335.13244001511
total_wallclock_sec: 1755.2932302951813
peak_vram_bytes:     7685622784
```

Comparison against `runs/baseline-p3a-b4-fineweb-edu-100m`:

```text
| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-155833-exp2-rmsnorm-b4 |
|---|---|---|
| best_val_loss | 4.7369 | 4.7441 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 23,335.1 |
| peak_vram_bytes | 7,524,103,680 | 7,685,622,784 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,755.3 |
```

Derived deltas:

- Validation loss: +0.007227 absolute, +0.15%.
- Throughput: -3.15%.
- Wall-clock: +3.25%.
- Peak VRAM: +161,519,104 bytes, +2.15%.

Validation curve:

```text
| step | baseline val_loss | RMSNorm val_loss | RMSNorm delta |
|---:|---:|---:|---:|
| 500 | 6.241714 | 6.244467 | +0.002754 |
| 1000 | 5.783716 | 5.788703 | +0.004987 |
| 1500 | 5.481412 | 5.487833 | +0.006421 |
| 2000 | 5.282323 | 5.291565 | +0.009242 |
| 2500 | 5.140076 | 5.149522 | +0.009446 |
| 3000 | 5.033428 | 5.039777 | +0.006349 |
| 3500 | 4.936568 | 4.943556 | +0.006988 |
| 4000 | 4.855029 | 4.865003 | +0.009973 |
| 4500 | 4.795664 | 4.801883 | +0.006219 |
| 5000 | 4.736919 | 4.744146 | +0.007227 |
```

## Interpretation

RMSNorm is not catastrophically bad, but it is consistently worse than LayerNorm across the full validation curve. The final +0.15% regression is inside the prediction band and below the hard rejection threshold, but the direction is uniform and there is no compensating systems benefit: throughput is lower and peak VRAM is higher.

The extra VRAM is likely an implementation artifact of the manual RMSNorm expression rather than an inherent property of the normalization choice. That does not change the decision for this ablation, because the benchmark contract measures the implementation we would actually ship.

This result argues against RMSNorm as an isolated Phase 4 improvement on the current B4 contract. It does not rule out RMSNorm inside a later modern stack, especially if SwiGLU or RoPE interactions alter optimization.

## Conclusion

Do not accept RMSNorm as a standalone Phase 4 candidate.

Keep the flag and implementation available for later interaction tests, but do not promote it into the candidate stack on the basis of this run alone. The next isolated ablation should be parameter-matched SwiGLU against the B4 baseline.

## Open questions

- Is the VRAM increase from the manual RMSNorm expression avoidable with a fused or lower-allocation implementation?
- Does RMSNorm become useful only when stacked with SwiGLU or RoPE?
- If a future stack uses RMSNorm, should it be tested as RoPE+RMSNorm before adding SwiGLU?
