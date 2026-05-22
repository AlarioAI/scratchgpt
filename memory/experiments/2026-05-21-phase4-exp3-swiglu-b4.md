# Phase 4 Exp-3: Parameter-Matched SwiGLU on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

SwiGLU should improve the feed-forward block by adding multiplicative gating while keeping roughly the same weight budget as the classic 4x MLP. For `embedding_size=384`, the parameter-matched hidden size is `round(8 * d / 3) = 1024`, giving nearly the same FFN matrix parameter count as the baseline MLP.

Prediction: B4 validation loss should be modestly better than `baseline-p3a-b4-fineweb-edu-100m`, with a plausible range of 0% to -2.5%. Throughput should be flat to modestly slower because this implementation uses three explicit Linear layers plus a gate multiply, even though the dominant matrix-multiply count is matched.

Decision rule: accept SwiGLU as a Phase 4 candidate if it improves validation loss by at least ~0.5% without a large throughput regression. Treat a flat result as inconclusive, and reject a clear validation-loss regression.

## Setup

Code delta:

- `ScratchGPTArchitecture.ffn_variant` selects `"mlp"` or `"swiglu"`.
- The classic model path rejects `ffn_variant="swiglu"` so the pedagogical `model.py` remains unchanged.
- The modern path still delegates to the classic model for the default learned-position LayerNorm MLP configuration.
- SwiGLU uses `SiLU(gate(x)) * value(x)` followed by a residual projection and dropout.

Training delta from the B4 baseline:

```text
model_variant = modern
ffn_variant = swiglu
position_encoding = learned
normalization = layernorm
```

RoPE and RMSNorm are intentionally off, so Exp-3 measures SwiGLU in isolation against the B4 baseline.

Parameter count check:

```text
B4 MLP params:    49386577
B4 SwiGLU params: 49389649
Delta:            +3072
```

Preflight:

```text
uv run pytest tests/test_config.py tests/test_model_factory.py -q
21 passed

uv run ruff check .
All checks passed!

uv run mypy scratchgpt
Success: no issues found in 22 source files

uv run python scripts/bench.py --block-size 16 --embedding-size 16 --num-heads 4 \
  --num-blocks 1 --batch-size 2 --vocab-size 32 --iters 2 --warmup 1 \
  --device cpu --model-variant modern --ffn-variant swiglu
tokens_per_sec: 14982.570362366427
params: 4614
```

Benchmark command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp3-swiglu-b4 \
  --arch-override model_variant=modern \
  --arch-override ffn_variant=swiglu \
  --device cuda
```

## Result

Run:

```text
runs/20260521-163244-exp3-swiglu-b4
```

Summary:

```text
best_val_loss:       4.699056146756427
best_val_step:       5000
tokens_per_sec:      23491.378944500426
total_wallclock_sec: 1743.6183757781982
peak_vram_bytes:     7730054656
```

Comparison against `runs/baseline-p3a-b4-fineweb-edu-100m`:

```text
| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-163244-exp3-swiglu-b4 |
|---|---|---|
| best_val_loss | 4.7369 | 4.6991 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 23,491.4 |
| peak_vram_bytes | 7,524,103,680 | 7,730,054,656 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,743.6 |
```

Derived deltas:

- Validation loss: -0.037863 absolute, -0.80%.
- Throughput: -2.50%.
- Wall-clock: +2.57%.
- Peak VRAM: +205,950,976 bytes, +2.74%.

Validation curve:

```text
| step | baseline val_loss | SwiGLU val_loss | SwiGLU delta |
|---:|---:|---:|---:|
| 500 | 6.241714 | 6.249833 | +0.008120 |
| 1000 | 5.783716 | 5.745614 | -0.038102 |
| 1500 | 5.481412 | 5.438110 | -0.043302 |
| 2000 | 5.282323 | 5.240759 | -0.041564 |
| 2500 | 5.140076 | 5.092416 | -0.047659 |
| 3000 | 5.033428 | 4.987244 | -0.046185 |
| 3500 | 4.936568 | 4.895358 | -0.041210 |
| 4000 | 4.855029 | 4.818687 | -0.036342 |
| 4500 | 4.795664 | 4.755935 | -0.039729 |
| 5000 | 4.736919 | 4.699056 | -0.037863 |
```

## Interpretation

SwiGLU is a real but smaller win than RoPE on B4. The first validation checkpoint is slightly worse, but every checkpoint from step 1000 onward is better than the baseline, with the final improvement at -0.80%.

The systems cost is moderate and acceptable for a candidate ablation: -2.50% throughput and +2.74% VRAM. Unlike RoPE, the runtime cost is small enough that no optimization follow-up is needed before stack testing.

The parameter match did what it was intended to do: the model only added 3,072 parameters at B4 scale, so the loss improvement is not explainable as a meaningful capacity increase.

## Conclusion

Accept parameter-matched SwiGLU as a Phase 4 candidate for the modern path. Do not make it a default yet.

Current isolated-candidate status on B4:

- RoPE: accepted candidate.
- RMSNorm: not accepted standalone.
- SwiGLU: accepted candidate.

The next isolated ablation should be QK-norm against the same B4 baseline. Stack runs should wait until QK-norm has its own single-change result.

## Open questions

- Does SwiGLU stack constructively with RoPE, or are their B4 gains partially overlapping?
- Is the step-500 regression just noise, or does SwiGLU change early optimization in a repeatable way?
- Would a fused gate/value projection improve throughput without hurting readability?
