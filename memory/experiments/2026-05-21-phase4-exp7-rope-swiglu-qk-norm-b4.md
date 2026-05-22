# Phase 4 Exp-7: RoPE + SwiGLU + QK-Norm on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

Exp-6 made RoPE + QK-norm the best B4 quality stack so far, while Exp-5 showed that SwiGLU still adds a smaller but real improvement on top of RoPE. Exp-7 tests whether the feed-forward-side SwiGLU gain still exists once the attention side already has RoPE + QK-norm.

Prediction: the full stack should be flat to modestly better than RoPE + QK-norm (`4.580770680199141`). A meaningful win would be at least -0.2% versus RoPE + QK-norm and would make the full stack the leading quality candidate. Throughput should be the worst of the Phase 4 stack runs unless kernel scheduling unexpectedly hides the extra SwiGLU cost.

Decision rule: accept the full stack as the leading B4 quality candidate if it beats RoPE + QK-norm by at least ~0.2% without an additional systems regression larger than the quality gain can plausibly justify. If it is flat or worse, keep RoPE + QK-norm as the best quality stack and RoPE + SwiGLU as the cheaper practical stack.

## Setup

Training delta from the B4 baseline:

```text
model_variant = modern
position_encoding = rope
ffn_variant = swiglu
qk_norm = true
normalization = layernorm
```

RMSNorm is intentionally off because it regressed as a standalone B4 ablation.

Parameter count check:

```text
B4 baseline params:              49386577
B4 RoPE+SwiGLU+QK-norm params:   49295953
Delta:                           -90624
```

The full stack remains smaller than the baseline because RoPE removes the learned position table.

Preflight:

```text
uv run pytest -q
91 passed, 1 skipped

uv run ruff check .
All checks passed!

uv run mypy scratchgpt
Success: no issues found in 22 source files

uv run python scripts/bench.py --block-size 16 --embedding-size 16 --num-heads 4 \
  --num-blocks 1 --batch-size 2 --vocab-size 32 --iters 2 --warmup 1 \
  --device cpu --model-variant modern --position-encoding rope \
  --ffn-variant swiglu --qk-norm
tokens_per_sec: 7247.408360611512
params: 4390
```

Benchmark command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp7-rope-swiglu-qk-norm-b4 \
  --arch-override model_variant=modern \
  --arch-override position_encoding=rope \
  --arch-override ffn_variant=swiglu \
  --arch-override qk_norm=true \
  --device cuda
```

## Result

Run directory:

```text
runs/20260521-201522-exp7-rope-swiglu-qk-norm-b4
```

Summary:

```json
{
  "last_logged_step": 5000,
  "total_steps": 5000,
  "best_val_loss": 4.535660772581914,
  "best_val_step": 5000,
  "tokens_per_sec": 20796.332197673906,
  "total_wallclock_sec": 1969.57807803154,
  "peak_vram_bytes": 8048031232
}
```

`scripts/compare.py` versus B4 baseline:

| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-201522-exp7-rope-swiglu-qk-norm-b4 |
|---|---:|---:|
| best_val_loss | 4.7369 | 4.5357 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 20,796.3 |
| peak_vram_bytes | 7,524,103,680 | 8,048,031,232 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,969.6 |

Exact deltas versus B4 baseline:

```text
best_val_loss:      -0.201258013619 (-4.2487%)
tokens_per_sec:     -3298.27905946  (-13.6889%)
total_wallclock_sec:+269.612904787  (+15.8599%)
peak_vram_bytes:    +523927552      (+6.9633%)
```

`scripts/compare.py` versus Exp-6 RoPE + QK-norm:

| metric | 20260521-193831-exp6-rope-qk-norm-b4 | 20260521-201522-exp7-rope-swiglu-qk-norm-b4 |
|---|---:|---:|
| best_val_loss | 4.5808 | 4.5357 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 21,093.4 | 20,796.3 |
| peak_vram_bytes | 7,841,293,824 | 8,048,031,232 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,941.8 | 1,969.6 |

Exact deltas versus Exp-6:

```text
best_val_loss:      -0.0451099076172 (-0.9848%)
tokens_per_sec:     -297.110227236   (-1.4085%)
total_wallclock_sec:+27.7423560619   (+1.4287%)
peak_vram_bytes:    +206737408       (+2.6365%)
```

`scripts/compare.py` versus Exp-5 RoPE + SwiGLU:

| metric | 20260521-190327-exp5-rope-swiglu-b4 | 20260521-201522-exp7-rope-swiglu-qk-norm-b4 |
|---|---:|---:|
| best_val_loss | 4.5939 | 4.5357 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 22,017.7 | 20,796.3 |
| peak_vram_bytes | 7,738,328,576 | 8,048,031,232 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,860.3 | 1,969.6 |

Exact deltas versus Exp-5:

```text
best_val_loss:      -0.0581926762646 (-1.2668%)
tokens_per_sec:     -1221.34731107   (-5.5471%)
total_wallclock_sec:+109.254878044   (+5.8729%)
peak_vram_bytes:    +309702656       (+4.0022%)
```

Validation curve:

| step | baseline | Exp-5 RoPE+SwiGLU | Exp-6 RoPE+QK | Exp-7 full | Exp-7 - baseline | Exp-7 - Exp-6 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6.2417 | 5.9600 | 6.0255 | 5.9476 | -0.2941 | -0.0779 |
| 1000 | 5.7837 | 5.5086 | 5.5809 | 5.4942 | -0.2895 | -0.0867 |
| 1500 | 5.4814 | 5.2458 | 5.3138 | 5.2253 | -0.2562 | -0.0885 |
| 2000 | 5.2823 | 5.0755 | 5.1321 | 5.0425 | -0.2398 | -0.0896 |
| 2500 | 5.1401 | 4.9549 | 4.9980 | 4.9035 | -0.2366 | -0.0946 |
| 3000 | 5.0334 | 4.8539 | 4.8823 | 4.7969 | -0.2366 | -0.0854 |
| 3500 | 4.9366 | 4.7750 | 4.7868 | 4.7104 | -0.2261 | -0.0764 |
| 4000 | 4.8550 | 4.7050 | 4.7047 | 4.6426 | -0.2125 | -0.0622 |
| 4500 | 4.7957 | 4.6443 | 4.6419 | 4.5815 | -0.2142 | -0.0604 |
| 5000 | 4.7369 | 4.5939 | 4.5808 | 4.5357 | -0.2013 | -0.0451 |

## Interpretation

The full stack is the strongest B4 result so far. It beats the previous best-quality stack, RoPE + QK-norm, by `0.0451099076172` val loss (`-0.9848%`) while adding only `+1.43%` wallclock, `-1.41%` throughput, and `+2.64%` peak VRAM relative to Exp-6.

This clears the pre-registered acceptance threshold of about `-0.2%` versus RoPE + QK-norm. It also changes the interaction story: SwiGLU was only a small standalone gain and a modest RoPE add-on, but it still contributes materially once QK-norm is present. The full-stack curve is ahead of Exp-6 at every validation point, so the win is not a one-step artifact.

Relative to the B4 baseline, the cost is substantial: `-13.69%` throughput, `+15.86%` wallclock, and `+6.96%` peak VRAM. The quality improvement is also substantial at `-4.25%` val loss. This is a credible quality candidate, not a free default.

## Conclusion

Accept RoPE + SwiGLU + QK-norm as the current leading B4 quality stack.

Do not promote it to a broad default yet. The next evidence needed is either a repeat at the same B4 contract to estimate run-to-run variance, or a B1/B2/B3 cross-benchmark check to see whether the B4 win transfers to the older benchmark suite. Given this is now the quality leader by nearly 1% over the previous best stack, the preferred next step is a confirmatory B4 repeat before spending broader benchmark time.

## Open questions

- How large is B4 run-to-run variance under the current deterministic contract and GPU environment?
- Does the full stack transfer to B1/B2/B3, or is the effect specific to the FineWeb-Edu token distribution?
- If the repeat confirms the gain, should there be two documented recommendations: `quality_stack=rope+swiglu+qk_norm` and `practical_stack=rope+swiglu` or `rope+qk_norm` depending on throughput budget?
