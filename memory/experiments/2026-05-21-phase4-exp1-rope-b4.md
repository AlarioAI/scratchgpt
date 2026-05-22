# Phase 4 Exp-1: RoPE on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

RoPE should be a better positional inductive bias than learned absolute position embeddings for a modern decoder path, especially as later work adds long-context evaluations. On the current B4 contract, however, the context length is only 256 and the run is only 5000 steps, so the expected single-ablation effect is modest rather than dramatic.

Prediction: B4 validation loss should be flat to slightly better than `baseline-p3a-b4-fineweb-edu-100m`, with a plausible range of 0% to -2%. A small regression would not rule out RoPE for longer-context work, but it would argue against promoting RoPE as a standalone default before the stack and long-context evals exist.

## Setup

Code delta:

- `ScratchGPTArchitecture.model_variant` selects `"classic"` or `"modern"`.
- `ScratchGPTArchitecture.position_encoding` selects `"learned"` or `"rope"`.
- `scratchgpt/model/model_modern.py` keeps exact learned-position parity with the classic path, and implements RoPE only when `model_variant="modern"` and `position_encoding="rope"`.
- The classic model path rejects `position_encoding="rope"` so RoPE cannot accidentally land inside `model.py`.

Training delta from the B4 baseline:

```text
model_variant = modern
position_encoding = rope
```

Everything else matches `runs/baseline-p3a-b4-fineweb-edu-100m`: standard 5000-step contract, B4 FineWeb-Edu 100M dataset key, GPT-2 tokenizer, accepted Phase 2 defaults, single RTX A6000.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp1-rope-b4 \
  --arch-override model_variant=modern \
  --arch-override position_encoding=rope \
  --device cuda
```

## Result

Run:

```text
runs/20260521-152220-exp1-rope-b4
```

Summary:

```text
best_val_loss:      4.608769744745419
best_val_step:      5000
tokens_per_sec:     22321.691607101104
total_wallclock_sec: 1834.9863765239716
peak_vram_bytes:    7537882624
```

Comparison against `runs/baseline-p3a-b4-fineweb-edu-100m`:

```text
| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-152220-exp1-rope-b4 |
|---|---|---|
| best_val_loss | 4.7369 | 4.6088 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 22,321.7 |
| peak_vram_bytes | 7,524,103,680 | 7,537,882,624 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,835.0 |
```

Derived deltas:

- Validation loss: -0.128149 absolute, -2.71%.
- Throughput: -7.36%.
- Wall-clock: +7.94%.
- Peak VRAM: +13,778,944 bytes, +0.18%.

Validation curve:

```text
| step | baseline val_loss | RoPE val_loss | RoPE delta |
|---:|---:|---:|---:|
| 500 | 6.241714 | 6.026380 | -0.215334 |
| 1000 | 5.783716 | 5.578805 | -0.204911 |
| 1500 | 5.481412 | 5.317397 | -0.164014 |
| 2000 | 5.282323 | 5.140749 | -0.141575 |
| 2500 | 5.140076 | 5.009099 | -0.130977 |
| 3000 | 5.033428 | 4.899757 | -0.133672 |
| 3500 | 4.936568 | 4.811327 | -0.125241 |
| 4000 | 4.855029 | 4.734474 | -0.120556 |
| 4500 | 4.795664 | 4.670555 | -0.125108 |
| 5000 | 4.736919 | 4.608770 | -0.128149 |
```

## Interpretation

RoPE beat the baseline at every validation checkpoint and ended slightly outside the optimistic edge of the predicted 0% to -2% range. The improvement is not a late-only artifact: the gap is already visible at step 500 and remains stable through the end of the run.

The tradeoff is speed. This first implementation computes rotary frequencies inside each attention head, so the -7.36% throughput hit is probably not the final cost of the idea. The memory cost is negligible relative to the B4 footprint.

Because the B4 contract is still short-context (`block_size=256`), this run does not prove the long-context value proposition for RoPE. It does show that RoPE is not merely a long-context tax here; it improves the current real-text benchmark enough to justify carrying it forward.

## Conclusion

Accept RoPE as a Phase 4 candidate for the modern path. Do not flip it to a project default yet.

The next isolated model ablation should be RMSNorm against the same B4 baseline, not RoPE+RMSNorm. Stacking should wait until each candidate has a clean single-change result.

## Open questions

- Can the RoPE implementation be optimized by caching rotary tables once per forward pass or per block, without making the modern path hard to read?
- Does RoPE still win after RMSNorm, SwiGLU, and QK-norm are tested in isolation?
- Does the B4 gain translate to longer-context evaluation once that benchmark exists?
