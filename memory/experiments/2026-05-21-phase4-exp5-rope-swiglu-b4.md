# Phase 4 Exp-5: RoPE + Parameter-Matched SwiGLU on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

RoPE and SwiGLU improved B4 in isolation through different parts of the transformer block: RoPE changed attention position handling, while SwiGLU changed the feed-forward transformation. Because their mechanisms are mostly separate, the stack should improve over either isolated result.

Prediction: validation loss should land below the RoPE-only result (`4.608769744745419`) and plausibly near an additive absolute improvement over the B4 baseline, around `4.57` to `4.60`. Throughput should be slower than both isolated runs, likely around -9% to -11% versus the B4 baseline if the costs combine roughly independently.

Decision rule: accept RoPE + SwiGLU as the leading Phase 4 stack if it beats RoPE-only validation loss by at least ~0.3% without a systems regression much worse than the isolated costs imply. If it matches RoPE but does not beat it, keep RoPE as the leading candidate and test RoPE + QK-norm next. If it regresses versus RoPE, treat the interaction as negative and do not add SwiGLU to the default stack without a repeat.

## Setup

Training delta from the B4 baseline:

```text
model_variant = modern
position_encoding = rope
ffn_variant = swiglu
normalization = layernorm
qk_norm = false
```

RMSNorm and QK-norm are intentionally off, so Exp-5 measures only the RoPE + SwiGLU interaction against the B4 baseline and the two isolated winners.

Parameter count check:

```text
B4 baseline params:    49386577
B4 RoPE+SwiGLU params: 49291345
Delta:                 -95232
```

The stack has fewer parameters than baseline because RoPE removes the learned position table, and the parameter-matched SwiGLU adds only a small FFN delta.

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
  --device cpu --model-variant modern --position-encoding rope --ffn-variant swiglu
tokens_per_sec: 8719.269699266648
params: 4358
```

Benchmark command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp5-rope-swiglu-b4 \
  --arch-override model_variant=modern \
  --arch-override position_encoding=rope \
  --arch-override ffn_variant=swiglu \
  --device cuda
```

## Result

Run:

```text
runs/20260521-190327-exp5-rope-swiglu-b4
```

Summary:

```text
best_val_loss:       4.593853448846535
best_val_step:       5000
tokens_per_sec:      22017.679508741905
total_wallclock_sec: 1860.3231999874115
peak_vram_bytes:     7738328576
```

Comparison against `runs/baseline-p3a-b4-fineweb-edu-100m`:

```text
| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-190327-exp5-rope-swiglu-b4 |
|---|---|---|
| best_val_loss | 4.7369 | 4.5939 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 22,017.7 |
| peak_vram_bytes | 7,524,103,680 | 7,738,328,576 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,860.3 |
```

Comparison against RoPE-only:

```text
| metric | 20260521-152220-exp1-rope-b4 | 20260521-190327-exp5-rope-swiglu-b4 |
|---|---|---|
| best_val_loss | 4.6088 | 4.5939 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 22,321.7 | 22,017.7 |
| peak_vram_bytes | 7,537,882,624 | 7,738,328,576 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,835.0 | 1,860.3 |
```

Derived deltas:

- Versus baseline validation loss: -0.143065 absolute, -3.02%.
- Versus baseline throughput: -8.62%.
- Versus baseline wall-clock: +9.43%.
- Versus baseline peak VRAM: +214,224,896 bytes, +2.85%.
- Versus RoPE-only validation loss: -0.014916 absolute, -0.32%.
- Versus RoPE-only throughput: -1.36%.
- Versus RoPE-only wall-clock: +1.38%.
- Versus RoPE-only peak VRAM: +200,445,952 bytes, +2.66%.

Validation curve:

```text
| step | baseline val_loss | RoPE val_loss | RoPE+SwiGLU val_loss | stack delta vs baseline | stack delta vs RoPE |
|---:|---:|---:|---:|---:|---:|
| 500 | 6.241714 | 6.026380 | 5.960014 | -0.281699 | -0.066365 |
| 1000 | 5.783716 | 5.578805 | 5.508558 | -0.275158 | -0.070247 |
| 1500 | 5.481412 | 5.317397 | 5.245773 | -0.235638 | -0.071624 |
| 2000 | 5.282323 | 5.140749 | 5.075481 | -0.206842 | -0.065267 |
| 2500 | 5.140076 | 5.009099 | 4.954935 | -0.185141 | -0.054164 |
| 3000 | 5.033428 | 4.899757 | 4.853854 | -0.179574 | -0.045902 |
| 3500 | 4.936568 | 4.811327 | 4.774987 | -0.161581 | -0.036340 |
| 4000 | 4.855029 | 4.734474 | 4.704973 | -0.150056 | -0.029500 |
| 4500 | 4.795664 | 4.670555 | 4.644305 | -0.151359 | -0.026251 |
| 5000 | 4.736919 | 4.608770 | 4.593853 | -0.143065 | -0.014916 |
```

## Interpretation

RoPE + SwiGLU is the best B4 quality result so far. It improves validation loss by -3.02% versus the B4 baseline and by -0.32% versus RoPE-only, clearing the decision rule by a narrow margin.

The interaction is constructive but not additive. SwiGLU alone improved validation loss by -0.037863 absolute; adding SwiGLU on top of RoPE improved by only -0.014916 absolute. The curve is better than RoPE-only at every checkpoint, but the advantage narrows over training.

Systems cost is acceptable for a leading stack candidate. The stack is -8.62% throughput versus baseline, which is slightly better than the predicted -9% to -11% range. Relative to RoPE-only, SwiGLU adds -1.36% throughput, +1.38% wall-clock, and +2.66% peak VRAM.

## Conclusion

Accept RoPE + parameter-matched SwiGLU as the current leading Phase 4 stack on B4. Do not make it a default yet.

Because the improvement over RoPE-only is real but smaller than the isolated SwiGLU result, the next stack experiment should be RoPE + QK-norm rather than immediately adding QK-norm to RoPE + SwiGLU. That will test the more direct attention-side interaction before expanding to three changes.

## Open questions

- Do RoPE and SwiGLU combine additively, or does one absorb most of the other's benefit?
- If the stack wins, is QK-norm still worth its additional systems cost?
- Does QK-norm have a stronger interaction with RoPE than SwiGLU does?
