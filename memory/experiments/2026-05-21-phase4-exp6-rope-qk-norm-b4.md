# Phase 4 Exp-6: RoPE + QK-Norm on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

QK-norm should have its most plausible interaction with RoPE, because it normalizes the per-head query/key vectors immediately before RoPE rotates them and before attention logits are formed. Exp-4 showed QK-norm was a standalone B4 win with learned positions, but its gain was smaller and more expensive than SwiGLU. Exp-6 tests whether QK-norm becomes more valuable when paired with RoPE.

Prediction: RoPE + QK-norm should beat RoPE-only validation loss (`4.608769744745419`). A strong result would beat or match the current RoPE + SwiGLU stack (`4.593853448846535`), which would justify testing the full RoPE + SwiGLU + QK-norm stack. Throughput should be slower than RoPE-only and probably slower than RoPE + SwiGLU, because QK-norm adds per-head reductions in the attention path.

Decision rule: if RoPE + QK-norm beats RoPE + SwiGLU on validation loss, test the full triple stack next. If it beats RoPE-only but loses to RoPE + SwiGLU, keep QK-norm as a lower-priority candidate and do not run the triple stack yet unless the quality gap is small enough to justify the systems cost. If it fails to beat RoPE-only, deprioritize QK-norm for B4.

## Setup

Training delta from the B4 baseline:

```text
model_variant = modern
position_encoding = rope
qk_norm = true
normalization = layernorm
ffn_variant = mlp
```

SwiGLU and RMSNorm are intentionally off, so Exp-6 isolates the RoPE + QK-norm interaction.

Parameter count check:

```text
B4 baseline params:     49386577
B4 RoPE+QK-norm params: 49292881
Delta:                  -93696
```

The stack has fewer parameters than baseline because RoPE removes the learned position table, while QK-norm adds only per-head normalization weights.

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
  --device cpu --model-variant modern --position-encoding rope --qk-norm
tokens_per_sec: 7323.596818857681
params: 4352
```

Benchmark command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp6-rope-qk-norm-b4 \
  --arch-override model_variant=modern \
  --arch-override position_encoding=rope \
  --arch-override qk_norm=true \
  --device cuda
```

## Result

Run:

```text
runs/20260521-193831-exp6-rope-qk-norm-b4
```

Summary:

```text
best_val_loss:       4.580770680199141
best_val_step:       5000
tokens_per_sec:      21093.44242491031
total_wallclock_sec: 1941.8357219696045
peak_vram_bytes:     7841293824
```

Comparison against `runs/baseline-p3a-b4-fineweb-edu-100m`:

```text
| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-193831-exp6-rope-qk-norm-b4 |
|---|---|---|
| best_val_loss | 4.7369 | 4.5808 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 21,093.4 |
| peak_vram_bytes | 7,524,103,680 | 7,841,293,824 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,941.8 |
```

Comparison against RoPE-only:

```text
| metric | 20260521-152220-exp1-rope-b4 | 20260521-193831-exp6-rope-qk-norm-b4 |
|---|---|---|
| best_val_loss | 4.6088 | 4.5808 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 22,321.7 | 21,093.4 |
| peak_vram_bytes | 7,537,882,624 | 7,841,293,824 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,835.0 | 1,941.8 |
```

Comparison against RoPE + SwiGLU:

```text
| metric | 20260521-190327-exp5-rope-swiglu-b4 | 20260521-193831-exp6-rope-qk-norm-b4 |
|---|---|---|
| best_val_loss | 4.5939 | 4.5808 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 22,017.7 | 21,093.4 |
| peak_vram_bytes | 7,738,328,576 | 7,841,293,824 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,860.3 | 1,941.8 |
```

Derived deltas:

- Versus baseline validation loss: -0.156148 absolute, -3.30%.
- Versus baseline throughput: -12.46%.
- Versus baseline wall-clock: +14.23%.
- Versus baseline peak VRAM: +317,190,144 bytes, +4.22%.
- Versus RoPE-only validation loss: -0.027999 absolute, -0.61%.
- Versus RoPE-only throughput: -5.50%.
- Versus RoPE + SwiGLU validation loss: -0.013083 absolute, -0.28%.
- Versus RoPE + SwiGLU throughput: -4.20%.

Validation curve:

```text
| step | baseline | RoPE | RoPE+SwiGLU | RoPE+QK-norm | delta vs baseline | delta vs RoPE | delta vs RoPE+SwiGLU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6.241714 | 6.026380 | 5.960014 | 6.025470 | -0.216244 | -0.000910 | +0.065455 |
| 1000 | 5.783716 | 5.578805 | 5.508558 | 5.580864 | -0.202852 | +0.002059 | +0.072305 |
| 1500 | 5.481412 | 5.317397 | 5.245773 | 5.313764 | -0.167648 | -0.003634 | +0.067991 |
| 2000 | 5.282323 | 5.140749 | 5.075481 | 5.132126 | -0.150197 | -0.008623 | +0.056645 |
| 2500 | 5.140076 | 5.009099 | 4.954935 | 4.998044 | -0.142031 | -0.011055 | +0.043109 |
| 3000 | 5.033428 | 4.899757 | 4.853854 | 4.882284 | -0.151144 | -0.017472 | +0.028430 |
| 3500 | 4.936568 | 4.811327 | 4.774987 | 4.786827 | -0.149741 | -0.024500 | +0.011840 |
| 4000 | 4.855029 | 4.734474 | 4.704973 | 4.704740 | -0.150289 | -0.029734 | -0.000233 |
| 4500 | 4.795664 | 4.670555 | 4.644305 | 4.641883 | -0.153781 | -0.028672 | -0.002422 |
| 5000 | 4.736919 | 4.608770 | 4.593853 | 4.580771 | -0.156148 | -0.027999 | -0.013083 |
```

## Interpretation

RoPE + QK-norm is the best B4 quality result so far. It beats the baseline by -3.30%, RoPE-only by -0.61%, and RoPE + SwiGLU by -0.28%.

The learning curve is unusual. RoPE + QK-norm starts behind RoPE + SwiGLU and only catches it at step 4000, then pulls ahead by step 5000. Against RoPE-only, it is essentially tied at step 500, slightly worse at step 1000, and then increasingly better from step 1500 onward. That suggests QK-norm may improve later optimization or stability rather than early sample efficiency.

The systems cost is substantial. Throughput is -12.46% versus baseline, -5.50% versus RoPE-only, and -4.20% versus RoPE + SwiGLU. This is expensive enough that QK-norm should not become a default without either a full-stack win, a longer-run confirmation, or a clear broader-benchmark result.

## Conclusion

Accept RoPE + QK-norm as the leading B4 quality stack so far, but mark the cost as a major caveat.

Because RoPE + QK-norm beats RoPE + SwiGLU, the next experiment should test the full RoPE + SwiGLU + QK-norm stack. The key question is whether SwiGLU still adds useful feed-forward-side gain on top of the stronger RoPE + QK-norm attention stack, or whether the current RoPE + QK-norm result is already the best quality/cost tradeoff.

## Open questions

- Is QK-norm's B4 value mostly an attention-side interaction with RoPE?
- Does SwiGLU add anything on top of RoPE + QK-norm, or are the gains mostly overlapping by this point?
- Would QK-norm look better at a longer context length than the B4 `block_size=256` contract?
