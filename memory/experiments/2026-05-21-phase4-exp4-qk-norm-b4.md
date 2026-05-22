# Phase 4 Exp-4: QK-Norm on B4 FineWeb-Edu

**Date:** 2026-05-21
**Phase:** 4a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a/4a benchmark and modern-model changes

## Hypothesis

QK-norm should stabilize attention logits by normalizing each head's query and key vectors before the dot product. That can help optimization by making attention score scale less dependent on projection weight drift, especially in deeper or longer-context stacks. On this small 6-block B4 contract, the expected standalone gain is uncertain and probably smaller than RoPE or SwiGLU.

Prediction: B4 validation loss should be flat to modestly better than `baseline-p3a-b4-fineweb-edu-100m`, with a plausible range of +0.5% to -1.5%. Throughput is expected to be modestly worse because the implementation adds per-head RMSNorm operations to both query and key.

Decision rule: accept QK-norm as a Phase 4 candidate if it improves validation loss by at least ~0.3% without a large throughput regression. Treat a flat result as inconclusive; reject a clear validation-loss regression or a systems cost that is too large for the quality delta.

## Setup

Code delta:

- `ScratchGPTArchitecture.qk_norm` toggles per-head query/key RMSNorm on the modern path.
- The classic model path rejects `qk_norm=True` so the pedagogical `model.py` remains unchanged.
- QK-norm is applied after query/key projection and before RoPE. For this isolated run RoPE is off, so Exp-4 measures only query/key normalization against the B4 baseline.

Training delta from the B4 baseline:

```text
model_variant = modern
qk_norm = true
position_encoding = learned
normalization = layernorm
ffn_variant = mlp
```

Parameter count check:

```text
B4 baseline params: 49386577
B4 QK-norm params: 49391185
Delta:              +4608
```

Preflight:

```text
uv run pytest tests/test_config.py tests/test_model_factory.py -q
23 passed

uv run ruff check .
All checks passed!

uv run mypy scratchgpt
Success: no issues found in 22 source files

uv run python scripts/bench.py --block-size 16 --embedding-size 16 --num-heads 4 \
  --num-blocks 1 --batch-size 2 --vocab-size 32 --iters 2 --warmup 1 \
  --device cpu --model-variant modern --qk-norm
tokens_per_sec: 11270.044043378848
params: 4608
```

Benchmark command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug exp4-qk-norm-b4 \
  --arch-override model_variant=modern \
  --arch-override qk_norm=true \
  --device cuda
```

## Result

Run:

```text
runs/20260521-182751-exp4-qk-norm-b4
```

Summary:

```text
best_val_loss:       4.710294140569083
best_val_step:       5000
tokens_per_sec:      22573.967156540315
total_wallclock_sec: 1814.4794716835022
peak_vram_bytes:     7829612032
```

Comparison against `runs/baseline-p3a-b4-fineweb-edu-100m`:

```text
| metric | baseline-p3a-b4-fineweb-edu-100m | 20260521-182751-exp4-qk-norm-b4 |
|---|---|---|
| best_val_loss | 4.7369 | 4.7103 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 24,094.6 | 22,574.0 |
| peak_vram_bytes | 7,524,103,680 | 7,829,612,032 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 1,700.0 | 1,814.5 |
```

Derived deltas:

- Validation loss: -0.026625 absolute, -0.56%.
- Throughput: -6.31%.
- Wall-clock: +6.74%.
- Peak VRAM: +305,508,352 bytes, +4.06%.

Validation curve:

```text
| step | baseline val_loss | QK-norm val_loss | QK-norm delta |
|---:|---:|---:|---:|
| 500 | 6.241714 | 6.229054 | -0.012660 |
| 1000 | 5.783716 | 5.739838 | -0.043878 |
| 1500 | 5.481412 | 5.468538 | -0.012874 |
| 2000 | 5.282323 | 5.274496 | -0.007827 |
| 2500 | 5.140076 | 5.136424 | -0.003652 |
| 3000 | 5.033428 | 5.026730 | -0.006698 |
| 3500 | 4.936568 | 4.926826 | -0.009742 |
| 4000 | 4.855029 | 4.845434 | -0.009595 |
| 4500 | 4.795664 | 4.774520 | -0.021144 |
| 5000 | 4.736919 | 4.710294 | -0.026625 |
```

## Interpretation

QK-norm is a consistent quality win on B4. It beats the baseline at every validation checkpoint and finishes at -0.56% validation loss, clearing the acceptance threshold.

The systems cost is the main caveat. Throughput is -6.31%, wall-clock is +6.74%, and peak VRAM is +4.06%. That cost is materially larger than SwiGLU's cost while producing a smaller standalone validation-loss improvement. It is still cheaper than RoPE on throughput, but RoPE produced a much larger quality gain.

Because this run intentionally used learned positions, it does not answer the most likely interaction question: whether QK-norm helps more when paired with RoPE, where normalized query/key vectors directly feed the rotary transform.

## Conclusion

Accept QK-norm as a Phase 4 candidate for the modern path, but rank it behind RoPE and SwiGLU for stack testing. Do not make it a default.

Current isolated-candidate status on B4:

- RoPE: accepted candidate; strongest quality win so far.
- RMSNorm: not accepted standalone.
- SwiGLU: accepted candidate; moderate quality win with modest systems cost.
- QK-norm: accepted candidate; smaller quality win with non-trivial systems cost.

The next experiment should test the strongest low-risk stack: RoPE + parameter-matched SwiGLU on B4. QK-norm should wait for a second stack run unless the RoPE + SwiGLU result is unexpectedly weak.

## Open questions

- Does QK-norm stack better with RoPE than with learned positions?
- Would a shared q/k norm implementation reduce overhead without making the modern path less readable?
- Is QK-norm's cost acceptable only when it contributes an interaction gain with RoPE?
