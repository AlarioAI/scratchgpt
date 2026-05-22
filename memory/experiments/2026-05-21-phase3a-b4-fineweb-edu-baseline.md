# Phase 3a B4 FineWeb-Edu baseline

**Date:** 2026-05-21
**Phase:** 3a
**Branch / commit:** `75ae5aa` plus uncommitted Phase 3a benchmark-refresh changes

## Hypothesis

This is a baseline-establishing run, not a model-change experiment. The expected result is a stable generation-3a reference for general-text language modeling on a FineWeb-Edu-derived corpus. B4 should be harder and more representative than B1 TinyStories because it contains broader web text rather than synthetic children's stories, while still being small enough for fast ablations on the available RTX A6000s.

Because B4 uses the GPT-2 tokenizer and a 50,257-token vocabulary, throughput should look more like B1 than B2/B3. The quality number should not be compared to B1/B2/B3 directly; the benchmark dataset changed, so B4 starts a new comparison axis.

## Setup

B4 uses `codelion/fineweb-edu-100M`, a fixed 100M-token reservoir sample derived from FineWeb-Edu. The benchmark materializes a deterministic fixed prefix after filtering and truncation:

```text
dataset_name = codelion/fineweb-edu-100M
split = train
subset_size = 100,000 usable documents
text_column = text
min_chars = 200
max_chars = 8192
tokenizer = gpt2
dataset_key = codelion-fineweb-edu-100m-first-100000-docs-min-200-max-8192-chars
```

The training contract is the standard 5000-step contract: block size 256, batch size 32, learning rate 3e-4, seed 1337, dropout 0.1, and chunking iteration. Architecture uses the accepted Phase 2 defaults: `attention_scale_mode="head"`, `ffn_activation="gelu"`, `init_scheme="gpt2"`, `tie_weights=false`, and `use_bias=true`.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/b4_fineweb_edu.py \
  --slug baseline-p3a-b4-fineweb-edu-100m \
  --device cuda
```

## Result

Run directory canonicalized to `runs/baseline-p3a-b4-fineweb-edu-100m/`.

Output of:

```bash
uv run python scripts/compare.py runs/baseline-p3a-b4-fineweb-edu-100m
```

| metric | baseline-p3a-b4-fineweb-edu-100m |
|---|---|
| best_val_loss | 4.7369 |
| best_val_step | 5,000 |
| tokens_per_sec | 24,094.6 |
| peak_vram_bytes | 7,524,103,680 |
| total_steps | 5,000 |
| total_wallclock_sec | 1,700.0 |

Validation curve:

| step | val_loss |
|------|----------|
| 500 | 6.2417 |
| 1,000 | 5.7837 |
| 1,500 | 5.4814 |
| 2,000 | 5.2823 |
| 2,500 | 5.1401 |
| 3,000 | 5.0334 |
| 3,500 | 4.9366 |
| 4,000 | 4.8550 |
| 4,500 | 4.7957 |
| 5,000 | 4.7369 |

Hardware and environment: single NVIDIA RTX A6000, torch 2.8.0 + CUDA 12.8, Python 3.12.7. The trainer-recorded wall time was 28m20s. Dataset materialization/tokenization added roughly another minute before the trainer timer started.

## Interpretation

B4 is much harder than B1, as expected. The absolute loss is not comparable across benchmarks because B4 uses a different corpus, but the number is credible as a general-text baseline for this small 6-layer GPT-2-tokenized model. The validation curve is smooth and still improving at step 5000, so B4 can detect architecture changes under the current budget but may also reward changes that improve early optimization rather than final convergence.

Throughput landed between the Phase 1 A6000 B1 baseline and the Phase 2 5090 B1 baseline, which is consistent with the hypothesis that GPT-2-vocab output projection still dominates this benchmark. B4 is evaluation-heavy: each 500-step interval includes a full validation pass over the 20% split, making the wall time about 28 minutes even though raw training runs around 6.8 steps/sec.

## Conclusion

B4 is accepted as the first generation-3a benchmark baseline. Use `runs/baseline-p3a-b4-fineweb-edu-100m/` as the comparison target for the first dense decoder ablations.

The next experiment should be a `model_modern.py` parity/plumbing step only if the new path can reproduce current logits or at least current B4 loss under equivalent architecture. After parity, run isolated ablations in this order: RoPE, RMSNorm, parameter-matched SwiGLU, and QK-norm.

## Open questions

The 5000-step curve has not flattened. If an ablation wins at step 5000, a longer follow-up may be needed to distinguish better optimization from better asymptotic quality. The current contract is still acceptable for first-pass screening because the same early-training budget is applied to every candidate.
