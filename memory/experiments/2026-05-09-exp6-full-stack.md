# Exp-6: Full Phase 2 stack with tied weights

**Date:** 2026-05-09
**Phase:** 2
**Branch / commit:** `feature/sota-upgrade-roadmap` @ `106ceae` (post-Exp-5 memory commit; defaults still post-Exp-4). Runs produced at this SHA.

## Hypothesis

GPT-2 initialization should rescue the catastrophic standalone `tie_weights=true` failure from Exp-3 because it removes the embedding-vs-linear initialization mismatch that made the tied output projection unusably large at step 0. Prediction before running: no Exp-3-style blowup; B1 is the most likely beneficiary because tying regularizes the 50k-vocab embedding/output pair, while B2 and B3 should be close to flat or mildly worse. Exp-5 showed `use_bias=false` is a small drag, so an Exp-6 win would be stronger evidence for the tied-weights-plus-GPT-2-init interaction than the raw full-stack number suggests. The test remains confounded because all five flags are on.

## Setup

Full Phase 2 stack:

- `attention_scale_mode=head`
- `ffn_activation=gelu`
- `tie_weights=true`
- `init_scheme=gpt2`
- `use_bias=false`

Commands:

```bash
uv run python benchmarks/b1_tinystories.py \
    --slug exp6-stack-b1 \
    --arch-override attention_scale_mode=head \
    --arch-override ffn_activation=gelu \
    --arch-override tie_weights=true \
    --arch-override init_scheme=gpt2 \
    --arch-override use_bias=false \
    --device cuda

TMPDIR=/home/ayeganov/tmp/scratchgpt uv run python benchmarks/b2_chess.py \
    --slug exp6-stack-b2 \
    --arch-override attention_scale_mode=head \
    --arch-override ffn_activation=gelu \
    --arch-override tie_weights=true \
    --arch-override init_scheme=gpt2 \
    --arch-override use_bias=false \
    --device cuda

TMPDIR=/home/ayeganov/tmp/scratchgpt uv run python benchmarks/b3_chemistry.py \
    --slug exp6-stack-b3 \
    --arch-override attention_scale_mode=head \
    --arch-override ffn_activation=gelu \
    --arch-override tie_weights=true \
    --arch-override init_scheme=gpt2 \
    --arch-override use_bias=false \
    --device cuda
```

This run used the collaborator-approved single-GPU fallback: one RTX 5090, serial B1 -> B2 -> B3. The original Phase 2 command plan was a two-GPU A6000 pipeline, so throughput and wallclock are recorded for provenance only and are not interpreted as architectural speed results. The standard benchmark contract stayed unchanged on all three runs (`max_steps=5000`, `block_size=256`, `batch_size=32`, `learning_rate=3e-4`, `random_seed=1337`, `dropout_rate=0.1`, `iteration_type="chunking"`, and the same per-benchmark `dataset_key`). `compare.py` accepted all comparisons without `--allow-contract-mismatch`.

Run dirs (ephemeral):

- `runs/20260509-182624-exp6-stack-b1/`
- `runs/20260509-184056-exp6-stack-b2/`
- `runs/20260509-184551-exp6-stack-b3/`

## Result

### Cumulative (vs Phase 1 baseline)

| benchmark | baseline val_loss | exp6 val_loss | delta val_loss | delta % |
|-----------|-------------------|---------------|----------------|---------|
| B1 TinyStories | 2.2261 | 1.9644 | -0.2617 | -11.76% |
| B2 Chess       | 2.1694 | 2.0018 | -0.1676 | -7.73% |
| B3 Chemistry   | 0.2772 | 0.2520 | -0.0252 | -9.09% |

### Marginal (vs pre-Exp-6 defaults, i.e. Exp-4)

Exp-4 committed results: B1 1.9752, B2 2.0329, B3 0.2510. Exp-5 was rejected, so the defaults before Exp-6 remained the Exp-4 stack.

| benchmark | pre-Exp-6 | exp6 val_loss | delta val_loss | delta % |
|-----------|-----------|---------------|----------------|---------|
| B1 TinyStories | 1.9752 | 1.9644 | -0.0108 | -0.55% |
| B2 Chess       | 2.0329 | 2.0018 | -0.0311 | -1.53% |
| B3 Chemistry   | 0.2510 | 0.2520 | +0.0010 | +0.40% |

### compare.py output

B1 TinyStories:

| metric | baseline-b1-tinystories | 20260509-182624-exp6-stack-b1 |
|---|---|---|
| best_val_loss | 2.2261 | 1.9644 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 19,411.3 | 47,644.9 |
| peak_vram_bytes | 7,224,604,160 | 7,291,027,968 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 2,110.1 | 859.6942 |

B2 Chess:

| metric | baseline-b2-chess | 20260509-184056-exp6-stack-b2 |
|---|---|---|
| best_val_loss | 2.1694 | 2.0018 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 83,015.7 | 144,511.0 |
| peak_vram_bytes | 3,146,654,208 | 3,390,746,624 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 493.4007 | 283.4385 |

B3 Chemistry:

| metric | baseline-b3-chemistry | 20260509-184551-exp6-stack-b3 |
|---|---|---|
| best_val_loss | 0.2772 | 0.2520 |
| best_val_step | 5,000 | 5,000 |
| tokens_per_sec | 86,728.4 | 150,015.4 |
| peak_vram_bytes | 1,906,259,968 | 2,172,544,000 |
| total_steps | 5,000 | 5,000 |
| total_wallclock_sec | 472.2792 | 273.0386 |

## Interpretation

Exp-6 rescues the standalone weight-tying failure from Exp-3. The full stack beats the accepted Exp-4 defaults on B1 and B2, with B2 showing the clearest marginal improvement (-1.53%). B3 regresses slightly (+0.40%) and lands below the Exp-4 cumulative improvement, so the result is not a clean across-the-board default flip.

The important interaction evidence is that `tie_weights=true` no longer blows up once paired with GPT-2 initialization. Exp-3's proposed mechanism was that PyTorch's default embedding scale was incompatible with using the same tensor as the output projection. GPT-2 init puts the tied tensor on a scale that works for both roles, and the Exp-6 results are consistent with that theory.

The result is stronger than the raw full-stack comparison because Exp-5 showed `use_bias=false` is a small consistent drag: +0.60% on B1, +0.19% on B2, and +0.28% on B3. Exp-6 wins B1/B2 despite carrying that drag. However, attribution is still confounded because Exp-6 changed both `tie_weights` and `use_bias` relative to the current accepted defaults.

## Conclusion

Do **not** flip the full Exp-6 stack as the default. Keep `use_bias=true`, keep `tie_weights=false` for now, and retain the accepted defaults from Exp-4.

The next scientifically clean move is a focused two-flag ablation: `tie_weights=true` plus `init_scheme=gpt2`, with the other accepted defaults unchanged. That run would isolate whether tied weights under compatible initialization should be promoted independently of the rejected no-bias flag.

## Open questions

- Is `tie_weights=true` with GPT-2 init a net win when `use_bias=true` stays enabled?
- Is the B3 regression caused by the no-bias component, tied weights, or the combination?
- Does the B2 gain reflect a real tied-weights benefit for chess notation, or a single-seed fluctuation?
- Should the clean tie+GPT-2-init ablation be treated as a Phase 2 addendum before the retrospective, or deferred to Phase 3?
