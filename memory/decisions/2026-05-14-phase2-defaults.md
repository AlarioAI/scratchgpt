# Phase 2 default architecture choices

**Date:** 2026-05-14
**Phase:** 2

## Context

Phase 2 introduced architecture flags for tier-1 transformer improvements and measured each training-affecting change on B1 TinyStories, B2 Chess, and B3 Chemistry under the standard 5000-step benchmark contract. Defaults can now diverge from the Phase 1 baseline, so this decision records the final accepted architecture state that generation-2 baselines and future model work should use.

## Chosen defaults

| field | default | decision basis |
|-------|---------|----------------|
| `attention_scale_mode` | `"head"` | Exp-1 accepted; textbook attention scaling improved B1/B3 and left B2 flat. |
| `ffn_activation` | `"gelu"` | Exp-2 accepted; strong B1/B3 gains, B2 flat, manageable VRAM cost. |
| `tie_weights` | `false` | Exp-3 rejected standalone tying; Exp-6 was not clean enough to promote tying. |
| `init_scheme` | `"gpt2"` | Exp-4 accepted; largest Phase 2 win on all three benchmarks. |
| `use_bias` | `true` | Exp-5 rejected no-bias; small but consistent regression. |

The clean `tie_weights=true` + `init_scheme="gpt2"` ablation is deferred. It should not be treated as pending Phase 2 work, and it should not alter generation-2 baselines. If it is revisited later, it needs a fresh hypothesis-first experiment entry and a default decision based on isolated results.

## Reproduction flags

The rejected or superseded settings remain available for reproducibility and future ablations:

- `attention_scale_mode="embedding"` reproduces the Phase 1 scaling bug.
- `ffn_activation="relu"` reproduces the Phase 1 FFN activation.
- `init_scheme="default"` reproduces PyTorch initialization.
- `tie_weights=true` remains available only as an opt-in experiment.
- `use_bias=false` remains available only as an opt-in experiment.

## Experiment summary

| experiment | tested state | B1 delta | B2 delta | B3 delta | decision |
|------------|--------------|----------|----------|----------|----------|
| Exp-1 | `attention_scale_mode="head"` | -0.27% | +0.01% | -1.84% | default on |
| Exp-2 | `ffn_activation="gelu"` on top of Exp-1 | -2.83% cumulative | +0.08% cumulative | -3.25% cumulative | default on |
| Exp-3 | `tie_weights=true` with default init | +51.97% | +35.63% | +28.11% | rejected |
| Exp-4 | `init_scheme="gpt2"` on top of accepted defaults | -11.27% cumulative | -6.29% cumulative | -9.45% cumulative | default on |
| Exp-5 | `use_bias=false` on top of accepted defaults | +0.60% marginal | +0.19% marginal | +0.28% marginal | rejected |
| Exp-6 | full stack: tied weights, GPT-2 init, no bias | -0.55% marginal | -1.53% marginal | +0.40% marginal | rejected as default |

The final accepted default state is Exp-4, not Exp-6.

## Expected consequences

Generation-2 canonical baselines should be trained with the final accepted default state:

```text
attention_scale_mode = "head"
ffn_activation = "gelu"
init_scheme = "gpt2"
tie_weights = false
use_bias = true
```

Future architecture experiments should compare against those generation-2 baselines unless the benchmark contract itself changes. If the benchmark suite is refreshed for model-first research, that refresh needs its own decision entry and a new baseline generation label.

## How we'd know if this was wrong

The accepted defaults should be revisited if a refreshed benchmark suite shows a consistent regression from one of the promoted flags, if multi-seed variance shows that a shipped improvement was within noise, or if a later isolated ablation demonstrates that a rejected flag becomes a net win under a clearly specified companion change. Until then, these defaults are the Phase 2 outcome.
