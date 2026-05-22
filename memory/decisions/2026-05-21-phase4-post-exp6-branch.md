# Phase 4 branch after RoPE + QK-norm stack result

**Date:** 2026-05-21
**Phase:** 4a

## Context

Exp-6 tested RoPE + QK-norm after Exp-5 showed that RoPE + SwiGLU was constructive but not additive. RoPE + QK-norm is now the best B4 quality result:

```text
| run | best_val_loss | delta vs B4 baseline | tokens/sec |
|---|---:|---:|---:|
| B4 baseline | 4.736918786201336 | 0.00% | 24094.6 |
| RoPE-only | 4.608769744745419 | -2.71% | 22321.7 |
| RoPE + SwiGLU | 4.593853448846535 | -3.02% | 22017.7 |
| RoPE + QK-norm | 4.580770680199141 | -3.30% | 21093.4 |
```

The quality ordering now favors the attention-side pair, but the systems ordering does not. RoPE + QK-norm is -12.46% throughput versus baseline, compared with -8.62% for RoPE + SwiGLU.

## Options considered

Option A: stop at RoPE + QK-norm as the leading candidate. This keeps the best current quality result and avoids more expensive stack tests, but it leaves the obvious three-change interaction unmeasured.

Option B: run RoPE + SwiGLU + QK-norm next. This tests whether the feed-forward gain still exists on top of the stronger attention-side pair.

Option C: repeat RoPE + QK-norm before expanding. This would measure noise, but the current result already clears the branch rule and the open question is the triple interaction.

## Chosen option

Run RoPE + SwiGLU + QK-norm next on B4.

The triple stack should be accepted only if it beats RoPE + QK-norm by enough to justify the extra cost, or if it matches quality with unexpectedly better throughput. If the triple stack is flat or worse, keep RoPE + QK-norm as the best quality stack and RoPE + SwiGLU as the cheaper practical stack.

## Expected consequences

The next run will separate "best B4 quality" from "best quality/cost tradeoff." That distinction matters before any default change or broader benchmark sweep.

## How we'd know if this was wrong

If the triple stack is clearly worse, then the pairwise gains do not compose and future work should focus on repeatability and throughput profiling rather than adding more modern features. If the triple stack wins only marginally at much worse throughput, QK-norm may remain a paper result rather than a practical default.
