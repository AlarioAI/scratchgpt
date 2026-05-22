# Phase 4 stack-testing order after isolated B4 ablations

**Date:** 2026-05-21
**Phase:** 4a

## Context

Phase 4a is testing modern decoder-block changes against the B4 FineWeb-Edu benchmark before changing defaults. The first four isolated ablations are complete:

```text
| ablation | B4 val-loss delta | throughput delta | decision |
|---|---:|---:|---|
| RoPE | -2.71% | -7.36% | accepted candidate |
| RMSNorm | +0.15% | -3.15% | not accepted standalone |
| SwiGLU | -0.80% | -2.50% | accepted candidate |
| QK-norm | -0.56% | -6.31% | accepted candidate |
```

All comparisons use the same B4 contract and the same canonical baseline: `runs/baseline-p3a-b4-fineweb-edu-100m`.

## Options considered

Option A: stack every accepted candidate at once. This would give a fast answer for a single full modern block, but it would hide whether the next gain came from RoPE, SwiGLU, QK-norm, or their interactions.

Option B: test the strongest cheap pair first: RoPE + parameter-matched SwiGLU. This combines the largest isolated quality win with the candidate that has the cleanest quality/cost tradeoff.

Option C: test RoPE + QK-norm first. This targets the most plausible interaction, because QK-norm directly changes the query/key vectors that RoPE rotates. The downside is that QK-norm had a smaller standalone win and a larger systems cost than SwiGLU.

Option D: include RMSNorm in the next stack despite its standalone regression. This keeps the stack closer to common modern LLM blocks, but the B4 evidence says RMSNorm should not get priority until a stronger interaction hypothesis exists.

## Chosen option

Run RoPE + parameter-matched SwiGLU next on B4.

If that stack gives an additive or near-additive gain, run RoPE + SwiGLU + QK-norm afterward. If RoPE + SwiGLU underperforms relative to the isolated results, run RoPE + QK-norm before expanding the stack further.

RMSNorm remains a later interaction-only candidate, not part of the next stack.

## Expected consequences

The next result should answer whether the two strongest practical candidates combine constructively. It also keeps the number of active variables small enough that the outcome will be interpretable in a future paper.

This decision does not make any modern feature a default. Defaults should wait for stack results and at least one repeat or broader benchmark check.

## How we'd know if this was wrong

If RoPE + SwiGLU produces a much smaller gain than expected, the pair may have overlapping mechanisms or unfavorable optimization dynamics. In that case, QK-norm should be tested with RoPE before assuming SwiGLU belongs in the full modern stack.

If RoPE + SwiGLU produces a strong gain but a much larger systems regression than predicted from isolated runs, later work should prioritize throughput profiling before adding QK-norm.
