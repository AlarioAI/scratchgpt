# Phase 4 branch after RoPE + SwiGLU stack result

**Date:** 2026-05-21
**Phase:** 4a

## Context

Exp-5 tested RoPE + parameter-matched SwiGLU on B4 after the isolated ablations selected RoPE, SwiGLU, and QK-norm as modern-path candidates. The result is the best B4 validation loss so far:

```text
| run | best_val_loss | delta vs B4 baseline | delta vs RoPE |
|---|---:|---:|---:|
| B4 baseline | 4.736918786201336 | 0.00% | +2.78% |
| RoPE-only | 4.608769744745419 | -2.71% | 0.00% |
| RoPE + SwiGLU | 4.593853448846535 | -3.02% | -0.32% |
```

The stack is constructive but not additive. SwiGLU's isolated absolute gain was -0.037863, but its gain on top of RoPE was -0.014916.

## Options considered

Option A: run RoPE + SwiGLU + QK-norm next. This would test the best available full stack quickly, but it would hide whether QK-norm actually interacts with RoPE or just rides along with the existing stack.

Option B: run RoPE + QK-norm next. This directly tests the attention-side interaction that QK-norm was hypothesized to have with RoPE.

Option C: stop stack testing and repeat RoPE + SwiGLU. That would quantify noise but would not answer whether QK-norm belongs in the modern block.

## Chosen option

Run RoPE + QK-norm next on B4, keeping SwiGLU off.

After that result, compare:

```text
RoPE-only
RoPE + SwiGLU
RoPE + QK-norm
```

If RoPE + QK-norm beats RoPE + SwiGLU or gives a comparable gain with an interpretable attention-side mechanism, then run the full RoPE + SwiGLU + QK-norm stack. If RoPE + QK-norm is weaker than RoPE + SwiGLU, the full stack is lower priority and should wait until we decide whether the extra QK-norm systems cost is worth testing.

## Expected consequences

This keeps the stack search interpretable. It also protects us from accepting a three-change result before we know which two-change interaction is doing the work.

## How we'd know if this was wrong

If RoPE + QK-norm is clearly worse than RoPE-only, we should have gone straight to either a repeat of RoPE + SwiGLU or a broader benchmark check. If the full triple stack later produces a large gain despite weak RoPE + QK-norm, then QK-norm's benefit depends on SwiGLU or another higher-order interaction.
