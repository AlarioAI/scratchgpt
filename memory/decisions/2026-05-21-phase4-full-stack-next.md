# Phase 4 Full Stack Next Step

**Date:** 2026-05-21
**Context:** Phase 4 B4 ablations through Exp-7

## Decision

Treat RoPE + parameter-matched SwiGLU + QK-norm as the current leading B4 quality candidate.

Next, run a confirmatory B4 repeat of the same full-stack contract before expanding to B1/B2/B3. The Exp-7 gain over the previous best stack is large enough to care about (`-0.9848%` versus Exp-6), but a repeat is the cleanest way to separate a robust architecture improvement from single-seed/run variance.

## Evidence

Exp-7 (`runs/20260521-201522-exp7-rope-swiglu-qk-norm-b4`) achieved:

```text
best_val_loss:       4.535660772581914
tokens_per_sec:      20796.332197673906
total_wallclock_sec: 1969.57807803154
peak_vram_bytes:     8048031232
```

Versus the B4 baseline:

```text
best_val_loss:      -0.201258013619 (-4.2487%)
tokens_per_sec:     -3298.27905946  (-13.6889%)
total_wallclock_sec:+269.612904787  (+15.8599%)
peak_vram_bytes:    +523927552      (+6.9633%)
```

Versus Exp-6 RoPE + QK-norm:

```text
best_val_loss:      -0.0451099076172 (-0.9848%)
tokens_per_sec:     -297.110227236   (-1.4085%)
total_wallclock_sec:+27.7423560619   (+1.4287%)
peak_vram_bytes:    +206737408       (+2.6365%)
```

## Consequence

The full stack is now the quality leader, but it is not a free default. It costs `-13.69%` throughput versus the B4 baseline. If the repeat confirms the effect, the next decision should split quality-oriented recommendations from practical throughput-oriented recommendations, then run B1/B2/B3 transfer checks.
