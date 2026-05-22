# Use a parallel modern model path for Phase 4 ablations

**Date:** 2026-05-21
**Phase:** 3a / 4a

## Context

The next research track tests RoPE, RMSNorm, parameter-matched SwiGLU, QK-norm, and later dense stacks. These are real architecture changes, but adding all of them to `scratchgpt/model/model.py` would erode the project's pedagogical value. The core model is intentionally small and readable; Phase 4 needs a research path without turning the educational path into a flag matrix.

## Options considered

**Rewrite `model.py` directly.** Rejected. It would make the central teaching artifact harder to read and violates the Phase 1/2 ground rule that modern machinery belongs behind flags or in a parallel module.

**Add individual flags to `model.py`.** Rejected as the default approach. A single flag was acceptable for Phase 2 tier-1 changes, but RoPE, RMSNorm, SwiGLU, QK-norm, and stack interactions would make the core implementation branchy.

**Create a parallel model path selected by config.** Chosen. `ScratchGPTArchitecture.model_variant` defaults to `"classic"` and can be set to `"modern"` for research ablations. The benchmark scripts already support `--arch-override model_variant=modern`, so experiments can use the modern path without editing files between runs.

## Chosen option

Add `scratchgpt/model/model_modern.py` and `scratchgpt/model/factory.py`. The modern model path initially delegates to the classic implementation and has exact forward-pass parity under fixed seed. Future modern components land in `model_modern.py`, not in the pedagogical `model.py`.

## Expected consequences

Phase 4 ablations will compare runs that differ by architecture implementation fields, while the benchmark contract remains the same. B4 `baseline-p3a-b4-fineweb-edu-100m` remains the default comparison target for the first modern decoder ablations.

The first real model-quality experiment should be RoPE in the modern path. Before running it, keep the exact-parity tests passing so any measured change comes from RoPE rather than plumbing drift.

## How we'd know if this was wrong

Revisit this if the modern path duplicates so much code that it becomes harder to maintain than a small, well-contained flag in the classic model, or if parity breaks in a way that makes architecture deltas impossible to attribute. Until then, preserving `model.py` readability is the stronger constraint.
