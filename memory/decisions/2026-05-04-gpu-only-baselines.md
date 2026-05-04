# Committed baselines must be GPU-trained at the full contract

**Date:** 2026-05-04
**Phase:** 1

## Context

The Phase 1 plan was originally drafted on a laptop without a GPU. A tempting shortcut: reduce `STANDARD_STEPS` from 5000 to 1000 when running baselines on weak hardware, so the plan's "commit the three baselines" task could be executed anywhere.

This temptation is the whole reason the benchmark_contract exists. If we committed "baselines" at 1000 steps and someone with a real GPU later ran variants at the default 5000 steps, `compare.py` would silently produce meaningless deltas — unless the contract catches it.

And the contract *does* catch it: `max_steps` is a contract field, so comparing 1000-step and 5000-step runs hard-errors. But that's a band-aid. The deeper question is whether we should be committing reference numbers we can't reproduce on the same hardware as everyone else.

## Options considered

1. **Commit baselines at whatever budget the original runner had.**

   Pro: maximally accessible — anyone can produce the baselines.

   Con: the baselines themselves become hardware-dependent. A "baseline" at 1000 steps on a laptop CPU tells you nothing about what 1000 steps on a GPU would look like, and both are useless for comparing against 5000-step variants. The contract would prevent silent cross-comparison, but the baselines would still be ~useless as reference numbers.

2. **Require baselines to be trained on a specific reference GPU model.**

   Pro: maximally reproducible.

   Con: locks the project to a specific hardware generation. As A6000s age out, baselines would go stale without a clear successor policy. Also elitist — contributors without the reference hardware couldn't contribute baseline runs.

3. **Baselines must be GPU-trained at the full contract; CPU users keep their own reference numbers locally (chosen).**

   Pro: the committed baselines are real training budget numbers, not scaled-down approximations. Reproduction requires compute, not specific hardware.

   Con: a contributor without a GPU can't validate a baseline change. Mitigation: baselines change rarely — once per phase at most — and coordinating with a GPU-equipped collaborator for that one PR is tractable.

## Chosen option

Option 3. Specifically:

- `STANDARD_STEPS` in `benchmarks/_shared.py` is locked at **5000**. The value is not a CLI arg. Changing it requires editing code and would be caught in PR review.
- If you don't have a GPU: run benchmarks locally at whatever reduced budget you choose, compare your own local baselines to your own local variants. **Do not commit "baselines" at reduced budgets.** The committed baselines under `runs/baseline-*/` are the immutable reference for the project.
- Hardware differences across GPUs are acceptable — we expect A6000 and H100 to produce numerically-close-but-not-identical results. What we commit is one specific hardware snapshot documented in `env.json`. If cross-hardware variance becomes a real issue, we'll add a reproducibility-variance appendix.

## Expected consequences

- **CI doesn't run full baselines.** It can't — 50 minutes of GPU time per push is infeasible. CI runs unit tests + lint + type-check only. Baseline regeneration is a human-in-the-loop event.
- **Baselines are updated rarely.** Each phase that legitimately changes a contract field produces a new baseline generation. We'll probably have "generation 1" baselines until Phase 3 ships LR scheduling (which might want to change the contract's effective budget interpretation).
- **The plan documentation explicitly forbids** committing baselines at reduced budgets. This is written into `docs/superpowers/plans/2026-05-03-phase1-harness.md` (the "Budget protocol" section at the top) and into `runs/BASELINES.md`.

## How we'd know if this was wrong

- **If contributors regularly need to reproduce baselines** and don't have access to a GPU. Signal: we're gatekeeping a common workflow on expensive hardware. Response: publish the baseline checkpoints somewhere (see the external-storage option in `memory/decisions/2026-05-04-checkpoint-exclusion.md`).
- **If cross-machine numerical variance is too large** to meaningfully compare a contributor's local run to the committed baseline. Signal: our "reproduction means matching numbers within noise" guarantee is broken. Response: require deterministic CUDA ops, possibly dropping to a fixed GPU generation, or publishing variance bands.
- **If GPU time becomes a bottleneck** even for the project's own maintainers. Signal: the 5-hour-per-phase compute budget is too high for the iteration cadence we want. Response: introduce a smaller "dev contract" (maybe 1000 steps) that's used for quick sanity-checking during development, while full 5000-step runs gate merges.
