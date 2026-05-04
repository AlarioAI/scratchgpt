# Checkpoints stay local; only lightweight artifacts commit

**Date:** 2026-05-04
**Phase:** 1

## Context

Every committed baseline run produces seven artifacts:

| artifact          | typical size |
|-------------------|--------------|
| summary.json      | 4 KB         |
| metrics.jsonl     | 8 KB         |
| env.json          | 4 KB         |
| config.yaml       | 4 KB         |
| checkpoints/best.pt | 50–210 MB  |
| checkpoints/last.pt | 50–210 MB  |
| (per-benchmark variable: legacy_ckpt/ empty dir, etc.) | — |

The Phase 1 baseline totals: B1 checkpoints 207 MB × 2, B2 90 MB × 2, B3 52 MB × 2. Roughly 670 MB of `.pt` blobs for three baselines, with more phases to come.

Question: do these go in git?

## Options considered

1. **Commit everything to git, including checkpoints.**

   Rationale: fully self-contained reference. Anyone who clones the repo can load the baseline and do inference without re-running.

   Rejected because: 670 MB per phase × many phases is repo-hostile. Git doesn't handle binary blobs gracefully — every clone pays the full history cost forever, even shallow clones aren't great because the commits exist. By Phase 5 the repo would be multi-GB.

2. **Git LFS for checkpoints.**

   Rationale: git-native-ish way to handle large blobs.

   Rejected because: adds a setup dependency for contributors, adds GitHub LFS quota costs, still tracks the blobs (just outside git's pack files), and doesn't actually solve the "do we need these" question. If we don't genuinely need them, LFS is complexity for nothing.

3. **External artifact storage (S3, HF Hub, etc.) with a download-script.**

   Rationale: keeps the repo lean, still makes checkpoints available.

   Deferred because: too early in the project to invest in artifact infrastructure. Phase 1's baseline checkpoints may not be worth preserving beyond a few weeks anyway — as soon as Phase 2 lands improvements, the Phase 2-baseline checkpoints will be the relevant reference. Worth revisiting at the end of Phase 2 or 3 if we find ourselves wanting old checkpoints.

4. **Gitignore the checkpoints; commit only the lightweight artifacts (chosen).**

   Rationale: the artifacts git actually needs to preserve are the *measurements*, not the weights. A run's summary.json + metrics.jsonl + env.json + config.yaml fully describe what happened. Weights are a byproduct. Anyone who wants the weights back can reproduce the run — that's what the determinism + fixed seed + committed config is *for*.

## Chosen option

Option 4. The `.gitignore` rule is:

```
runs/*
!runs/baseline-*/
!runs/baseline-*/summary.json
!runs/baseline-*/metrics.jsonl
!runs/baseline-*/env.json
!runs/baseline-*/config.yaml
!runs/BASELINES.md
```

This ignores everything under `runs/`, then re-includes the baseline directories themselves (so `git add runs/baseline-*/` works on the tracked files), then re-includes the four lightweight artifacts explicitly. Checkpoint blobs remain ignored and require `git add -f` to force-add (which we never want).

Note on the pattern mechanics: an earlier version used `!runs/baseline-*/**` to un-ignore everything in the baseline dir, which would have swept checkpoints into the staging area as soon as someone ran `git add runs/`. That was caught during Phase 1 implementation when `du -sh runs/baseline-*/` showed 670 MB pending. Fixed by making the un-ignore rules explicit per-filename.

## Expected consequences

- **The repo stays small.** Each phase's baseline adds ~60 KB of lightweight artifacts plus ~3 KB of BASELINES.md updates. Many phases can land without bloat.
- **Reproducing a baseline is a `uv run python benchmarks/b1_tinystories.py --slug <x>` call.** Takes 8–35 min of GPU time per benchmark. For anyone who wants to load a trained model, this is the canonical path.
- **Baselines are verifiable by reproduction, not by blob equality.** If someone clones the repo and reruns B1, they should get the same summary.json numbers (within torch's small but real cross-machine numerical variance). That's the guarantee we provide.

## How we'd know if this was wrong

- **If we find ourselves repeatedly re-running a specific baseline** to get weights back for ad-hoc inference / analysis / downstream work. Signal: weights *are* valuable artifacts in their own right, and the time cost of regenerating them is real. Response: set up option 3 (external storage).
- **If reproduction fails to match** (numerically significant differences in summary.json across machines). Signal: our determinism guarantee is weaker than we thought. Response: investigate; may require `torch.use_deterministic_algorithms(True)` and cudnn-deterministic mode, both of which have perf costs.
- **If a baseline becomes "special"** — e.g., takes 12 hours of careful tuning to produce and can't be casually regenerated. Signal: we're outside the Phase 1–4 regime where regeneration is cheap. Response: revisit artifact storage.
