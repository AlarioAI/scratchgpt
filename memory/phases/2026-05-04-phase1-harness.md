# Phase 1 retrospective: measurement harness

**Phase dates:** 2026-05-03 to 2026-05-04
**Commit range:** `4b9424f..0c147b6` on branch `feature/sota-upgrade-roadmap`

## Goal recap

Before making any SOTA changes to ScratchGPT's transformer, build the substrate that makes every future change measurable. The roadmap stated the principle bluntly: no change ships without a before/after A/B run against a committed baseline. Phase 1 was the infrastructure required to make that rule enforceable, not aspirational.

A secondary goal was to adopt the Karpathy-style "autoresearch" pattern (deterministic, fixed-budget, machine-readable metrics, one-command compare) as a *pattern* from day one, so that when we actually plug in an LLM-driven experiment agent in a later phase, the loop is already ready.

## What shipped

Eleven commits, each TDD'd with pytest and reviewed by spec + code-quality subagent reviewers before landing. The phase produced:

- **A reproducibility layer.** `scratchgpt/training/determinism.py` with `seed_everything()` seeding torch / numpy / stdlib / CUDA, and `seed_worker()` for DataLoader worker determinism. The existing trainer was already deterministic at fixed seed on single-worker mode; the seed_worker addition is insurance against regressions introduced by any future move to `num_workers > 0`.
- **A recording layer.** `scratchgpt/training/run_recorder.py` (`RunRecorder` class) writes one timestamped directory per run containing `env.json` (torch version, CUDA version, GPU name, git sha + dirty bit, Python version, platform), `config.yaml` (the resolved Pydantic config), `metrics.jsonl` (one line per eval step, streaming), `summary.json` (best val loss + step, total steps, throughput, peak VRAM, and the benchmark_contract), and a `checkpoints/` subdirectory with `best.pt` and `last.pt`.
- **A step-based trainer.** `scratchgpt/training/trainer.py` was rewritten: primary path is step-based with `log_every_steps` and `eval_every_steps` cadences; epoch-based path is preserved as a fallback when `max_steps is None`, for backwards compatibility with pre-existing configs. The trainer emits the benchmark_contract into summary.json at finalize time. A `_training_step` helper (added after code review) collapses the zero_grad / forward / backward / step sequence so both loop modes share it.
- **New config fields** in `ScratchGPTTraining`: `max_steps`, `eval_every_steps`, `log_every_steps`, `warmup_steps`. The last one is reserved for Phase 3 LR scheduling and was landed here so Phase 2 configs stay forward-compatible.
- **An HFDataSource overhaul.** `__init__` now accepts an optional pre-loaded `dataset` kwarg (bypasses `load_dataset`), and `from_hf_dataset(...)` is a thin classmethod factory that routes through `__init__`. This was driven by a code review observation — the earlier design used `__new__` to monkey-patch in a dataset, which would silently break if `__init__` grew new attributes.
- **Two CLIs.** `scripts/bench.py` reports forward/backward ms, tokens/sec, peak VRAM, and param count for any model config, with correct CUDA synchronization around each timed section. `scripts/compare.py` loads summary.json files from N run directories and prints a markdown delta table, **hard-erroring on benchmark_contract mismatch** unless `--allow-contract-mismatch` is passed. The contract covers 8 fields: max_steps, block_size, batch_size, learning_rate, random_seed, dropout_rate, iteration_type, dataset_key.
- **Three benchmark scripts.** `benchmarks/b1_tinystories.py`, `b2_chess.py`, `b3_chemistry.py`, all wired through `benchmarks/_shared.py` which provides `build_standard_config()` (locks the architecture and training hyperparams for the 5000-step contract), `run_benchmark()` (seeds, builds model + optimizer + recorder + trainer, stamps the per-benchmark dataset_key onto the trainer via `trainer._dataset_key`, and runs), and `cached_chess_corpus()` (caches the ~1GB Lichess .zst parse output under `~/.cache/scratchgpt/bench/` keyed by url+max_games).
- **Three committed baselines** at `runs/baseline-{b1-tinystories,b2-chess,b3-chemistry}/`. Each directory contains the lightweight artifacts (summary.json, metrics.jsonl, env.json, config.yaml); checkpoint blobs are gitignored because a 207MB .pt file per baseline per phase would balloon the repo.

Five structural decisions that will shape every subsequent phase are captured in `memory/decisions/`.

## What we learned

The harness did two useful things on its first real run:

1. **It exposed that B1 is throughput-bound by the output projection.** Same architecture as B2 and B3, but B1 runs at 19k tokens/sec vs. 83–87k for the others. The culprit is the 50,257-element GPT-2 vocabulary: the final linear projection from 384 → 50,257 and the cross-entropy softmax over that vocab dominate the compute. This was only visible because we explicitly benchmarked all three domains with identical architecture. See `memory/findings/2026-05-04-b1-vocab-throughput.md`.

2. **It exposed that B3 is on a floor.** val_loss=0.28 at 5000 steps looks impressive until you remember that SMILES reactions at the character level are extremely repetitive. Most characters in a product SMILES are directly copied from a reactant SMILES. Phase 2+ will show small absolute deltas on B3; we'll want to watch % loss reduction, not absolute. See `memory/findings/2026-05-04-b3-charlevel-floor.md`.

Both observations are exactly the kind of non-obvious signal the harness is designed to surface. Neither would be visible from reading the code.

A third observation, more procedural: **the benchmark_contract enforcement paid for itself within an hour of shipping.** When we ran `compare.py` across the three baselines to write BASELINES.md, it immediately hard-errored on `dataset_key` mismatch — which is exactly right, you can't compare loss numbers across datasets. We passed `--allow-contract-mismatch` deliberately because in that context we *wanted* the side-by-side for documentation purposes. The flag works exactly as intended: making deliberate deviations visible.

A fourth, about AI tooling: **the subagent-driven-development pattern worked well for mechanical tasks.** Sonnet handled TDD cycles on well-specified tasks cleanly. The one place it slipped was adding a `Co-Authored-By: Claude` trailer on the first commit (before AGENTS.md existed) — fixed by amending and then by adding an explicit anti-attribution rule to AGENTS.md. Subsequent subagent commits were clean.

## Numbers

Three baselines, all at the standard contract (`max_steps=5000, block_size=256, batch_size=32, lr=3e-4, seed=1337, dropout=0.1`, chunking iteration, `embedding_size=384, num_heads=6, num_blocks=6`). Hardware: single NVIDIA RTX A6000, torch 2.8.0 + CUDA 12.8, Python 3.12.

| benchmark                | best_val_loss | val_ppl* | tokens/sec | peak VRAM | wall time |
|--------------------------|---------------|----------|------------|-----------|-----------|
| B1 TinyStories (500k rows, gpt2 tok) | 2.2261 | 9.26  | 19,411  | 7.22 GiB  | 35m 10s |
| B2 Chess (Lichess 50k games, ChessTokenizer) | 2.1694 | 8.75 | 83,016 | 3.13 GiB | 8m 13s |
| B3 Chemistry (USPTO-50k, CharTokenizer) | 0.2772 | 1.32 | 86,728 | 1.91 GiB | 7m 52s |

*perplexity = exp(val_loss), not recorded directly by the harness but a useful readability number.

## What's different in the next phase because of this

Phase 2 (Tier-1 "free lunch": fix attention-scale bug, GELU, weight tying, init, bias=False, sampling) is now a pure A/B exercise. For each change:

1. Land the change behind a config flag (no default behavior change until the whole tier lands).
2. Run B1 + B2 + B3 at the same contract.
3. `compare.py` against the baseline (same dataset_key; contract must match).
4. Write a `memory/experiments/` entry with hypothesis, result, interpretation.

Each variant should cost ~30 minutes of B1 + 8 + 8 ≈ 50 minutes wall. Six Tier-1 changes → ~5 hours of compute total. That's feasible in a single work day.

A specific prediction for Phase 2 to falsify: **weight tying (sharing the token embedding matrix with the lm_head) should have outsized throughput impact on B1** because it directly removes parameters from the component we now know dominates B1's compute. B2 and B3 should see the normal small-quality improvement weight tying produces but negligible throughput change.

## Outstanding debt

- **`examples/gpt2.py` is gone.** The file existed on one earlier machine but was never committed; when we moved to the GPU machine via `git clone`, it didn't come along. The task plan still has a step to "fix a kwarg in examples/gpt2.py" that's moot. Not blocking, just confusing if someone reads the plan literally.
- **The `legacy_ckpt/` subdirectory** that `run_benchmark()` creates inside every run dir is empty in step-mode (the epoch fallback uses it; step mode writes only to `checkpoints/`). Harmless but noisy. Can remove the `experiment_path` arg plumbing in step mode or just leave it — low priority.
- **The `warmup_steps` field in config is not consumed anywhere yet.** It was landed here for forward-compat with Phase 3's LR scheduler. Any Phase 2 run that sets a non-zero warmup will get no warmup behavior — trainer silently ignores it. That's fine for now; Phase 3 plan will note this.
- **Memory entries were written retroactively** for Phase 1. Going forward, entries are written *during* the work, not after. This retrospective itself is the exception.
