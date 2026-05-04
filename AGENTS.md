# AGENTS.md

Guidance for coding agents working in this repo. Keep it short, specific, and current.

## Project in one paragraph

ScratchGPT is a deliberately small, readable decoder-only transformer in ~150 lines. The value proposition is *pedagogical*: a learner should be able to read the core `scratchgpt/model/model.py` end-to-end and understand every line. Any change that obscures the core model loses the thing the project is trading on. Modern techniques (SDPA, RoPE, SwiGLU, etc.) are welcome but belong behind config flags or in a parallel `model_modern.py`, never by rewriting the original in place.

## Ground rules

- **Readability of the core is a hard constraint.** If a proposed change makes `model/model.py` materially harder to read, either gate it behind a flag or move it to a parallel module. Reviewing agents should push back on PRs that trade core clarity for a small accuracy lift.
- **Measure, don't assert.** Every change that could affect quality or speed must come with a B1/B2/B3 run and a `scripts/compare.py` delta. "Should be a win" without numbers is not acceptable.
- **Benchmark contract is load-bearing.** See `docs/superpowers/plans/2026-05-03-phase1-harness.md` for the full protocol. Short version: comparing runs with mismatched contracts is a hard error, not a warning. Use `--allow-contract-mismatch` only when the deviation is the *point* of the experiment, and call it out in the PR.
- **Tests first.** The project uses pytest, mypy strict, and ruff. Everything new lands with tests.
- **No narration-style comments.** Don't write comments that restate what the code does. Only write a comment when the *why* is non-obvious.

## Commands you'll want

```bash
uv sync --all-groups                 # install everything including dev tools
uv run pytest -q                     # full test suite (fast)
uv run ruff check --fix .            # lint + autofix
uv run mypy scratchgpt               # strict typing on the package
uv run python benchmarks/b1_tinystories.py --slug my-experiment   # a benchmark
uv run python scripts/compare.py runs/baseline-b1-tinystories runs/*-my-experiment
uv run python scripts/bench.py --device cpu                       # pure throughput
```

## Layout

```
scratchgpt/              # library, imported by examples and benchmarks
  model/model.py           # the 150-line core — treat as precious
  config.py                # Pydantic settings; env-var overridable
  training/                # trainer, determinism, run recorder
  data/                    # DataSource protocol + HFDataSource
  tokenizer/               # CharTokenizer, HuggingFaceTokenizer
examples/                # pedagogical use cases (chess, chemistry, etc.)
benchmarks/              # Phase 1 reproducible harness
scripts/                 # bench.py, compare.py
runs/                    # .gitignored except runs/baseline-*/
tests/
docs/superpowers/        # LOCAL planning notes — GITIGNORED, never commit
```

## What not to do

- Don't add heavy new dependencies. The pitch is "simple to read." A 500MB dependency to shave 2% off val loss is a bad trade.
- Don't "clean up" `examples/` by deduplicating them. The duplication is intentional — each example is meant to be copy-pasteable and readable in isolation.
- Don't delete or fold `_train_by_epochs`. It's the back-compat fallback for users with old configs that don't set `max_steps`.
- Don't commit anything under `docs/superpowers/` or `runs/` (other than `runs/baseline-*/`).
- Don't mix emoji in code or in commit subject lines unless the user explicitly asks. Existing prints use them; new ones don't need them.

## Where to find the plan

The SOTA upgrade roadmap and phase plans live in `docs/superpowers/plans/`. That directory is gitignored — each collaborator keeps their own working copies. The current state of work is tracked there, not in this file.

## When in doubt

Ask. The maintainers prefer a clarifying question over a PR that re-architects something intentional.
