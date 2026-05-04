# What the benchmark_contract covers and why

**Date:** 2026-05-04
**Phase:** 1

## Context

`scripts/compare.py` refuses to compare runs whose `benchmark_contract` fields disagree — hard-errors with exit code 2 unless `--allow-contract-mismatch` is passed. The question is: which fields belong in the contract?

The trade-off is between under-constraining (allowing meaningless comparisons to go through silently) and over-constraining (making the harness useless for its own purpose, since Phase 2+ is *supposed* to change things).

## Options considered

1. **Budget-only** (`max_steps`, `block_size`, `batch_size`, `learning_rate`, `random_seed`).

   Rationale: these are the knobs that determine "how much compute / data the model saw." Architecture changes are what we're testing, so they shouldn't be constrained.

   Rejected because: dropout_rate and iteration_type (chunking vs. sliding) also materially affect loss numbers without being architecture — a PR that silently bumps dropout from 0.1 to 0.2 would pass compare.py with a mystery drop in loss. Similarly for dataset changes: someone re-running with 200k TinyStories rows instead of 500k would get different numbers for reasons unrelated to the change under test.

2. **Everything-in-config**, including architecture fields.

   Rationale: maximally strict; zero risk of silent contamination.

   Rejected because: the harness would refuse every Phase 2+ comparison, since architecture changes are exactly what we're testing. `--allow-contract-mismatch` would become the routine invocation, defeating its purpose as a "make intentional deviations visible" gate.

3. **Budget + optimization + dataset** (chosen).

   Fields: `max_steps`, `block_size`, `batch_size`, `learning_rate`, `random_seed`, `dropout_rate`, `iteration_type`, `dataset_key`.

   Architecture knobs (`embedding_size`, `num_heads`, `num_blocks`, activation choice, normalization choice, etc.) are deliberately excluded.

## Chosen option

Option 3. The contract covers anything that would make a comparison numerically misleading if changed silently, minus the things Phase 2+ is explicitly exploring.

The `dataset_key` field is interesting: it's not a config parameter, it's a per-benchmark string (`"tinystories-500000-rows"`, etc.) stamped onto the Trainer via `trainer._dataset_key = ...` by `benchmarks/_shared.run_benchmark()`. It captures "which data subset did this model see" in a way a config field alone can't — e.g., if B1 is rerun with `--subset-size 200000`, the key becomes `tinystories-200000-rows` and compare.py will reject the comparison.

## Expected consequences

- **Phase 2 ablations on architecture will work transparently.** Weight tying, GELU, attention-scale fix, etc. can all be compared without `--allow-contract-mismatch`.
- **The `--allow-contract-mismatch` flag becomes a red flag.** Any use of it in a comparison is a signal that the reader needs to pay attention to what was changed and why. It should appear in PRs rarely, and when it does, the accompanying RESULTS.md should explain the intentional deviation.
- **Changing a contract field mid-project is a real event.** If Phase 3 legitimately needs to change, say, `dropout_rate` (e.g., "we want to show a longer schedule tolerates higher dropout"), that experiment will require the flag, and the baselines effectively get a generation marker. Old baselines vs. new variants would need a separate cross-generation experiment to translate.

## How we'd know if this was wrong

- **Symptom of under-constraint**: a Phase 2+ PR ships a "win" that later turns out to be caused by accidentally running on a dataset subset, a different seed, or changed dropout. The harness would have been silent on it. We'd see this when attempting to reproduce a win and getting different numbers.
- **Symptom of over-constraint**: `--allow-contract-mismatch` becomes routine. Every comparison PR uses it. That'd mean the contract is blocking legitimate comparisons. Cue: if we see ourselves writing "I had to pass --allow-contract-mismatch because X, which isn't actually a real difference between runs," the contract needs revision.

Initial consideration suggested locking only 5 fields (max_steps, block_size, batch_size, learning_rate, random_seed). External review correctly flagged that `dropout_rate`, `iteration_type`, and `dataset_key` also determine results. Updated before shipping. Documented here so future-us knows the contract's scope is already a considered judgment, not an oversight.
