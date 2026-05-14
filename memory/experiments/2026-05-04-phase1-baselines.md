# B1 / B2 / B3 baselines at the standard contract

**Date:** 2026-05-04
**Phase:** 1
**Branch / commit:** `feature/sota-upgrade-roadmap` @ `0c147b6`

## Hypothesis

Not a hypothesis-driven experiment — this run establishes the reference numbers against which every Phase 2+ change will be compared. The only prediction worth recording: with identical architecture and training budget across three domains, val_loss numbers will differ substantially because each dataset has a different intrinsic difficulty and tokenization granularity, so absolute cross-benchmark comparison is meaningless. Only same-dataset deltas will be scientifically valid.

## Setup

**Standard contract** (locked for all three runs and for every future Phase 1-baseline comparison):

| field             | value        |
|-------------------|--------------|
| max_steps         | 5000         |
| block_size        | 256          |
| batch_size        | 32           |
| learning_rate     | 3e-4         |
| random_seed       | 1337         |
| dropout_rate      | 0.1          |
| iteration_type    | chunking     |

**Architecture** (also locked, but not formally part of the contract — Phase 2+ will modify these):

| field             | value |
|-------------------|-------|
| embedding_size    | 384   |
| num_heads         | 6     |
| num_blocks        | 6     |

**Per-benchmark differences:**

- **B1 TinyStories:** `roneneldan/TinyStories` first 500,000 rows. Tokenizer: GPT-2 (50,257 vocab). `dataset_key = "tinystories-500000-rows"`.
- **B2 Chess:** Lichess `lichess_db_standard_rated_2016-02.pgn.zst`, first 50,000 parsed games. Tokenizer: `examples.chess_tokenizer.ChessTokenizer` (domain-specific, 12,341 vocab). `dataset_key = "lichess-lichess_db_standard_rated_2016-02.pgn-first-50000"`.
- **B3 Chemistry:** `pingzhili/uspto-50k`, 49,015 reactions after `>>` filter. Tokenizer: `CharTokenizer` (vocab derived from data, 47 chars). Note: no `[BOS]`/`[EOS]` wrapping because CharTokenizer would split those into 5 tokens of bracket-noise — Phase 5 will introduce a tokenizer that supports special tokens. `dataset_key = "uspto-50k-49015-reactions"`.

**Hardware:** single NVIDIA RTX A6000 (48 GiB), torch 2.8.0 + CUDA 12.8, Python 3.12, Ubuntu on `vai03`. See the `env.json` in each run dir for exact versions.

**Commands used:**

```bash
uv run python benchmarks/b1_tinystories.py --slug baseline-b1-tinystories --device cuda
uv run python benchmarks/b2_chess.py        --slug baseline-b2-chess        --device cuda
uv run python benchmarks/b3_chemistry.py    --slug baseline-b3-chemistry    --device cuda
# Then rename timestamped dirs to canonical baseline-* paths.
```

Each run was executed sequentially on the same GPU; no contention. B2's Lichess dataset had to be downloaded + parsed once (~2 min extra), cached to `~/.cache/scratchgpt/bench/chess-<hash>.txt` for future runs.

## Result

Run directories (all under `runs/` and committed for reference, minus the checkpoints):

- `runs/baseline-p1-b1-tinystories/summary.json`
- `runs/baseline-p1-b2-chess/summary.json`
- `runs/baseline-p1-b3-chemistry/summary.json`

`uv run python scripts/compare.py --allow-contract-mismatch runs/baseline-p1-*`:

```
| metric              | baseline-p1-b1-tinystories | baseline-p1-b2-chess | baseline-p1-b3-chemistry |
|---------------------|----------------------------|----------------------|--------------------------|
| best_val_loss       | 2.2261                     | 2.1694               | 0.2772                   |
| best_val_step       | 5,000                      | 5,000                | 5,000                    |
| tokens_per_sec      | 19,411.3                   | 83,015.7             | 86,728.4                 |
| peak_vram_bytes     | 7,224,604,160              | 3,146,654,208        | 1,906,259,968            |
| total_steps         | 5,000                      | 5,000                | 5,000                    |
| total_wallclock_sec | 2,110.1                    | 493.4                | 472.3                    |
```

The `--allow-contract-mismatch` flag was required because each run has a different `dataset_key`. This is the first and probably the only context where cross-dataset comparison is appropriate (documentation / overview); every Phase 2+ comparison will match on dataset_key and the flag won't be needed.

**Derived perplexities** (exp(val_loss), not stored in summary.json):

- B1: 9.264
- B2: 8.753
- B3: 1.319

**All three hit `best_val_step == 5000`**, i.e. loss was still improving at the final step. This is expected and fine — 5000 steps is a budget, not a convergence target. Phase 2+ compares *loss at the same step count*, not "convergence loss."

## Interpretation

The three val_loss numbers are not comparable to each other in absolute terms, and that's the whole point of locking `dataset_key` in the contract. Each number establishes its domain's reference. What *is* interesting, and what the harness surfaced immediately:

- **B1 runs at ~23% of the throughput of B2/B3 despite identical architecture.** This is because the output projection scales with vocab size and the cross-entropy softmax over 50k classes is expensive. See `memory/findings/2026-05-04-b1-vocab-throughput.md` for the detailed mechanism.
- **B3 reaches val_loss=0.28 in 5000 steps.** This looks impressive but is misleading: SMILES reaction strings are highly redundant at the character level (products inherit most characters from reactants, the same functional groups recur, syntax is rigid), so a character-level model can drive cross-entropy very low by learning local copying plus functional-group patterns. This is a floor effect, not a quality ceiling. See `memory/findings/2026-05-04-b3-charlevel-floor.md`.
- **Peak VRAM tracks the output-projection + activation picture.** B1's 7.2 GiB vs. B2/B3's 2–3 GiB is almost entirely the vocab_size × embedding_size = 50257 × 384 ≈ 19M extra parameters in the lm_head, plus their gradient and Adam state.
- **Wallclock follows directly from tokens/sec and step count.** 5000 × 32 × 256 = 40.96M tokens trained. 40.96M / 19.4k tokens/sec = 2111s = 35m 11s, which matches the 2110s wallclock within rounding.

All three baselines trained without error, with stable loss trajectories (see `metrics.jsonl` in each run dir), and produced artifacts identical in structure. The harness works as designed.

## Conclusion

Baselines accepted. Originally committed to `runs/baseline-*/`; later canonicalized to `runs/baseline-p1-*` when generation-2 baselines were added.

## Open questions

- **Would a different seed produce meaningfully different numbers?** We didn't run multi-seed baselines because seed-variance characterization isn't free (3 seeds × 3 benchmarks × 50 min ≈ 7 hours). For Phase 2 we'll run same-seed A/B and trust that seed-invariant improvements should show up. If a Phase 2 change produces a small effect (~1% val_loss reduction), we may need to quantify seed variance before trusting the delta. Tracking this as a deferred question.
- **Is B1's "still improving at step 5000" meaningful for Phase 2?** If a Phase 2 change converges earlier (e.g. reaches the baseline's step-5000 loss at step 3000), that's a real improvement even if the final step-5000 loss is the same. The harness logs every eval step, so this is detectable post-hoc, but we should watch for it.
- **Chess val_loss 2.17 is surprisingly close to TinyStories' 2.23** despite very different vocab sizes and domain structure. Is this a coincidence of this model size, or does it imply something about the relative entropy of the two domains at this budget? Worth revisiting if the same ordering survives Phase 2.
