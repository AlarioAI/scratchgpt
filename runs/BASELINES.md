# Phase 1 Baselines

These runs are the reference for all subsequent phases. Any phase PR must run B1, B2, B3 with its changes and use `scripts/compare.py` to show deltas against the *matching* baseline (same-task compare — never mix datasets).

## Benchmark contract

Every baseline here was trained at:

| field            | value                      |
|------------------|----------------------------|
| max_steps        | 5000                       |
| block_size       | 256                        |
| batch_size       | 32                         |
| learning_rate    | 3e-4                       |
| random_seed      | 1337                       |
| dropout_rate     | 0.1                        |
| iteration_type   | chunking                   |

Architecture (also locked across baselines): `embedding_size=384, num_heads=6, num_blocks=6`. Phase 2+ will modify architecture — those experiments compare against these baselines via `scripts/compare.py`, which hard-errors on contract mismatch unless `--allow-contract-mismatch` is passed. That flag is reserved for comparisons where the contract delta *is* the experiment (e.g. testing a different learning rate).

## Hardware

See `runs/baseline-*/env.json` for exact versions. These runs were produced on:

- GPU: NVIDIA RTX A6000
- PyTorch: 2.8.0 + CUDA 12.8
- Python: 3.12

## The three benchmarks

- **B1 — TinyStories**. Dataset: `roneneldan/TinyStories`, first 500k rows. Tokenizer: GPT-2 (50,257 vocab). `dataset_key = tinystories-500000-rows`.
- **B2 — Chess**. Dataset: Lichess standard-rated 2016-02, first 50k games, parsed + cleaned. Tokenizer: `ChessTokenizer` (domain-specific). `dataset_key = lichess-lichess_db_standard_rated_2016-02.pgn-first-50000`.
- **B3 — Chemistry**. Dataset: `pingzhili/uspto-50k`, 49,015 valid reactions (raw SMILES, no special tokens). Tokenizer: `CharTokenizer`. `dataset_key = uspto-50k-49015-reactions`.

## Current numbers

Output of `scripts/compare.py --allow-contract-mismatch runs/baseline-*`:

```
| metric              | baseline-b1-tinystories | baseline-b2-chess | baseline-b3-chemistry |
|---------------------|-------------------------|-------------------|-----------------------|
| best_val_loss       | 2.2261                  | 2.1694            | 0.2772                |
| best_val_step       | 5,000                   | 5,000             | 5,000                 |
| tokens_per_sec      | 19,411.3                | 83,015.7          | 86,728.4              |
| peak_vram_bytes     | 7,224,604,160           | 3,146,654,208     | 1,906,259,968         |
| total_steps         | 5,000                   | 5,000             | 5,000                 |
| total_wallclock_sec | 2,110.1                 | 493.4             | 472.3                 |
```

## Observations

- **B1** is ~4x slower per token than B2/B3 because the GPT-2 vocab (50k) makes the output projection dominate the compute at this model size. B2/B3 use tiny vocabs (~200 chess moves, ~60 chars), so the lm_head is trivial.
- **B1** best_val_loss=2.23 at 5k steps corresponds to perplexity ~9.3 on GPT-2-tokenized TinyStories. Room to improve; this is the headroom Phase 2+ will be measured against.
- **B3** best_val_loss=0.28 is very low because SMILES reactions are highly repetitive at the character level — same functional groups, same atoms — and 5k steps is enough to memorize large chunks. This loss floor will be harder to push down; smaller deltas expected in Phase 2+.
- **B2** best_val_loss=2.17 over a ~200-token chess vocab is a more honest signal than raw loss suggests: the chess tokenizer is closer to word-level than character-level.
- Peak VRAM scales with `batch_size * block_size * embedding_size + vocab_size * embedding_size` — B1's larger vocab pushes it to 7GB vs. 2–3GB for B2/B3.
