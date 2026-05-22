# Canonical Baselines

These runs are the committed references for ScratchGPT experiments. Compare only within the same benchmark generation and dataset unless the comparison is explicitly an overview table using `--allow-contract-mismatch`.

## Benchmark contract

Every canonical baseline here was trained at:

| field            | value    |
|------------------|----------|
| max_steps        | 5000     |
| block_size       | 256      |
| batch_size       | 32       |
| learning_rate    | 3e-4     |
| random_seed      | 1337     |
| dropout_rate     | 0.1      |
| iteration_type   | chunking |

Shared architecture size: `embedding_size=384`, `num_heads=6`, `num_blocks=6`.

## Generations

**Generation 1 (`baseline-p1-*`)** is the Phase 1 harness baseline before model upgrades.

**Generation 2 (`baseline-p2-*`)** is the post-Phase-2 baseline and should be the default comparison target for model-first work after 2026-05-14. Its accepted architecture defaults are:

| field                | value  |
|----------------------|--------|
| attention_scale_mode | head   |
| ffn_activation       | gelu   |
| init_scheme          | gpt2   |
| tie_weights          | false  |
| use_bias             | true   |

**Generation 3a (`baseline-p3a-*`)** adds the first refreshed benchmark for model-first work. It keeps the accepted Phase 2 model defaults and adds B4 FineWeb-Edu 100M as a stronger general-text benchmark.

## Benchmarks

- **B1 TinyStories:** `roneneldan/TinyStories`, first 500,000 rows. Tokenizer: GPT-2, vocab 50,257. `dataset_key = tinystories-500000-rows`.
- **B2 Chess:** Lichess standard-rated 2016-02, first 50,000 parsed games. Tokenizer: `ChessTokenizer`, vocab 12,341. `dataset_key = lichess-lichess_db_standard_rated_2016-02.pgn-first-50000`.
- **B3 Chemistry:** `pingzhili/uspto-50k`, 49,015 valid reactions after `>>` filtering. Tokenizer: `CharTokenizer`, vocab 47. `dataset_key = uspto-50k-49015-reactions`.
- **B4 FineWeb-Edu 100M:** `codelion/fineweb-edu-100M`, first 100,000 usable documents after `min_chars=200` and `max_chars=8192`. Tokenizer: GPT-2, vocab 50,257. `dataset_key = codelion-fineweb-edu-100m-first-100000-docs-min-200-max-8192-chars`.

## Hardware

See each run's `env.json` for exact versions.

| generation | GPU | PyTorch | CUDA | Python |
|------------|-----|---------|------|--------|
| P1 | NVIDIA RTX A6000 | 2.8.0 | 12.8 | 3.12.7 |
| P2 | NVIDIA GeForce RTX 5090 | 2.8.0 | 12.8 | 3.12.9 |
| P3a | NVIDIA RTX A6000 | 2.8.0 | 12.8 | 3.12.7 |

## Generation 1 numbers

Output of:

```bash
uv run python scripts/compare.py --allow-contract-mismatch \
  runs/baseline-p1-b1-tinystories \
  runs/baseline-p1-b2-chess \
  runs/baseline-p1-b3-chemistry
```

| metric | baseline-p1-b1-tinystories | baseline-p1-b2-chess | baseline-p1-b3-chemistry |
|---|---|---|---|
| best_val_loss | 2.2261 | 2.1694 | 0.2772 |
| best_val_step | 5,000 | 5,000 | 5,000 |
| tokens_per_sec | 19,411.3 | 83,015.7 | 86,728.4 |
| peak_vram_bytes | 7,224,604,160 | 3,146,654,208 | 1,906,259,968 |
| total_steps | 5,000 | 5,000 | 5,000 |
| total_wallclock_sec | 2,110.1 | 493.4007 | 472.2792 |

## Generation 2 numbers

Output of:

```bash
uv run python scripts/compare.py --allow-contract-mismatch \
  runs/baseline-p2-b1-tinystories \
  runs/baseline-p2-b2-chess \
  runs/baseline-p2-b3-chemistry
```

| metric | baseline-p2-b1-tinystories | baseline-p2-b2-chess | baseline-p2-b3-chemistry |
|---|---|---|---|
| best_val_loss | 1.9778 | 2.0188 | 0.2515 |
| best_val_step | 5,000 | 5,000 | 5,000 |
| tokens_per_sec | 46,745.2 | 142,864.8 | 141,625.0 |
| peak_vram_bytes | 7,524,103,680 | 3,448,250,880 | 2,196,191,744 |
| total_steps | 5,000 | 5,000 | 5,000 |
| total_wallclock_sec | 876.2405 | 286.7046 | 289.2146 |

## P1 to P2 deltas

Same-dataset comparisons:

| benchmark | P1 loss | P2 loss | delta | delta % |
|-----------|---------|---------|-------|---------|
| B1 TinyStories | 2.2261 | 1.9778 | -0.2483 | -11.15% |
| B2 Chess | 2.1694 | 2.0188 | -0.1505 | -6.94% |
| B3 Chemistry | 0.2772 | 0.2515 | -0.0257 | -9.27% |

The throughput numbers are not a clean algorithmic comparison because P1 and P2 were run on different GPUs. For algorithmic claims, use loss deltas within the same benchmark generation and record the run hardware in `memory/experiments/`.

## Generation 3a numbers

Output of:

```bash
uv run python scripts/compare.py runs/baseline-p3a-b4-fineweb-edu-100m
```

| metric | baseline-p3a-b4-fineweb-edu-100m |
|---|---|
| best_val_loss | 4.7369 |
| best_val_step | 5,000 |
| tokens_per_sec | 24,094.6 |
| peak_vram_bytes | 7,524,103,680 |
| total_steps | 5,000 |
| total_wallclock_sec | 1,700.0 |

B4 starts a new benchmark axis. Do not compare its loss directly to B1/B2/B3; use it as the same-dataset reference for Phase 4 dense decoder ablations.
