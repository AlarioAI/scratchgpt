# Phase 3a benchmark refresh starts with a FineWeb-Edu sample

**Date:** 2026-05-21
**Phase:** 3a

## Context

Phase 2 improved the model under the original B1/B2/B3 suite, but that suite is not enough for the next architecture phase. TinyStories is narrow, chess reports only token loss, and chemistry is a character-level reaction corpus near a copy-heavy floor. Before building `model_modern.py`, Phase 3a needs at least one benchmark that resembles open small-LM pretraining data and can expose general-text quality gains.

The new benchmark still needs to respect the ScratchGPT measurement contract: fixed seed, fixed step budget, explicit `dataset_key`, committed canonical baselines only, and `compare.py` hard errors on mismatched contracts.

## Options considered

**FineWeb-Edu sample-10BT.** This was the initial target. FineWeb-Edu is a public educational web corpus filtered from FineWeb, with `sample-10BT` available as a smaller named config. Hugging Face's dataset card reports that FineWeb-Edu was built using an educational quality classifier and that `sample-10BT` is a randomly sampled subset. Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu. In this environment, however, `datasets` streaming over the raw FineWeb-Edu config successfully yielded rows but aborted during Python teardown, and non-streaming range loading was slow enough to be a bad benchmark default.

**codelion/fineweb-edu-100M.** This is the chosen first addition. It is a fixed 100M-token reservoir sample derived from FineWeb-Edu, with 115,482 rows and a reported 329 MB file size. The dataset card describes reservoir sampling as statistically unbiased and explicitly positions the sample for rapid experiments and ablations. Source: https://huggingface.co/datasets/codelion/fineweb-edu-100M.

**SmolLM-Corpus fineweb-edu-dedup.** This is also attractive because SmolLM work is directly small-model oriented and the dataset includes a deduplicated FineWeb-Edu subset. It is heavier to use as the first benchmark because the full subset is large and the repo needs a stable finite contract before adding mixture logic. Source: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus.

**BabyLM strict / strict-small.** BabyLM is useful for sample-efficient human-text work, especially if the project wants a paper-grade narrative around small budgets. It should come after B4 because BabyLM's real value includes its evaluation suite, not just next-token validation loss. Source for the current strict dataset candidate: https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict.

**RULER-style long-context tasks.** RULER is the right direction for RoPE/p-RoPE and local/global attention work, but it evaluates behavior after training rather than providing the next pretraining corpus. It also needs a generation/evaluation harness that ScratchGPT does not yet have. Source: https://github.com/NVIDIA/RULER and https://arxiv.org/abs/2404.06654.

## Chosen option

Add B4 FineWeb-Edu as the first Phase 3a benchmark:

```text
dataset_name = codelion/fineweb-edu-100M
dataset_config = none
subset_size = 100,000 usable documents
text_column = text
min_chars = 200
max_chars = 8192
tokenizer = gpt2
training contract = standard 5000-step contract
```

The benchmark loads the fixed sample, materializes a deterministic fixed prefix after filtering/truncation, and then routes through `HFDataSource` so validation splitting and run recording stay unchanged.

## Expected consequences

B4 should become the first comparison target for "real world" dense decoder improvements. It does not replace B1/B2/B3 immediately; it adds a stronger general-text axis while preserving continuity with the Phase 2 baselines.

Generation labels matter. The first canonical B4 baseline should be recorded as `baseline-p3a-b4-fineweb-edu-100m` rather than folded into P2, because this is a new benchmark generation even though the model defaults are still the accepted Phase 2 defaults.

The next benchmark-refresh slices should be:

1. Add chess legal-move-rate evaluation before treating B2 as domain-quality evidence.
2. Add chemistry validity/tokenization fixes before treating B3 as chemistry-quality evidence.
3. Add a BabyLM benchmark or evaluation harness for sample-efficient human text.
4. Add a RULER-style synthetic long-context eval before judging RoPE, p-RoPE, or local/global attention.

## How we'd know if this was wrong

Revisit this choice if B4 takes too long to materialize relative to its training time, if the fixed-prefix subset is too sensitive to source ordering, if GPT-2 tokenization makes B4 mostly another output-head throughput benchmark, or if B4 loss is too noisy at 5000 steps to distinguish the expected Phase 4 architecture deltas. The fallback would be a smaller fixed BabyLM or SmolLM-Corpus subset with a committed local manifest.
