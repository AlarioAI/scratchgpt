# B3 val_loss=0.28 is a char-level floor, not a hard target

**Date:** 2026-05-04
**Source:** Observed while writing `memory/experiments/2026-05-04-phase1-baselines.md`.

## Observation

B3 Chemistry reaches `best_val_loss = 0.2772` (perplexity ≈ 1.32) in 5000 steps. Compared to B1 (`val_loss=2.2261`, perplexity 9.26) and B2 (`val_loss=2.1694`, perplexity 8.75), B3's loss looks dramatically better — almost an order of magnitude lower.

This is misleading. The absolute loss number reflects a property of the dataset and tokenizer combination, not a property of the model's quality. Treating B3 as "the easy benchmark where improvements should be obvious" would be wrong.

## Evidence

- `runs/baseline-b3-chemistry/summary.json` — `best_val_loss: 0.2772, best_val_step: 5000`.
- `runs/baseline-b3-chemistry/config.yaml` — uses `CharTokenizer` with vocab size ~60 (exact size depends on characters present in the reaction text).
- Dataset key: `uspto-50k-49015-reactions`. The raw text is 49k reactions in SMILES notation like `CC(=O)O.CCO>>CC(=O)OCC.O` (esterification of acetic acid + ethanol), one per line.

## Mechanism

Three compounding reasons a char-level SMILES reaction corpus hits low cross-entropy quickly:

1. **Most of the product string is copied from the reactants.** A typical reaction transforms a small subset of atoms and leaves the rest untouched. A model that learns "most characters after `>>` are characters I've just seen before `>>`" scores very well on cross-entropy across the full reaction string, because the ratio of "easy character" (copy) to "hard character" (predict the transformation) is heavily skewed toward copy.

2. **Character-level SMILES has low local entropy.** After `C` you're very likely to see another `C`, `(`, `=`, or a digit. After `(` you almost always see a `=`, atom, or another `(`. The character-level bigram entropy of SMILES is probably 1–2 bits/char, while English character-level is 3–4 bits/char. A reasonable model should drive cross-entropy per token down close to that intrinsic entropy fairly quickly.

3. **60-character vocabulary.** The cross-entropy loss for a uniform random baseline is `log(60) ≈ 4.1 nats`. For B1 (50k vocab) it's `log(50257) ≈ 10.8 nats`. So B1 starts ~2.6× higher in absolute terms just from vocab size. In normalized terms (loss / log(V), a rough "bits-per-token normalized"), B1 is at 2.2261 / 10.82 ≈ 0.206 and B3 is at 0.2772 / 4.09 ≈ 0.068. Still lower for B3, but less dramatic than the raw numbers suggest.

## Implications

- **Phase 2+ deltas on B3 will be small in absolute terms but may be substantial in relative terms.** A change that reduces B3 val_loss from 0.28 to 0.25 is an 11% reduction — larger than the equivalent-percent change would be on B1 going from 2.23 to 1.99 (which would be a more visible absolute change but also 11%). We should report both absolute and % deltas in Phase 2+ experiment entries.

- **Don't use "B3 is easy" as a basis for skipping it.** B3 is where PR authors will be tempted to say "already-low val_loss, not much room to improve, skip the run." That's a mistake: B3 is our cleanest signal for *regression*. A change that makes B3 worse is almost certainly a real problem, precisely because the baseline is so tight. B3 is a sensitive detector.

- **Char-level tokenization produces a ceiling of its own.** The Phase 5 plan includes introducing a tokenizer that supports special tokens (`[BOS]`, `[EOS]`, etc.) for chemistry. Once that lands, the char-level floor becomes irrelevant for that benchmark, and we'll need to establish a new B3 baseline with the improved tokenizer. Pre-Phase 5 experiments on B3 are evaluating models operating on char-level SMILES, which is a different task than what a production chemistry model would do.

- **The `exp(val_loss)` perplexity framing is still useful** — B3's 1.32 perplexity means the model is almost certain about each next character, which is the correct reading. But perplexity-of-next-token on a corpus where next-token is *usually trivially predictable* is not the same signal as perplexity on harder text. Cross-benchmark perplexity comparison is category-confused; we won't do it in PR RESULTS tables.

- **If a Phase 2+ change shows no improvement on B3** (absolute delta within a few thousandths of zero), that's not automatically damning — it may just be hitting the floor. The benchmark's job is to confirm the change doesn't *hurt*. Wins will come from B1 and B2.
