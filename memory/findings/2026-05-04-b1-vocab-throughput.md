# B1 runs at ~23% of B2/B3 throughput because the GPT-2 lm_head dominates compute

**Date:** 2026-05-04
**Source:** Observed while writing `memory/experiments/2026-05-04-phase1-baselines.md`.

## Observation

All three Phase 1 baselines use identical architecture: `embedding_size=384, num_heads=6, num_blocks=6, block_size=256, batch_size=32`. They differ only in dataset and tokenizer. Measured training throughput on a single RTX A6000:

| benchmark | tokenizer (vocab size) | tokens/sec |
|-----------|------------------------|------------|
| B1 TinyStories | GPT-2 (50,257)      | 19,411     |
| B2 Chess       | ChessTokenizer (~200) | 83,016   |
| B3 Chemistry   | CharTokenizer (~60)  | 86,728    |

B1 is ~4.3× slower per token than B2 and B3, despite identical transformer-block compute.

## Evidence

- `runs/baseline-b1-tinystories/summary.json` — `tokens_per_sec: 19411.3`
- `runs/baseline-b2-chess/summary.json` — `tokens_per_sec: 83015.7`
- `runs/baseline-b3-chemistry/summary.json` — `tokens_per_sec: 86728.4`
- B1 peak VRAM 7.22 GiB vs. B2 3.13 GiB vs. B3 1.91 GiB (`peak_vram_bytes` in each summary).

The VRAM delta between B1 and B2 is ~4.1 GiB. B1 has 50257 × 384 × 4 bytes = 77 MiB more parameters in the lm_head compared to a hypothetical ~200-token-vocab version, *plus* the gradient (another 77 MiB) *plus* AdamW's first and second moment buffers (another 2 × 77 = 154 MiB) = ~308 MiB of extra state. The remaining ~3.8 GiB difference is activation memory, dominated by the `(B, T, V) = (32, 256, 50257)` logits tensor at ~1.6 GiB per copy in fp32 (and multiple copies exist in the autograd graph during backward).

## Mechanism

Two compounding effects:

1. **Output projection FLOPs.** The final `nn.Linear(embedding_size, vocab_size)` does `B*T*E*V` multiply-adds per forward pass. For B1 that's `32 * 256 * 384 * 50257 ≈ 1.58 × 10^11` FLOPs per step, dwarfing a single attention or FFN block (each ~`B*T*E*E*8 ≈ 9.6 × 10^8` FLOPs for attention, ~`B*T*E*E*8 ≈ 9.6 × 10^8` for the FFN expansion layers). Even with 6 blocks, the transformer body is ~`6 * 10^10` FLOPs, i.e. smaller than a single pass through the lm_head at B1's vocab size.

2. **Softmax + cross-entropy over the same `(B*T, V)` logits tensor.** Softmax touches every one of 32 * 256 * 50257 = 411M logits per step. For B2/B3 with ~200/60 tokens, the same operation is 1.6M / 0.49M — essentially free.

The lm_head is a shallow but extremely wide operation that a six-layer transformer can't amortize away.

## Implications

- **Weight tying is a particularly attractive Phase 2 target for B1.** Standard weight-tying shares the weight matrix between the input token embedding and the output projection. It removes 50257 × 384 ≈ 19M parameters from the gradient graph, removes the corresponding chunk of Adam state, and reduces activation memory (one weight tensor instead of two). We should expect a measurable throughput increase on B1 (possibly 10–25%) with no effect on B2/B3 (their lm_head is tiny, so tying saves almost nothing). This is a clean falsifiable prediction: weight tying's throughput delta should be asymmetric across the three benchmarks.

- **Profile-guided optimization order:** If Phase 2's goal is to maximize measurable improvements per unit of engineering effort, "anything that affects the lm_head" is the highest-leverage intervention on B1. Weight tying first, then fp16 / bf16 autocast (Phase 3) which will especially help the wide lm_head matmul.

- **This finding also explains why B1's val_loss is higher in absolute terms than B2/B3** — not really, actually. Val loss being higher on TinyStories vs. chess isn't a throughput story; it's an entropy story (natural language is higher-entropy than chess moves at the tokenization granularity used). Keeping that separate from this throughput observation to avoid conflating two things.

- **Future benchmarks that use small vocabularies will train much faster** than natural-language ones at the same architecture. If we add a protein-sequence or a music-notation benchmark later, plan wallclock accordingly.
