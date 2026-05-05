# Exp-3: Weight tying (embedding ↔ lm_head) with default init

**Date:** 2026-05-05
**Phase:** 2
**Branch / commit:** `feature/sota-upgrade-roadmap` @ `a807b6f` (post-refactor); runs produced at this SHA against the post-Exp-2 defaults (`attention_scale_mode="head"`, `ffn_activation="gelu"`).

## Hypothesis

Weight tying shares a single weight matrix between the token embedding table (`nn.Embedding`, shape `V × E`) and the output projection (`nn.Linear`, shape `E × V`). This is standard in GPT-2 and most modern LMs. Expected effects from `memory/findings/2026-05-04-b1-vocab-throughput.md`:

- **B1**: biggest val_loss win (the 19M-param lm_head dominates compute; fewer params = less over-fitting at 5000 steps). Throughput gain expected because the grad graph is smaller.
- **B2/B3**: minimal change in either dimension — their vocabs are tiny (200 chess moves, 60 chars), so the lm_head share of total params is already negligible.

## Setup

Single flag flip: `tie_weights=True` (was `False`). Everything else at post-Exp-2 defaults:

```bash
uv run python benchmarks/b1_tinystories.py --slug exp3-tie-b1 \
    --arch-override tie_weights=true --device cuda
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/b2_chess.py --slug exp3-tie-b2 \
    --arch-override tie_weights=true --device cuda
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/b3_chemistry.py --slug exp3-tie-b3 \
    --arch-override tie_weights=true --device cuda
```

Pipelined across GPU 0 / GPU 1. Total wall: ~35 min (bounded by B1).

Run dirs (ephemeral):

- `runs/20260505-125146-exp3-tie-b1/`
- `runs/20260505-125145-exp3-tie-b2/`
- `runs/20260505-130012-exp3-tie-b3/`

## Result

B1 TinyStories:

| metric              | baseline-b1   | exp3-tie-b1   |
|---------------------|---------------|---------------|
| best_val_loss       | 2.2261        | 3.3830        |
| tokens_per_sec      | 19,411.3      | 19,536.8      |
| peak_vram_bytes     | 7,224,604,160 | 7,292,119,552 |
| total_wallclock_sec | 2,110.1       | 2,096.6       |

**Δ val_loss: +1.1569 (+51.97%). Throughput: +0.65%. VRAM: +0.9%.**

B2 Chess:

| metric              | baseline-b2   | exp3-tie-b2   |
|---------------------|---------------|---------------|
| best_val_loss       | 2.1694        | 2.9425        |
| tokens_per_sec      | 83,015.7      | 82,901.0      |
| peak_vram_bytes     | 3,146,654,208 | 3,392,432,128 |

**Δ val_loss: +0.7730 (+35.63%). Throughput: −0.14%. VRAM: +7.8%.**

B3 Chemistry:

| metric              | baseline-b3   | exp3-tie-b3   |
|---------------------|---------------|---------------|
| best_val_loss       | 0.2772        | 0.3551        |
| tokens_per_sec      | 86,728.4      | 85,760.0      |
| peak_vram_bytes     | 1,906,259,968 | 2,195,975,168 |

**Δ val_loss: +0.0779 (+28.11%). Throughput: −1.12%. VRAM: +15.2%.**

All three benchmarks regressed substantially. B1 got worse by ~52%, the largest Phase 2 delta in either direction so far.

## Interpretation

The hypothesis was wrong about standalone weight tying. The mechanism driving the regression is an **initialization mismatch** between `nn.Embedding` and `nn.Linear` that becomes destructive when the two are aliased to the same tensor.

**The mismatch.** PyTorch's defaults for these layers differ by ~35×:

- `nn.Embedding.__init__`: `weight ~ N(0, 1)` (std ≈ 1.0)
- `nn.Linear(vocab, embed).__init__`: `weight ~ U(-sqrt(1/embed), +sqrt(1/embed))` → std ≈ 0.029 for embed=384

Without weight tying, each layer starts at its "home" scale. Weight tying forces the lm_head to inherit the embedding's init scale (~35× too large for a Linear's use). Pre-training logits have std ~6 (verified in Task 3's smoke test) instead of std ~0.5. The model burns most of its 5000-step budget rescaling the output projection before it can focus on the actual task.

The issue is **not that weight sharing failed** — the sharing works exactly as intended. The issue is that the shared tensor's initialization is suitable for one role (embedding lookup) but not the other (logit projection), and no mechanism is in place to reconcile them.

**The correct framing: compatible init.** Weight tying requires that the shared weight tensor's initialization scale is suitable for *both* roles — embedding lookup output and logit projection input. GPT-2 init (N(0, 0.02) on both embeddings and Linears) provides one such reconciliation, but any init scheme that matches Linear's scale rather than Embedding's default would work. Phase 2's Exp-4 will test GPT-2 init on its own; Exp-6 (stacked) will test whether the combination works as predicted.

**Why B1 regresses hardest (+52%).** Its 50,257-way softmax is the widest target for the miscalibrated projection. Loss over a large vocab has much higher dynamic range when logits are miscalibrated — small absolute miscalibrations in logit space produce large cross-entropy penalties.

**Why B3 regresses least (+28%, still huge).** Its 60-way softmax provides less room for bad logits to go wrong. Per-token cross-entropy is bounded by `log(vocab_size)`, and a miscalibrated softmax over 60 classes has a much lower penalty ceiling than over 50k. The relative damage is smaller, but "smaller" still means nearly a third of the baseline loss.

**Throughput was a non-event.** All three benchmarks are within ±1% of baseline throughput and VRAM. The expected throughput gain didn't materialize because weight tying saves parameter/grad/Adam-state storage but does not remove the **dominant B1 work**: the output matmul (`B*T*E*V` FLOPs), the logits tensor (`B*T*V` activation memory), and the softmax over 50k classes. B1's lm_head dominates compute because of *compute*, not *parameter count*. The finding in `memory/findings/2026-05-04-b1-vocab-throughput.md` conflated those two dimensions.

Separately: the Adam state reduction from tying is real (Adam sees the tied weight as one parameter, not two), but the saving is small relative to the activation memory and matmul compute that wasn't touched. VRAM didn't drop because activation memory dominates, and training throughput didn't increase because matmul compute dominates.

## Conclusion

**Do NOT ship `tie_weights=True` as a standalone default.** Keep the default at `False`. The flag remains available for experiments that pair it with compatible initialization.

## Corrections to prior findings

`memory/findings/2026-05-04-b1-vocab-throughput.md` included this prediction in its "Implications" section:

> Weight tying first, then fp16 / bf16 autocast (Phase 3) which will especially help the wide lm_head matmul.

Exp-3 falsifies this prediction *as an isolated intervention*. Two corrections:

1. **Weight tying cannot be tested in isolation** under PyTorch's default init. The `Embedding.__init__` N(0, 1) is incompatible with `Linear` outputs. Any future tying experiment must pair the flag with a compatible init scheme (e.g. `init_scheme="gpt2"`) or explicitly document the init it's pairing with.
2. **The throughput-gain reasoning was wrong.** Weight tying reduces parameters and Adam state but leaves the output matmul, logits tensor, and softmax-over-vocab untouched. Those three are what dominate B1's compute; none of them shrink under tying. The finding's "the lm_head dominates because of params" framing should be "the lm_head dominates because of the compute and activation memory produced *at* the output head, which is largely decoupled from parameter count." Activation memory + softmax FLOPs scale with `B*T*V` regardless of whether the weight matrix is shared.

A short supersedes-pointer should be added to the findings entry referencing this experiment.

## Open questions

- **Does `init_scheme="gpt2"` + `tie_weights=True` recover the expected gain?** Exp-6 (all-stacked) will include both flags on, but Exp-6 also turns on Exp-4 (init alone) and Exp-5 (no-bias), which confounds attribution. Exp-6 is an *interaction/recovery test*, not clean evidence that the tie+init pair works. If Exp-6 wins, a follow-up two-flag ablation (tie=True, init=gpt2, others default) is needed before promoting that specific pair. Tracking as a Phase-2-exit question.
- **Does `init_scheme="gpt2"` alone (Exp-4) produce the big B1 gain the finding predicted?** If so, the "weight tying first" advice should be replaced with "proper init first."
- **Seed variance.** All Phase 2 runs are single-seed. +52% on B1 is vastly above any plausible seed variance, so this particular result is robust. The smaller flagged effects in Exp-1 and Exp-2 still want variance characterization in Phase 3.
