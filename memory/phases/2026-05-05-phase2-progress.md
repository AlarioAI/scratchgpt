# Phase 2 progress note (mid-phase handoff)

**Date:** 2026-05-05
**Status:** 4 of 6 experiments complete. Exp-5 and Exp-6 pending. Retrospective and generation-2 baselines after that.
**Branch:** `feature/sota-upgrade-roadmap`
**Commit at time of writing:** `efde85e`
**Phase 1 baseline reference:** commit `0c147b6`

This is a mid-phase progress note, not a retrospective. It exists so that anyone (future-us, a collaborator, an LLM agent picking up mid-stream) can land and continue Phase 2 exactly where the last session left off, without depending on chat transcripts.

## What is shipped

Four default-flip decisions have been committed, each backed by an experiment entry in `memory/experiments/`:

| Exp | Flag change             | B1 cumul. val_loss | B2 cumul. | B3 cumul. | Decision |
|-----|-------------------------|--------------------|-----------|-----------|----------|
| 1   | `attention_scale_mode="head"` | −0.27% | +0.01% | −1.84% | default on (`ecb73f6`) |
| 2   | `ffn_activation="gelu"`       | −2.83% | +0.08% | −3.25% | default on (`858ab54`) |
| 3   | `tie_weights=True`            | **+51.97%** | +35.63% | +28.11% | **REJECTED** (`da44062`) |
| 4   | `init_scheme="gpt2"`          | **−11.27%** | −6.29% | −9.45% | default on (`efde85e`) |

Cumulative Phase 2 improvement vs. Phase 1 baselines (at commit `efde85e`):
- **B1 TinyStories: 2.2261 → 1.9752 (−11.27%)**
- **B2 Chess: 2.1694 → 2.0329 (−6.29%)**
- **B3 Chemistry: 0.2772 → 0.2510 (−9.45%)**

Also shipped (infrastructure + quality):
- 5 architecture flags added to `ScratchGPTArchitecture` (`cbd0c13`)
- Parity test locking down default-config forward-pass determinism + 29,952 param count (`da0ff2f`)
- Flags wired into `model.py` with a `_training_step` helper and an epoch fallback docstring clarification (`7b54572`, `70eb1e6` is Phase 1 but relevant)
- `--arch-override KEY=VALUE` CLI added to all three benchmark scripts (`b2098a2`), so experiments are pure CLI invocations with no file edits
- Declarative factory-pattern refactor (`a807b6f`) — pushed Literal→object resolution into `config.py` via `make_activation()` and `attention_scale_for(head_size)`; replaced the if/else init logic with an `INIT_SCHEMES` registry. Model.py's 5 remaining `if` statements are all legitimate runtime checks (isinstance, None-check, early-exit in generate), not flag switching.

## What is pending

### Exp-5: `use_bias=False`

Single flag flip. Removes bias from all Linears and LayerNorms in the transformer blocks (Head's Q/K/V already have `bias=False` hardcoded; this affects MHA's `_proj`, both FFN Linears, both Block LayerNorms, `_block_norm`, and `_lm_head`).

**Hypothesis**: small/flat effect. Literature often reports a wash or tiny regularization-like benefit. Not confidently predicting after Exp-4's prediction miss.

Commands (pipelined across GPUs 0 and 1):

```bash
uv run python benchmarks/b1_tinystories.py \
    --slug exp5-nobias-b1 \
    --arch-override use_bias=false \
    --device cuda 2>&1 | tee /tmp/exp5_b1.log &  # GPU 0 via default CUDA_VISIBLE

CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/b2_chess.py \
    --slug exp5-nobias-b2 \
    --arch-override use_bias=false \
    --device cuda 2>&1 | tee /tmp/exp5_b2.log &  # GPU 1

# After B2 finishes:
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/b3_chemistry.py \
    --slug exp5-nobias-b3 \
    --arch-override use_bias=false \
    --device cuda 2>&1 | tee /tmp/exp5_b3.log &
```

Wall time: ~35 min (B1 bounds).

### Exp-6: all 5 flags on (interaction/recovery test)

```bash
uv run python benchmarks/b1_tinystories.py \
    --slug exp6-stack-b1 \
    --arch-override attention_scale_mode=head \
    --arch-override ffn_activation=gelu \
    --arch-override tie_weights=true \
    --arch-override init_scheme=gpt2 \
    --arch-override use_bias=false \
    --device cuda
# ...analogously for b2 and b3 on GPU 1.
```

Note: Exp-6 is the *interaction/recovery test* for the tie_weights+GPT-2-init pairing (Exp-3's regression theory). If Exp-6 wins on B1, it's suggestive but not proof — the stack also includes Exp-5's no-bias. A clean two-flag ablation (`tie_weights=true init_scheme=gpt2`, everything else at baseline) is the scientifically cleaner evidence; should be run if Exp-6's result is ambiguous or if we want to promote the `tie+init` pair on its own.

### Task 10: Sampling refactor (no training run)

Add `top_k`, `top_p`, `repetition_penalty` kwargs to `TransformerLanguageModel.generate()`. TDD with tests in `tests/test_model_generate.py`. No benchmark runs needed. Any caller of `generate()` should continue to work unchanged (defaults preserve pre-Phase-2 behavior).

### Task 11: Phase 2 retrospective + default-flip decision doc

Write `memory/phases/2026-05-05-phase2-tier1.md` (the actual retrospective, replacing/amending this progress note) and `memory/decisions/2026-05-05-phase2-default-flip.md`. Include a summary table: which flags flipped, which didn't, why.

### Task 12: Generation-2 canonical baselines

Since Phase 2 has flipped defaults (Exp-1, Exp-2, Exp-4 all flipped; Exp-5 and Exp-6 may or may not flip), the committed `runs/baseline-*` directories no longer reflect the current default config. Phase 3 will need a new reference.

Protocol:

```bash
# After all Phase 2 experiments and decisions are final:
git mv runs/baseline-b1-tinystories runs/baseline-p1-b1-tinystories
git mv runs/baseline-b2-chess       runs/baseline-p1-b2-chess
git mv runs/baseline-b3-chemistry   runs/baseline-p1-b3-chemistry

uv run python benchmarks/b1_tinystories.py --slug baseline-p2-b1-tinystories --device cuda
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/b2_chess.py        --slug baseline-p2-b2-chess       --device cuda
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/b3_chemistry.py    --slug baseline-p2-b3-chemistry   --device cuda

mv runs/*-baseline-p2-b1-tinystories runs/baseline-p2-b1-tinystories
mv runs/*-baseline-p2-b2-chess       runs/baseline-p2-b2-chess
mv runs/*-baseline-p2-b3-chemistry   runs/baseline-p2-b3-chemistry

# Update runs/BASELINES.md with both Phase 1 and Phase 2 numbers.
# git add -f runs/baseline-p1-*/{summary,metrics,env,config}.{json,jsonl,yaml} ... etc
```

## The decision protocol (every Phase 2+ experiment)

Any "what happens next" step in Phase 2+ follows this cadence:

1. **Hypothesis first.** Written *before* the run starts, committed to the experiment entry. Include predicted direction + rough magnitude. The point is calibration: wrong predictions are valuable data if they were recorded honestly.
2. **Standard command**: `uv run python benchmarks/bN_....py --slug expN-<name>-bN --arch-override KEY=VALUE --device cuda`. Pipeline B1 on GPU 0 and B2→B3 serialized on GPU 1 via `CUDA_VISIBLE_DEVICES=1` to cut wall time from ~50 min to ~35 min.
3. **Present raw numbers + proposed interpretation to the user** before committing. The user eyeballs the interpretation, suggests sharpenings, approves.
4. **Write the memory entry** at `memory/experiments/YYYY-MM-DD-exp<N>-<slug>.md` using the template in `memory/README.md`. Include both *cumulative* and *marginal* deltas — cumulative is vs. Phase 1 baseline, marginal is vs. the immediately-prior default state. Distinction matters because defaults evolve across Phase 2.
5. **Update `memory/LOG.md`** with a one-liner.
6. **If the decision is "flip default"**: edit the Literal default in `scratchgpt/config.py`, update `tests/test_config.py::test_architecture_defaults_reflect_phase2_decisions` to assert the new default.
7. **If the decision is "reject"**: keep the flag available but don't flip the default. Add a Correction section to any prior findings/predictions that this experiment contradicts.
8. **Run the standard checks** before committing: `uv run pytest -q` (must pass), `uv run ruff check .` (clean), `uv run mypy scratchgpt` (clean).
9. **Clean up the ephemeral run dirs** with `rm -rf runs/*-exp<N>-*` (the numbers are in the memory entry; run dirs are scratch). The Phase 1 baselines under `runs/baseline-*` are immutable and stay.
10. **Commit** with a structured message: "Phase 2 Exp-N: <outcome>" as subject, bullets describing the decision, cumulative + marginal deltas, mechanism summary, and links to the memory entry.
11. **Never add AI attribution trailers** (`Co-Authored-By: Claude`, "Generated with [Claude Code]", etc.). See AGENTS.md. This is the default-harness behavior that needs to be actively stripped on every commit.

## Pattern: GPU pipelining

vai03 has two RTX A6000s. Standard pattern:

- **B1 (TinyStories)** → GPU 0 (default). Takes ~35 min at 5000 steps.
- **B2 (Chess)** → GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Takes ~8 min.
- **B3 (Chemistry)** → GPU 1 via `CUDA_VISIBLE_DEVICES=1`, started right after B2 completes. Takes ~8 min.

Total wall time: ~35 min (bounded by B1). Serial would be ~50 min.

## The benchmark_contract and marginal vs. cumulative deltas

Every `summary.json` written by a Phase 1/2 run contains a `benchmark_contract` with: `max_steps=5000, block_size=256, batch_size=32, learning_rate=3e-4, random_seed=1337, dropout_rate=0.1, iteration_type="chunking", dataset_key=<per-benchmark>`. Architecture fields (`embedding_size`, `num_heads`, `num_blocks`, and now the 5 Phase 2 flags) are NOT in the contract, so `compare.py` accepts comparisons across architecture changes. That's what enables Phase 2 experiments to compare against Phase 1 baselines without `--allow-contract-mismatch`.

Cumulative delta = Exp-N vs. Phase 1 baseline (value recorded in `runs/baseline-*/summary.json`).
Marginal delta = Exp-N vs. the immediately prior committed-default state. This requires remembering what the prior state's val_loss was — see the experiment table at the top of this note, or compute it from prior experiment entries.

## Prediction-calibration notes (pattern to continue)

Three prediction misses so far, each informative:

1. **Exp-1**: I predicted "mixed results, possibly LR-regression." Actual: mild consistent improvement. Lesson: over-weighted the LR-coupling concern.
2. **Exp-3**: I predicted "biggest B1 win" citing a finding entry. Actual: +52% regression. Lesson: techniques like weight tying can't be evaluated in isolation — they carry implicit init requirements. Corrected the prior finding.
3. **Exp-4**: I predicted "modest, may not show at 5000 steps." Actual: biggest Phase 2 win by far. Lesson: "well-known technique" ≠ "small effect"; short budgets *amplify* init effects, not diminish them.

When predicting Exp-5 and Exp-6: default to literature-reported effect sizes, not smaller. Note explicit uncertainty about direction if there's an interaction with what's already been flipped.

## Entry points for a new session

If picking up from scratch:

1. Read `AGENTS.md` (rules of the road).
2. Read `memory/README.md` (lab notebook conventions).
3. Read `memory/LOG.md` (chronological index).
4. Read this progress note.
5. Read the relevant experiment entries under `memory/experiments/`.
6. Run `git log --oneline main..HEAD` to see the commit history.
7. Run `uv sync --all-groups && uv run pytest -q` to verify the environment works.
8. Fire the next pending experiment per the protocol above.

## Current branch state at time of writing

```
efde85e Phase 2 Exp-4: flip init_scheme default to gpt2                       <- HEAD
da44062 Phase 2 Exp-3: REJECT tie_weights=True as standalone default
a807b6f Refactor model.py to declarative factory pattern
858ab54 Phase 2 Exp-2: flip ffn_activation default to gelu
ecb73f6 Phase 2 Exp-1: flip attention_scale_mode default to head
b2098a2 Add --arch-override CLI to benchmarks for Phase 2 ablations
7b54572 Wire Phase 2 architecture flags into TransformerLanguageModel
da0ff2f Add Phase 2 parity test for default model behavior
cbd0c13 Add Phase 2 architecture flags to ScratchGPTArchitecture
0189ac4 Point AGENTS.md at the memory/ lab notebook
b9eafff Bootstrap memory/ scientific lab notebook
0c147b6 Commit Phase 1 baselines for B1, B2, B3
...
```

Tests: 63 passed, 1 skipped (unchanged across all Phase 2 commits). ruff and mypy strict both clean.
