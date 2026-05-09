# Memory log

Append-only chronological index of memory entries. Newest at the bottom.

```
2026-05-04  phases/2026-05-04-phase1-harness.md                 Phase 1 retrospective: measurement harness shipped
2026-05-04  experiments/2026-05-04-phase1-baselines.md          B1/B2/B3 baselines at the standard contract
2026-05-04  decisions/2026-05-04-benchmark-contract-scope.md    What the benchmark_contract covers and why
2026-05-04  decisions/2026-05-04-checkpoint-exclusion.md        Checkpoints stay local; only lightweight artifacts commit
2026-05-04  decisions/2026-05-04-gpu-only-baselines.md          Committed baselines must be GPU-trained at full contract
2026-05-04  findings/2026-05-04-b1-vocab-throughput.md          B1 is 4x slower per token because the GPT-2 lm_head dominates
2026-05-04  findings/2026-05-04-b3-charlevel-floor.md           B3 val_loss=0.28 is a char-level floor, not a hard target
2026-05-04  experiments/2026-05-04-exp1-attention-scale-fix.md  Exp-1: attention_scale_mode=head ships as default (-1.84% on B3, -0.27% on B1, flat on B2)
2026-05-05  experiments/2026-05-04-exp2-gelu.md                 Exp-2: ffn_activation=gelu ships as default (-3.25% on B3, -2.83% on B1, flat on B2, +4-15% VRAM)
2026-05-05  experiments/2026-05-05-exp3-weight-tying.md         Exp-3: tie_weights=True standalone REJECTED (+52% val_loss on B1); init-mismatch bug, must pair with compatible init
2026-05-05  experiments/2026-05-05-exp4-gpt2-init.md            Exp-4: init_scheme=gpt2 ships as default; largest Phase 2 win (-11.27% B1, -6.29% B2, -9.45% B3 cumulative)
2026-05-05  phases/2026-05-05-phase2-progress.md                Phase 2 mid-phase handoff note: state, protocol, pending work, and entry points for resuming
2026-05-09  experiments/2026-05-09-exp5-no-bias.md              Exp-5: use_bias=false REJECTED; small consistent marginal regression (+0.60% B1, +0.19% B2, +0.28% B3)
2026-05-09  experiments/2026-05-09-exp6-full-stack.md           Exp-6: full stack rescues tie_weights under GPT-2 init on B1/B2, but no full-stack default flip; run clean tie+init ablation next
```
