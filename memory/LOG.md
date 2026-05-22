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
2026-05-14  decisions/2026-05-14-model-first-roadmap.md         Reorder roadmap toward model-first architecture work, benchmark refresh, and paper-grade experiment tracking
2026-05-14  phases/2026-05-14-phase2-tier1.md                   Phase 2 retrospective: accepted attention-scale, GELU, GPT-2 init; rejected tying/no-bias/full-stack defaults
2026-05-14  decisions/2026-05-14-phase2-defaults.md             Final Phase 2 architecture defaults for generation-2 baselines and future comparisons
2026-05-14  experiments/2026-05-14-phase2-baselines.md          Generation-2 canonical baselines after accepted Phase 2 defaults
2026-05-21  decisions/2026-05-21-phase3a-benchmark-refresh.md   Phase 3a benchmark refresh starts with B4 FineWeb-Edu 100M sample
2026-05-21  experiments/2026-05-21-phase3a-b4-fineweb-edu-baseline.md  B4 FineWeb-Edu 100M generation-3a baseline accepted
2026-05-21  decisions/2026-05-21-modern-model-path.md           Phase 4 ablations use model_variant=modern in a parallel model path
2026-05-21  experiments/2026-05-21-phase4-exp1-rope-b4.md       Phase 4 Exp-1: RoPE accepted as modern-path candidate on B4 (-2.71% val loss, -7.36% throughput)
2026-05-21  experiments/2026-05-21-phase4-exp2-rmsnorm-b4.md    Phase 4 Exp-2: RMSNorm not accepted standalone on B4 (+0.15% val loss, -3.15% throughput)
2026-05-21  experiments/2026-05-21-phase4-exp3-swiglu-b4.md     Phase 4 Exp-3: parameter-matched SwiGLU accepted as modern-path candidate on B4 (-0.80% val loss, -2.50% throughput)
2026-05-21  experiments/2026-05-21-phase4-exp4-qk-norm-b4.md    Phase 4 Exp-4: QK-norm accepted as modern-path candidate on B4 (-0.56% val loss, -6.31% throughput)
2026-05-21  decisions/2026-05-21-phase4-stack-order.md          Phase 4 stack order: run RoPE + parameter-matched SwiGLU next on B4
2026-05-21  experiments/2026-05-21-phase4-exp5-rope-swiglu-b4.md  Phase 4 Exp-5: RoPE + SwiGLU accepted as leading B4 stack (-3.02% val loss, -8.62% throughput)
2026-05-21  decisions/2026-05-21-phase4-post-exp5-branch.md     Phase 4 branch: run RoPE + QK-norm next before a triple stack
2026-05-21  experiments/2026-05-21-phase4-exp6-rope-qk-norm-b4.md  Phase 4 Exp-6: RoPE + QK-norm becomes best B4 quality stack (-3.30% val loss, -12.46% throughput)
2026-05-21  decisions/2026-05-21-phase4-post-exp6-branch.md     Phase 4 branch: run RoPE + SwiGLU + QK-norm next on B4
2026-05-21  experiments/2026-05-21-phase4-exp7-rope-swiglu-qk-norm-b4.md  Phase 4 Exp-7 setup: full RoPE + SwiGLU + QK-norm stack on B4
2026-05-21  experiments/2026-05-21-phase4-exp7-rope-swiglu-qk-norm-b4.md  Phase 4 Exp-7: full RoPE + SwiGLU + QK-norm accepted as leading B4 quality stack (-4.25% val loss, -13.69% throughput vs baseline)
2026-05-21  decisions/2026-05-21-phase4-full-stack-next.md      Phase 4 decision: repeat full-stack B4 before B1/B2/B3 transfer checks
```
