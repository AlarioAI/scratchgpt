# Model-first roadmap after Phase 2

**Date:** 2026-05-14
**Phase:** 2 closeout / Phase 4 planning

## Context

The original SOTA roadmap ordered the work as Phase 3 trainer upgrades, Phase 4 architecture modernization, Phase 5 data and eval, Phase 6 research-grade techniques, and Phase 7 autoresearch. After the Phase 2 tier-1 runs, the collaborator preference is to prioritize algorithmic model improvements over hardware and throughput work. Mixed precision, `torch.compile`, KV-cache efficiency, and grouped-query attention still matter, but they should not dominate the next research cycle if the goal is to squeeze more quality out of the model and eventually write a paper.

The current benchmark suite is useful but not sufficient for that paper-quality story. TinyStories is pedagogically helpful but too narrow. Chess has domain structure, but val_loss needs a legal-move evaluator before it becomes a strong quality metric. Chemistry currently uses character-level SMILES and is already near a floor, so architecture deltas can look artificially small or noisy. RoPE and long-context attention changes also need a benchmark that can expose context-length behavior; the current `block_size=256` contract will not.

This entry records the high-level roadmap decision. It is not an experiment result. Each concrete model change still needs its own hypothesis-first `memory/experiments/` entry, B1/B2/B3 or refreshed benchmark runs, `scripts/compare.py` output, and a decision before any default flip.

## Public architecture signals to incorporate

The next model phase should use current open model families as reference points, while keeping ScratchGPT's pedagogical core intact.

- **Qwen3 / Qwen3-Next / Qwen3.5 / Qwen3.6:** public materials point to RoPE, RMSNorm, SiLU-gated MLPs, Q/K normalization, grouped-query attention, sparse MoE, hybrid attention/linear-attention blocks, and multi-token prediction as recurring design choices. Relevant references: [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/), [Qwen3 Transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3), [Qwen3 source](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py), [Qwen3-Next docs](https://huggingface.co/docs/transformers/model_doc/qwen3_next), and [Qwen3.6 README](https://github.com/QwenLM/Qwen3.6/blob/main/README.md).
- **Gemma 4:** public materials emphasize hybrid local/global attention, p-RoPE for global layers, unified Keys/Values, dense and MoE variants, Per-Layer Embeddings for smaller effective-parameter models, and public MTP drafters for speculative decoding. Relevant references: [Gemma 4 launch](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), and [Gemma 4 MTP note](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/).
- **Foundational references already relevant to the roadmap:** RoPE ([RoFormer](https://arxiv.org/abs/2104.09864)), RMSNorm ([RMSNorm](https://arxiv.org/abs/1910.07467)), SwiGLU / gated FFNs ([GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)), QK-norm ([QKNorm](https://arxiv.org/abs/2010.04245)), NormFormer ([NormFormer](https://arxiv.org/abs/2110.09456)), Peri-LN ([Peri-LN](https://arxiv.org/abs/2502.02732)), Differential Attention ([Differential Transformer](https://arxiv.org/abs/2410.05258)), nGPT ([nGPT](https://arxiv.org/abs/2410.01131)), Mamba ([Mamba](https://arxiv.org/abs/2312.00752)), and Mamba-2 / SSM duality ([Mamba-2](https://arxiv.org/abs/2405.21060)).

## Roadmap decision

The next roadmap is model-first, but not benchmark-blind. Phase 2 still needs closeout, then the project should refresh the evaluation surface enough that architecture changes have a fair test. Trainer work that is primarily hardware or throughput oriented moves behind the first dense architecture stack, except when a trainer feature is necessary for scientific validity or tractable experimentation.

The clean `tie_weights=true` + `init_scheme=gpt2` ablation is explicitly deferred. It is not a blocker for Phase 2 closeout or generation-2 baselines. Until that future ablation is run, `tie_weights=false` remains the accepted default.

The reordered high-level plan:

1. **Phase 2 closeout remains mandatory.**
   Write the Phase 2 retrospective, write the default-flip decision doc, and produce generation-2 canonical baselines using the accepted defaults: `attention_scale_mode="head"`, `ffn_activation="gelu"`, `init_scheme="gpt2"`, `tie_weights=false`, and `use_bias=true`.

2. **Phase 3a becomes benchmark refresh for model research.**
   Keep TinyStories as a fast smoke test, but add a stronger general-text benchmark from FineWeb-Edu or SmolLM-Corpus, add BabyLM as a sample-efficient human-text benchmark, keep chess but add legal-move-rate before treating it as a quality benchmark, rework chemistry tokenization so BOS/EOS are atomic rather than bracket noise, add SMILES-validity evaluation, and add a long-context or retrieval-style evaluation for RoPE/p-RoPE.

3. **Phase 4a builds the modern dense decoder branch.**
   Create a parallel `model_modern.py` or equivalent architecture path so the educational `model.py` remains readable. Use single-QKV projection as plumbing for SDPA and modern attention, but do not present it as a model-quality win by itself.

4. **Phase 4b runs isolated dense architecture ablations.**
   Test RoPE or p-RoPE, RMSNorm, parameter-matched SwiGLU, and QK-norm individually. Each gets a hypothesis, benchmark triple or refreshed benchmark suite, compare output, and memory entry.

5. **Phase 4c tests dense stacks.**
   Test RoPE+RMSNorm+SwiGLU, then add QK-norm. Compare isolated gains against the stack to identify interactions and avoid blindly shipping a bundled architecture.

6. **Phase 4d tests long-context architecture choices.**
   Test local/global attention mixes, p-RoPE variants, and possibly grouped-query attention if the new long-context eval shows a reason. Treat GQA primarily as an efficiency/generation-memory feature unless it changes quality under the benchmark.

7. **Phase 6-style research branches move after the dense stack.**
   Sparse MoE, Differential Attention, nGPT, Gated DeltaNet / linear-attention hybrids, Mamba-style branches, and multi-token prediction are interesting but should be separate branches after the dense baseline is strong. They are more invasive and require clearer experiment framing.

8. **Trainer/hardware upgrades become support work, not the main thread.**
   Mixed precision, `torch.compile`, KV cache, and resumable checkpoints are still useful, but should be scheduled when they unblock experiment throughput, reproducibility, or a specific model evaluation. They should not be recorded as model-quality wins unless they change training outcomes under the benchmark contract.

## Benchmark candidates

The benchmark refresh should prefer fixed, reproducible subsets with explicit dataset keys and committed baseline artifacts.

- **FineWeb-Edu or SmolLM-Corpus fineweb-edu-dedup** for general LM quality. FineWeb-Edu is a high-quality educational subset of FineWeb, and SmolLM-Corpus is explicitly curated for small-language-model training.
- **BabyLM strict-small or another fixed BabyLM split** for sample-efficient human-written text. This gives a better paper comparison point than TinyStories alone.
- **Existing TinyStories** retained as a pedagogical smoke benchmark, not the primary generalization claim.
- **Existing chess** retained, but with legal-move-rate added before architecture claims rely on it.
- **Chemistry with atomic special tokens and valid-SMILES-rate** replacing the current char-level-only interpretation.
- **A long-context evaluation** added before judging RoPE/p-RoPE, sliding-window/global attention, or retrieval-sensitive variants.

## Experiment tracking requirements

Every roadmap item that can affect training outcomes must follow the existing lab-notebook protocol:

1. Write the hypothesis and setup in `memory/experiments/` before spending the run.
2. Use exact run slugs and preserve the standard contract unless the contract change is the point of the experiment.
3. If a benchmark changes, write a `memory/decisions/` entry explaining the new contract and why old/new comparisons need generation labels.
4. Record `scripts/compare.py` output and both marginal and cumulative deltas.
5. Keep run directories ephemeral unless they are canonical baselines; canonical baselines commit only lightweight artifacts.
6. Do not flip defaults from a bundled stack unless isolated ablations explain the gain.
7. For paper-writing, keep public-source links and mechanism hypotheses in the memory entry, not only in chat.

## Expected consequences

This reordering should produce a stronger research narrative: the project first establishes a credible measurement surface, then tests modern dense-decoder changes, then escalates to larger architectural departures. It also keeps ScratchGPT's educational promise intact by preserving the original readable model and treating modern machinery as an opt-in research path.

The cost is that benchmark refresh delays the most exciting architecture code. That delay is intentional. Without better benchmarks, a Phase 4 win could be a TinyStories artifact, a char-level chemistry floor effect, or a long-context improvement with no task that can observe it.

## How we'd know if this was wrong

This decision should be revisited if the benchmark refresh takes more time than the model work it is meant to support, if FineWeb-Edu / SmolLM-Corpus subsets are too expensive for the project's available compute, if BabyLM does not produce stable enough deltas at this model size, or if the first dense architecture ablations show large unambiguous improvements even on the existing B1/B2/B3 suite. In that case, the project can temporarily run architecture-first while keeping the benchmark refresh as a parallel track.
