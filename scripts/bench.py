"""Throughput + memory benchmark for a given model config.

Usage:
  python scripts/bench.py --block-size 256 --embedding-size 384 --num-heads 6 \
      --num-blocks 6 --batch-size 32 --vocab-size 50257 --iters 20 --device cuda
"""
import argparse
import json
import time

import torch
from torch.nn import functional as F

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTConfig, ScratchGPTTraining
from scratchgpt.model.factory import build_language_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--embedding-size", type=int, default=384)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=50_257)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-variant", choices=["classic", "modern"], default="classic")
    p.add_argument("--position-encoding", choices=["learned", "rope"], default="learned")
    p.add_argument("--normalization", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--ffn-variant", choices=["mlp", "swiglu"], default="mlp")
    p.add_argument("--qk-norm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    arch = ScratchGPTArchitecture(
        block_size=args.block_size, embedding_size=args.embedding_size,
        num_heads=args.num_heads, num_blocks=args.num_blocks, vocab_size=args.vocab_size,
        model_variant=args.model_variant,
        position_encoding=args.position_encoding,
        normalization=args.normalization,
        ffn_variant=args.ffn_variant,
        qk_norm=args.qk_norm,
    )
    training = ScratchGPTTraining(batch_size=args.batch_size)
    model = build_language_model(ScratchGPTConfig(architecture=arch, training=training)).to(device)
    model.train()

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    # Warmup
    for _ in range(args.warmup):
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t_fwd = t_bwd = 0.0
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd += time.perf_counter() - t1
        model.zero_grad(set_to_none=True)

    tokens = args.iters * args.batch_size * args.block_size
    total = t_fwd + t_bwd
    report = {
        "forward_ms": (t_fwd / args.iters) * 1000,
        "backward_ms": (t_bwd / args.iters) * 1000,
        "tokens_per_sec": tokens / total,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
