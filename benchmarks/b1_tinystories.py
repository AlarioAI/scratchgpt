"""B1: TinyStories language benchmark. Fixed 5000 steps."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from benchmarks._shared import build_standard_config, run_benchmark
from scratchgpt.data.hf_datasource import HFDataSource
from scratchgpt.tokenizer.hf_tokenizer import HuggingFaceTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--slug", type=str, default="b1-tinystories")
    p.add_argument("--subset-size", type=int, default=500_000,
                   help="Number of TinyStories rows to use (0 = all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.subset_size > 0 and args.subset_size < len(ds):
        ds = ds.select(range(args.subset_size))
    print(f"  using {len(ds):,} rows")

    tokenizer = HuggingFaceTokenizer.from_hub("gpt2")
    data_source = HFDataSource.from_hf_dataset(ds, text_column="text")

    config = build_standard_config(vocab_size=tokenizer.vocab_size)
    run_benchmark(
        slug=args.slug, data_source=data_source, tokenizer=tokenizer,
        config=config, runs_dir=args.runs_dir, device=torch.device(args.device),
        dataset_key=f"tinystories-{len(ds)}-rows",
    )


if __name__ == "__main__":
    main()
