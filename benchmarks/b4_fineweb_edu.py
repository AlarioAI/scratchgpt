"""B4: FineWeb-Edu general-text benchmark."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from benchmarks._shared import (
    build_standard_config,
    materialize_text_dataset,
    parse_arch_overrides,
    run_benchmark,
)
from scratchgpt.data.hf_datasource import HFDataSource
from scratchgpt.tokenizer.hf_tokenizer import HuggingFaceTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--slug", type=str, default="b4-fineweb-edu")
    p.add_argument("--dataset-name", type=str, default="codelion/fineweb-edu-100M")
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset-size", type=int, default=100_000, help="Number of usable documents to materialize")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--min-chars", type=int, default=200, help="Skip shorter documents before counting subset-size")
    p.add_argument("--max-chars", type=int, default=8192, help="Truncate each document before tokenization")
    p.add_argument("--tokenizer-repo", type=str, default="gpt2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--arch-override",
        action="append",
        default=None,
        help="Architecture flag override as KEY=VALUE; may repeat.",
    )
    args = p.parse_args()

    dataset_label = args.dataset_name if args.dataset_config is None else f"{args.dataset_name}/{args.dataset_config}"
    print(f"Loading {dataset_label}...")
    loaded = load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split=args.split,
    )
    ds = materialize_text_dataset(
        loaded,
        text_column=args.text_column,
        limit=args.subset_size,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(f"  using {len(ds):,} documents")

    tokenizer = HuggingFaceTokenizer.from_hub(args.tokenizer_repo)
    data_source = HFDataSource.from_hf_dataset(ds, text_column=args.text_column)

    config = build_standard_config(
        vocab_size=tokenizer.vocab_size,
        arch_overrides=parse_arch_overrides(args.arch_override),
    )
    dataset_key = (
        f"codelion-fineweb-edu-100m-first-{len(ds)}-docs"
        f"-min-{args.min_chars}-max-{args.max_chars}-chars"
    )
    run_benchmark(
        slug=args.slug,
        data_source=data_source,
        tokenizer=tokenizer,
        config=config,
        runs_dir=args.runs_dir,
        device=torch.device(args.device),
        dataset_key=dataset_key,
    )


if __name__ == "__main__":
    main()
