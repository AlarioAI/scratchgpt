"""B3: Chemistry reaction prediction benchmark."""
import argparse
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset

from benchmarks._shared import build_standard_config, run_benchmark
from scratchgpt.data import create_data_source
from scratchgpt.tokenizer.char_tokenizer import CharTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--slug", type=str, default="b3-chemistry")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("Loading USPTO-50k...")
    ds = load_dataset("pingzhili/uspto-50k", split="train")
    # NOTE: CharTokenizer tokenizes per character, so BOS/EOS bracket strings
    # would become literal bracket noise. Phase 5 will introduce a tokenizer
    # that supports special tokens properly; Phase 1 baseline uses raw text.
    reactions: list[str] = []
    for row in ds:
        rxn = str(row.get("rxn_smiles", row.get("reaction_smiles", ""))).strip()
        if rxn and ">>" in rxn:
            reactions.append(rxn)
    text = "\n".join(reactions)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_file = tmp_path / "reactions.txt"
        data_file.write_text(text)

        tokenizer = CharTokenizer(text=text)
        data_source = create_data_source(str(data_file))

        config = build_standard_config(vocab_size=tokenizer.vocab_size)
        run_benchmark(
            slug=args.slug, data_source=data_source, tokenizer=tokenizer,
            config=config, runs_dir=args.runs_dir, device=torch.device(args.device),
            dataset_key=f"uspto-50k-{len(reactions)}-reactions",
        )


if __name__ == "__main__":
    main()
