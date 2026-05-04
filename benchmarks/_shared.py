"""Shared wiring for Phase 1 baseline benchmarks."""
import hashlib
import os
from pathlib import Path

import torch
from torch.optim import AdamW

from scratchgpt.config import ScratchGPTArchitecture, ScratchGPTConfig, ScratchGPTTraining
from scratchgpt.data.datasource import DataSource
from scratchgpt.model.model import TransformerLanguageModel
from scratchgpt.tokenizer.base_tokenizer import Tokenizer
from scratchgpt.training.determinism import seed_everything
from scratchgpt.training.run_recorder import RunRecorder
from scratchgpt.training.trainer import Trainer

STANDARD_STEPS = 5000
STANDARD_EVAL_EVERY = 500
STANDARD_LOG_EVERY = 50


def build_standard_config(vocab_size: int) -> ScratchGPTConfig:
    """Config used by every Phase 1 baseline. Phase 2+ will override pieces."""
    return ScratchGPTConfig(
        architecture=ScratchGPTArchitecture(
            block_size=256, embedding_size=384, num_heads=6, num_blocks=6,
            vocab_size=vocab_size,
        ),
        training=ScratchGPTTraining(
            max_steps=STANDARD_STEPS,
            eval_every_steps=STANDARD_EVAL_EVERY,
            log_every_steps=STANDARD_LOG_EVERY,
            learning_rate=3e-4,
            batch_size=32,
            dropout_rate=0.1,
            random_seed=1337,
            iteration_type="chunking",
        ),
    )


def run_benchmark(
    slug: str,
    data_source: DataSource,
    tokenizer: Tokenizer,
    config: ScratchGPTConfig,
    runs_dir: Path,
    device: torch.device,
    dataset_key: str,
) -> Path:
    """Train and write a run directory; stamps dataset_key on the trainer so
    the benchmark_contract in summary.json identifies the exact data subset."""
    seed_everything(config.training.random_seed)
    model = TransformerLanguageModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)

    recorder = RunRecorder(base_dir=runs_dir, run_slug=slug)
    recorder.save_config(config.model_dump())

    trainer = Trainer(
        model=model, config=config.training, optimizer=optimizer,
        experiment_path=recorder.run_dir / "legacy_ckpt", device=device,
        recorder=recorder,
    )
    trainer._dataset_key = dataset_key  # type: ignore[attr-defined]
    trainer.train(data_source=data_source, tokenizer=tokenizer)
    print(f"Run complete: {recorder.run_dir}")
    return recorder.run_dir


def cached_chess_corpus(game_url: str, max_games: int) -> str:
    """Download + parse Lichess games once per (url, max_games) pair; cache the parsed text."""
    from examples.chess import ChessDataLoader

    cache_root = Path(
        os.environ.get("SCRATCHGPT_BENCH_CACHE", Path.home() / ".cache" / "scratchgpt" / "bench")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{game_url}|{max_games}".encode()).hexdigest()[:16]
    cache_file = cache_root / f"chess-{key}.txt"

    if cache_file.exists():
        print(f"Using cached chess corpus: {cache_file}")
        return cache_file.read_text()

    loader = ChessDataLoader(game_url)
    games_text = loader.download_and_parse()
    lines = games_text.split("\n")[:max_games]
    subset = "\n".join(lines)
    cache_file.write_text(subset)
    print(f"Cached chess corpus to: {cache_file}")
    return subset
