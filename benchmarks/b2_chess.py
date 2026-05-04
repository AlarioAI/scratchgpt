"""B2: Chess move prediction benchmark, via cached Lichess corpus."""
import argparse
import tempfile
from pathlib import Path

import torch

from benchmarks._shared import build_standard_config, cached_chess_corpus, run_benchmark
from examples.chess import DEFAULT_LICHESS_URL
from examples.chess_tokenizer import ChessTokenizer
from scratchgpt.data import create_data_source


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--slug", type=str, default="b2-chess")
    p.add_argument("--game-url", type=str, default=DEFAULT_LICHESS_URL)
    p.add_argument("--max-games", type=int, default=50_000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    games_text = cached_chess_corpus(args.game_url, args.max_games)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        games_file = tmp_path / "games.txt"
        games_file.write_text(games_text)
        data_source = create_data_source(str(games_file))
        tokenizer = ChessTokenizer()

        config = build_standard_config(vocab_size=tokenizer.vocab_size)
        url_tail = args.game_url.rsplit("/", 1)[-1].replace(".zst", "")
        run_benchmark(
            slug=args.slug, data_source=data_source, tokenizer=tokenizer,
            config=config, runs_dir=args.runs_dir, device=torch.device(args.device),
            dataset_key=f"lichess-{url_tail}-first-{args.max_games}",
        )


if __name__ == "__main__":
    main()
