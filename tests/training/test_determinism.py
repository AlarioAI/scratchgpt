import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scratchgpt.training.determinism import seed_everything, seed_worker


class _ToyDataset(Dataset[int]):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> int:
        # Access RNGs so non-determinism would show up here.
        return int(np.random.randint(0, 1_000_000) + torch.randint(0, 1_000_000, (1,)).item())


def test_seed_everything_makes_torch_and_numpy_deterministic() -> None:
    seed_everything(1337)
    a = (torch.randn(10).tolist(), np.random.randn(10).tolist())
    seed_everything(1337)
    b = (torch.randn(10).tolist(), np.random.randn(10).tolist())
    assert a == b


def test_dataloader_with_worker_seeding_is_deterministic() -> None:
    def make_loader() -> DataLoader[int]:
        seed_everything(1337)
        generator = torch.Generator()
        generator.manual_seed(1337)
        return DataLoader(
            _ToyDataset(64),
            batch_size=8,
            shuffle=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    first = [list(batch.tolist()) for batch in make_loader()]
    second = [list(batch.tolist()) for batch in make_loader()]
    assert first == second
