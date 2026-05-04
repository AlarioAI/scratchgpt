"""Deterministic seeding for reproducible experiment runs."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG we know about.

    Call this ONCE at the start of a run, after which every subsequent random
    draw in torch / numpy / python stdlib is reproducible from this seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: makes each worker's RNG deterministic.

    PyTorch seeds torch per worker automatically; numpy and stdlib random are
    NOT seeded, which causes subtle non-determinism with num_workers > 0.
    """
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed)
    random.seed(base_seed)
