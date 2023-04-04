import random
from pathlib import Path

import torch
from environs import Env

env = Env()

RUN_DIR: Path = env.path("RUN_DIR")
DATASETS_DIR: Path = env.path("DATASETS_DIR", default=Path.home() / "datasets")
RND_SEED: int = env.int("RND_SEED", default=42)

random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
