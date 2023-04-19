import random

import torch
from environs import Env

env = Env()

RND_SEED: int = env.int("RND_SEED", default=42)

random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
