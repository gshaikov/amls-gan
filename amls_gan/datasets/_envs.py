from pathlib import Path

from environs import Env

env = Env()

DATASETS_DIR: Path = env.path("DATASETS_DIR", default=Path.home() / "datasets")
