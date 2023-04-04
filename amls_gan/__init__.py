from pathlib import Path

from environs import Env

env = Env()

RUN_DIR: Path = env.path("RUN_DIR")
DATASETS_DIR: Path = env.path("DATASETS_DIR", default=Path.home() / "datasets")
