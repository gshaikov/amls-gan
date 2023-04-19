import logging
from pathlib import Path
from typing import Any, Literal, cast

import torch
from environs import Env
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from amls_gan.datasets._envs import DATASETS_DIR

logging.basicConfig(level=logging.INFO)

env = Env()

Splits = Literal["train", "test"]


class TensorCIFAR10(Dataset[Tensor]):
    """
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
    """

    dir: Path = env.path("CIFAR10_DIR", default=DATASETS_DIR / "cifar10")

    @classmethod
    def create(cls, split: Splits, download: bool = False) -> "TensorCIFAR10":
        """
        Initialise CIFAR10 tensor dataset, optionally downloading it first.

        TODO: download should be an ENV var.

        Args:
            split: train or test.
            download (optional): if True, download dataest if it's not yet done.
        """
        if not cls.dir.exists():
            cls.dir.mkdir(parents=True)
        ds = CIFAR10(root=str(cls.dir), train=split == "train", download=download)
        return cls(ds)

    def __init__(self, ds: CIFAR10) -> None:
        self.ds = ds

        self.split: Splits = "train" if ds.train is True else "test"

        self.transform = T.PILToTensor()

    def __getitem__(self, index: int) -> Tensor:
        img, _ = cast(tuple[Image, Any], self.ds[index])
        img_t = self.transform(img)
        assert img_t.dtype == torch.uint8
        return img_t

    def __len__(self) -> int:
        return len(self.ds)

    def image_size(self) -> tuple[int, int, int]:
        return tuple(self[0].shape)


if __name__ == "__main__":
    train = TensorCIFAR10.create("train", download=True)
    test = TensorCIFAR10.create("test", download=True)
    assert isinstance(train[0], Tensor)
    assert isinstance(test[0], Tensor)
    assert train[0].dtype == torch.uint8
    assert test[0].dtype == torch.uint8

    logging.info(f"example size: {train.image_size()}")
