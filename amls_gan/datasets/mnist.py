import logging
from pathlib import Path
from typing import Any, Literal, cast

import torch
from environs import Env
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import MNIST

from amls_gan import DATASETS_DIR

logging.basicConfig(level=logging.INFO)

env = Env()

Splits = Literal["train", "test"]


class TensorMNIST(Dataset[Tensor]):
    """
    https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
    """

    dir: Path = env.path("MNIST_DIR", default=DATASETS_DIR / "mnist")

    @classmethod
    def create(cls, split: Splits, download: bool = False) -> "TensorMNIST":
        """
        Initialise MNIST tensor dataset, optionally downloading it first.

        TODO: download should be an ENV var.

        Args:
            split: train or test.
            download (optional): if True, download dataest if it's not yet done.
        """
        if not cls.dir.exists():
            cls.dir.mkdir(parents=True)
        cifar = MNIST(root=str(cls.dir), train=split == "train", download=download)
        return cls(cifar)

    def __init__(self, mnist: MNIST) -> None:
        self.mnist = mnist

        self.split: Splits = "train" if mnist.train is True else "test"
        self.transform = T.PILToTensor()

    def __getitem__(self, index: int) -> Tensor:
        img, _ = cast(tuple[Image, Any], self.mnist[index])
        img_t = self.transform(img)
        assert img_t.dtype == torch.uint8
        return img_t

    def __len__(self) -> int:
        return len(self.mnist)

    def image_size(self) -> tuple[int, int, int]:
        return tuple(self[0].shape)


if __name__ == "__main__":
    train = TensorMNIST.create("train", download=True)
    test = TensorMNIST.create("test", download=True)
    assert isinstance(train[0], Tensor)
    assert isinstance(test[0], Tensor)
    assert train[0].dtype == torch.uint8
    assert test[0].dtype == torch.uint8

    logging.info(f"example size: {train.image_size()}")
    assert train.image_size() == (1, 28, 28)
