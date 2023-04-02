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

logging.basicConfig(level=logging.INFO)
env = Env()

Splits = Literal["train", "test"]


class TensorCIFAR10(Dataset):
    """
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
    """

    dir: Path = env.path("CIFAR10_DIR", default=Path("./data/cifar10"))

    @classmethod
    def create(cls, split: Splits, download: bool = False) -> "TensorCIFAR10":
        cls.dir.mkdir(parents=True, exist_ok=True)
        cifar = CIFAR10(root=str(cls.dir), train=split == "train", download=download)
        return cls(cifar)

    def __init__(self, cifar10: CIFAR10) -> None:
        self.cifar10 = cifar10

        self.split: Splits = "train" if cifar10.train is True else "test"
        self.transform = T.PILToTensor()

    def __getitem__(self, index: int) -> Tensor:
        img, _ = cast(tuple[Image, Any], self.cifar10[index])
        img_t = self.transform(img)
        assert img_t.dtype == torch.uint8
        return img_t

    def __len__(self) -> int:
        return len(self.cifar10)


if __name__ == "__main__":
    train = TensorCIFAR10.create("train", download=True)
    test = TensorCIFAR10.create("test", download=True)
    assert isinstance(train[0], Tensor)
    assert isinstance(test[0], Tensor)
    assert train[0].dtype == torch.uint8
    assert test[0].dtype == torch.uint8
    logging.info(f"example size: {train[0].shape}")
