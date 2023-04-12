from typing import Type

from torch import Tensor
from torch.utils.data import DataLoader

from amls_gan.datasets import TensorCelebA, TensorCIFAR10, TensorMNIST

MyDatasets = TensorCIFAR10 | TensorMNIST | TensorCelebA


class DataModule:
    def __init__(self, *, train: MyDatasets, test: MyDatasets) -> None:
        self.train = train
        self.test = test

    @classmethod
    def create(cls, ds_t: Type[MyDatasets]) -> "DataModule":
        return cls(
            train=ds_t.create("train"),
            test=ds_t.create("test"),
        )

    def image_size(self) -> tuple[int, int, int]:
        return self.train.image_size()

    def train_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            self.train,
            batch_size=128,
            shuffle=True,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            self.test,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
