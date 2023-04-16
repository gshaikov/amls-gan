from typing import Any, Type

from torch import Tensor
from torch.utils.data import DataLoader

from amls_gan.datasets import TensorCelebA, TensorCIFAR10, TensorMNIST

MyDatasets = TensorCIFAR10 | TensorMNIST | TensorCelebA


class DataModule:
    def __init__(self, *, train: MyDatasets, test: MyDatasets, **dl_kwargs: Any) -> None:
        self.train = train
        self.test = test

        self.dl_kwargs = dict(
            batch_size=128,
            num_workers=0,
            **dl_kwargs,
        )

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
            shuffle=True,
            **self.dl_kwargs,
        )

    def test_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            self.test,
            shuffle=False,
            **self.dl_kwargs,
        )
