from torch.utils.data import DataLoader, Dataset

from amls_gan.datasets.cifar10 import TensorCIFAR10
from amls_gan.datasets.mnist import TensorMNIST


class DataModule:
    def __init__(self, *, train: Dataset, test: Dataset) -> None:
        self.train = train
        self.test = test

    @classmethod
    def cifar10(cls, download: bool = False) -> "DataModule":
        return cls(
            train=TensorCIFAR10.create("train", download=download),
            test=TensorCIFAR10.create("test", download=download),
        )

    @classmethod
    def mnist(cls, download: bool = False) -> "DataModule":
        return cls(
            train=TensorMNIST.create("train", download=download),
            test=TensorMNIST.create("test", download=download),
        )

    def train_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(
            self.train,
            batch_size=128,
            shuffle=True,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader[Dataset]:
        return DataLoader(
            self.test,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
