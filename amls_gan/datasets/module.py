from torch import Tensor
from torch.utils.data import DataLoader

from amls_gan.datasets.cifar10 import TensorCIFAR10
from amls_gan.datasets.mnist import TensorMNIST

MyDatasets = TensorCIFAR10 | TensorMNIST


class DataModule:
    def __init__(self, *, train: MyDatasets, test: MyDatasets) -> None:
        self.train = train
        self.test = test

    @classmethod
    def cifar10(cls, download: bool = False) -> "DataModule":
        """
        Args:
            download (optional): if True, download dataest if it's not yet done.
        """
        return cls(
            train=TensorCIFAR10.create("train", download=download),
            test=TensorCIFAR10.create("test", download=download),
        )

    @classmethod
    def mnist(cls, download: bool = False) -> "DataModule":
        """
        Args:
            download (optional): if True, download dataest if it's not yet done.
        """
        return cls(
            train=TensorMNIST.create("train", download=download),
            test=TensorMNIST.create("test", download=download),
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
