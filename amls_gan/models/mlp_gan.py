import logging

import torch
from torch import Tensor, nn

from amls_gan._math import prod


class MLPGenerator(nn.Module):
    def __init__(self, *, noise_dim: int, image_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.image_shape = image_shape

        self.output_dim = prod(image_shape)
        self.layers = nn.Sequential(
            nn.Linear(noise_dim, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1200, self.output_dim),
            nn.Tanh(),
        )

    @staticmethod
    def _init_weights_fn(module: nn.Module) -> None:
        name = module.__class__.__name__
        if name.find("Linear") != -1:
            assert isinstance(module, nn.Linear)
            nn.init.uniform_(module.weight.data, -0.05, 0.05)
            logging.info(f"Gen init: {name}")
        elif name.find("BatchNorm") != -1:
            assert isinstance(module, nn.BatchNorm1d)
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
            logging.info(f"Gen init: {name}")

    def init_weights_(self) -> None:
        self.apply(self._init_weights_fn)

    def noise(self, N: int) -> Tensor:
        return torch.randn((N, self.noise_dim))

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: noise float tensor
                shape: (N, Z)

        Returns:
            flattened normalised image as float tensor
                shape: (N, C*H*W)
        """
        x = self.layers(z)
        return x

    def unflatten(self, x: Tensor) -> Tensor:
        """
        Args:
            x: flattened normalised image as float tensor
                shape: (N, C*H*W)

        Returns:
            normalised image as float tensor
                shape: (N, C, H, W)
        """
        total_dims = len(x.shape)
        return x.unflatten(dim=total_dims - 1, sizes=self.image_shape)


class MLPDiscriminator(nn.Module):
    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.image_shape = image_shape

        self.input_dim = prod(image_shape)
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1200, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _init_weights_fn(module: nn.Module) -> None:
        name = module.__class__.__name__
        if name.find("Linear") != -1:
            assert isinstance(module, nn.Linear)
            nn.init.uniform_(module.weight.data, -0.05, 0.05)
            logging.info(f"Dis init: {name}")
        elif name.find("BatchNorm") != -1:
            assert isinstance(module, nn.BatchNorm1d)
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
            logging.info(f"Dis init: {name}")

    def init_weights_(self) -> None:
        self.apply(self._init_weights_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: flattened normalised image as float tensor
                shape: (N, C*H*W)

        Returns:
            tensor of probabilities [0, 1]
                shape: (N, 1)
        """
        p = self.layers(x)
        return p

    def flatten(self, x: Tensor) -> Tensor:
        """
        Args:
            x: normalised image as float tensor
                shape: (N, C, H, W)

        Returns:
            flattened normalised image as float tensor
                shape: (N, C*H*W)
        """
        total_dims = len(x.shape)
        return x.flatten(start_dim=total_dims - 3, end_dim=-1)
