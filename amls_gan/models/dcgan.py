"""
Notes:

- images are zero mean and range of hyperbolic tangent
- multi-res discriminator
- hinge loss
- make sure discriminator is not too small

"""

import logging
from typing import Any, Callable

import torch
from torch import Tensor, nn


class DCGenerator(nn.Module):
    def __init__(self, *, noise_dim: int) -> None:
        super().__init__()
        self.noise_dim = noise_dim

        # Hout = (Hin - 1) * S + K - 2 * P
        self.layers = nn.Sequential(
            # (N, 100) -> (N, 100, 1, 1)
            nn.Unflatten(1, (self.noise_dim, 1, 1)),
            # Layer 1: (100, 1, 1) -> (1024, 4, 4)
            nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # Layer 2: (1024, 4, 4) -> (512, 8, 8)
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Layer 3: (512, 8, 8) -> (256, 16, 16)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 4: (256, 16, 16) -> (128, 32, 32)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 5: (128, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def _init_weights_fn(module: nn.Module) -> None:
        name = module.__class__.__name__
        if name.find("Conv") != -1:
            assert isinstance(module, nn.ConvTranspose2d)
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            logging.info(f"Gen init: {name}")
        elif name.find("BatchNorm") != -1:
            assert isinstance(module, nn.BatchNorm2d)
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
            logging.info(f"Gen init: {name}")

    def init_weights_(self) -> None:
        self.apply(self._init_weights_fn)

    def noise(self, N: int) -> Tensor:
        """
        Random uniform noise in [-1, 1] as (N, Z) tensor.
        """
        return torch.randn((N, self.noise_dim))

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: noise float tensor
                shape: (N, Z)

        Returns:
            normalised image as float tensor
                shape: (N, C, H, W)
        """
        x = self.layers(z)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Hout = ceil[(Hin - K + 2 * P) / 2] + 1
        self.layers = nn.Sequential(
            # Layer 1: (3, 64, 64) -> (128, 32, 32)
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # Layer 2: (128, 32, 32) -> (256, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # Layer 3: (256, 16, 16) -> (512, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # Layer 4: (512, 8, 8) -> (1024, 4, 4)
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            # Layer 5: (1024, 4, 4) -> (1, 1, 1)
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # (N, 1, 1, 1) -> (N, 1)
            nn.Flatten(1),
        )

    @staticmethod
    def _init_weights_fn(module: nn.Module) -> None:
        name = module.__class__.__name__
        if name.find("Conv2d") != -1:
            assert isinstance(module, nn.Conv2d)
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            logging.info(f"Dis init: {name}")
        elif name.find("BatchNorm") != -1:
            assert isinstance(module, nn.BatchNorm2d)
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
            logging.info(f"Dis init: {name}")

    def init_weights_(self) -> None:
        self.apply(self._init_weights_fn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: normalised image as float tensor
                shape: (N, C, H, W)

        Returns:
            tensor of probabilities [0, 1]
                shape: (N, 1)
        """
        p = self.layers(x)
        return p


class _PrintLambda(nn.Module):
    def __init__(self, fn: Callable[[Tensor], Any]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        logging.debug(f"PrintLambda: {self.fn(x)}")
        return x
