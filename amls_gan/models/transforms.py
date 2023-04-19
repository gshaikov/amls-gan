from typing import Any

import torch
from torchvision import transforms as T


def mnist_transforms(*args: Any) -> T.Compose:
    return T.Compose(
        [
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.5], [0.5]),
        ]
    )


def cifar10_transforms(img_h_w: tuple[int, int]) -> T.Compose:
    return T.Compose(
        [
            T.Resize(img_h_w, antialias=True),  # type: ignore
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def celeba_transforms(img_h_w: tuple[int, int]) -> T.Compose:
    return T.Compose(
        [
            T.Resize(img_h_w, antialias=True),  # type: ignore
            T.CenterCrop(img_h_w),
            T.ConvertImageDtype(torch.float),  # Will covert [0, 255] to [0, 1]
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
