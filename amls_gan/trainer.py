import logging

import torch
from torch import Tensor, nn, optim
from torchvision import transforms as T

from amls_gan.datasets.module import DataModule
from amls_gan.models.gan import Discriminator, Generator

logging.basicConfig(level=logging.INFO)


def mnist_transforms() -> T.Compose:
    return T.Compose(
        [
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.5], [0.5]),
        ]
    )


def cifar10_transforms() -> T.Compose:
    return T.Compose(
        [
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class Trainer:
    def __init__(self) -> None:
        self.datamodule = DataModule.mnist()

        self.transforms = mnist_transforms()

        self.device = torch.device("mps")

        image_shape = self.datamodule.image_size()

        noise_dim = 100

        gen = Generator(noise_dim=noise_dim, image_shape=image_shape).to(self.device)
        gen.init_weights_()
        self.gen = gen

        dis = Discriminator(image_shape).to(self.device)
        dis.init_weights_()
        self.dis = dis

        self.loss = nn.BCELoss()

        lr = 1e-6

        self.opt_gen = optim.AdamW(
            self.gen.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

        self.opt_dis = optim.AdamW(
            self.dis.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

        self.epochs = 1

    def fit(self) -> None:
        train_dl = self.datamodule.train_dataloader()
        for epoch in range(self.epochs):
            real_imgs: Tensor
            for i, real_imgs in enumerate(train_dl):
                batch_size = len(real_imgs)

                # Train discriminator
                self.gen.eval()
                self.dis.train()

                real_imgs = self.transforms(real_imgs)
                real_imgs = real_imgs.to(self.device)

                real_flat_imgs = self.dis.flatten(real_imgs)

                with torch.no_grad():
                    noise_dis = self.gen.noise(batch_size).to(self.device)
                    fake_flat_imgs_dis = self.gen(noise_dis)

                batch_dis = torch.cat([real_flat_imgs, fake_flat_imgs_dis])
                targets_dis = torch.cat(
                    [
                        torch.ones((batch_size, 1), dtype=torch.float, device=self.device),
                        torch.zeros((batch_size, 1), dtype=torch.float, device=self.device),
                    ]
                )

                probs_dis = self.dis(batch_dis)
                loss_dis = self.loss(probs_dis, targets_dis)

                self.opt_dis.zero_grad()
                loss_dis.backward()
                self.opt_dis.step()

                # Train generator
                self.gen.train()
                self.dis.eval()

                targets_gen = torch.ones((batch_size, 1), dtype=torch.float, device=self.device)

                noise_gen = self.gen.noise(batch_size).to(self.device)
                fake_flat_imgs_gen = self.gen(noise_gen)
                probs_gen = self.dis(fake_flat_imgs_gen)
                loss_gen = self.loss(probs_gen, targets_gen)

                self.opt_gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                logging.info(f"Dis: {loss_dis.detach().cpu()}, Gen: {loss_gen.detach().cpu()}")


def run() -> None:
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    run()
