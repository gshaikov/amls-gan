import logging

import torch
import torchvision.utils as vutils
from torch import Tensor, nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from amls_gan import RUN_DIR
from amls_gan.datasets.module import DataModule
from amls_gan.models.gan import Discriminator, Generator

logging.basicConfig(level=logging.DEBUG)


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

        self.epochs = 100

    def fit(self) -> None:
        tensorboard = SummaryWriter(log_dir=RUN_DIR)

        train_dl = self.datamodule.train_dataloader()
        steps_in_epoch = len(train_dl)

        static_noise = self.gen.noise(64).to(self.device)

        for epoch in range(self.epochs):
            real_imgs: Tensor

            with tqdm(train_dl) as t:
                t.set_description(f"Epoch: {epoch}")

                for inner_step, real_imgs in enumerate(t):
                    global_step = epoch * steps_in_epoch + inner_step

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

                    # Log loss per step
                    loss_gen = loss_gen.detach().cpu().item()
                    loss_dis = loss_dis.detach().cpu().item()
                    t.set_postfix(loss_gen=loss_gen, loss_dis=loss_dis)
                    tensorboard.add_scalar("Loss/Gen", loss_gen, global_step)
                    tensorboard.add_scalar("Loss/Dis", loss_dis, global_step)

                # Log fake images per epoch
                with torch.no_grad():
                    fake_imgs_static = self.gen(static_noise).cpu()
                grid = vutils.make_grid(fake_imgs_static, nrow=8, padding=2, normalize=True)
                tensorboard.add_image("image", grid, epoch)

        tensorboard.close()


def run() -> None:
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    run()
