import logging

import torch
import torchvision.utils as vutils
from torch import Tensor, nn, optim
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from amls_gan import ACCELERATOR, EPOCHS, RUN_DIR
from amls_gan.datasets.module import DataModule
from amls_gan.models.dcgan import DCDiscriminator, DCGenerator

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
        image_h_w = (64, 64)

        self.datamodule = DataModule.cifar10(image_h_w=image_h_w)

        self.transforms = cifar10_transforms()

        self.device = torch.device(ACCELERATOR)
        self.amp_enabled = "cuda" in str(self.device)

        noise_dim = 100

        gen = DCGenerator(noise_dim=noise_dim).to(self.device)
        gen.init_weights_()
        self.gen = gen

        dis = DCDiscriminator().to(self.device)
        dis.init_weights_()
        self.dis = dis

        self.loss = nn.BCELoss()

        lr = 2e-4

        self.opt_gen = optim.AdamW(
            self.gen.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
            # weight_decay=1e-4,
        )

        self.opt_dis = optim.AdamW(
            self.dis.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
            # weight_decay=1e-4,
        )

    def fit(self) -> None:
        tensorboard = SummaryWriter(log_dir=RUN_DIR)

        train_dl = self.datamodule.train_dataloader()
        steps_in_epoch = len(train_dl)

        static_noise = self.gen.noise(64).to(self.device)

        for epoch in range(EPOCHS):
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

                    with autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled
                    ):
                        with torch.no_grad():
                            noise_dis = self.gen.noise(batch_size).to(self.device)
                            fake_imgs_dis = self.gen(noise_dis)

                        batch_dis = torch.cat([real_imgs, fake_imgs_dis])
                        targets_dis = torch.cat(
                            [
                                torch.ones((batch_size, 1), dtype=torch.float, device=self.device),
                                torch.zeros((batch_size, 1), dtype=torch.float, device=self.device),
                            ]
                        )

                        probs_dis = self.dis(batch_dis)
                        loss_dis = self.loss(probs_dis, targets_dis)

                    # TODO: add gradient scaling
                    self.opt_dis.zero_grad()
                    loss_dis.backward()
                    self.opt_dis.step()

                    # Train generator
                    self.gen.train()
                    self.dis.eval()

                    targets_gen = torch.ones((batch_size, 1), dtype=torch.float, device=self.device)

                    with autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled
                    ):
                        noise_gen = self.gen.noise(batch_size).to(self.device)
                        fake_imgs_gen = self.gen(noise_gen)
                        probs_gen = self.dis(fake_imgs_gen)
                        loss_gen = self.loss(probs_gen, targets_gen)

                    # TODO: add gradient scaling
                    self.opt_gen.zero_grad()
                    loss_gen.backward()
                    self.opt_gen.step()

                    # Log loss per step
                    loss_gen = loss_gen.detach().cpu().item()
                    loss_dis = loss_dis.detach().cpu().item()
                    t.set_postfix(loss_gen=f"{loss_gen:.4f}", loss_dis=f"{loss_dis:.4f}")
                    tensorboard.add_scalar("Loss/Gen", loss_gen, global_step)
                    tensorboard.add_scalar("Loss/Dis", loss_dis, global_step)

                # Log fake images per epoch
                with autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled):
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
