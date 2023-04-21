import logging
from pathlib import Path

import torch
import torchvision.utils as vutils
from environs import Env
from torch import Tensor, nn, optim
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from amls_gan.datasets import Datasets
from amls_gan.datasets.module import DataModule
from amls_gan.models.dcgan import DCDiscriminator, DCGenerator, ModelStats
from amls_gan.models.transforms import celeba_transforms

env = Env()

RUN_DIR: Path = env.path("RUN_DIR")
ACCELERATOR: str = env.str("ACCELERATOR")
EPOCHS: int = env.int("EPOCHS")

logging.basicConfig(level=logging.DEBUG)


class Trainer:
    def __init__(self) -> None:
        # Config
        image_h_w = (64, 64)
        noise_dim = 100
        feat_maps = 64
        batch_size = 128
        learning_rate = 2e-4

        self.datamodule = DataModule.create(Datasets.TensorCelebA, batch_size=batch_size)

        self.transforms = celeba_transforms(image_h_w)

        self.device = torch.device(ACCELERATOR)
        self.amp_enabled = "cuda" in str(self.device)

        gen = DCGenerator(noise_dim=noise_dim, feat_maps=feat_maps).to(self.device)
        gen.init_weights_()
        self.gen = gen

        dis = DCDiscriminator(feat_maps=feat_maps).to(self.device)
        dis.init_weights_()
        self.dis = dis

        self.loss = nn.BCELoss(reduction="mean")

        self.opt_gen = optim.AdamW(
            self.gen.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
            # weight_decay=1e-4,
        )

        self.opt_dis = optim.AdamW(
            self.dis.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
            # weight_decay=1e-4,
        )

    def fit(self) -> None:
        tensorboard = SummaryWriter(log_dir=RUN_DIR)

        train_dl = self.datamodule.train_dataloader()

        real_batch = self.transforms(next(iter(train_dl)))
        real_grid = vutils.make_grid(real_batch[:64], nrow=8, padding=2, normalize=True)
        tensorboard.add_image("Images/Real", real_grid, 0)

        steps_in_epoch = len(train_dl)

        static_noise = self.gen.noise(64).to(self.device)

        global_step = 0

        real_imgs: Tensor
        for epoch in range(EPOCHS):
            with tqdm(train_dl) as t:
                t.set_description(f"Epoch: {epoch}")

                for inner_step, real_imgs in enumerate(t):
                    global_step = epoch * steps_in_epoch + inner_step

                    batch_size = len(real_imgs)

                    # Train discriminator
                    self.gen.train()
                    self.dis.train()

                    real_imgs = self.transforms(real_imgs).to(self.device)

                    with autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled
                    ):
                        with torch.no_grad():
                            noise_dis = self.gen.noise(batch_size).to(self.device)
                            fake_imgs_dis: Tensor = self.gen(noise_dis)

                        # Separate passes for fake and real images.
                        # Batchnorm will normalise over all examples per channel, however fake and
                        # real images come from different distributions in the beginning of
                        # training, which means that discriminator will have an easy job separating
                        # them.
                        # The result is that discriminator error is low.
                        # During Generator update, Distriminator will see only fake images,
                        # and batchnorm will normalise the distribution which will make it look
                        # *different* from the normalised distribution of fake images during
                        # Discriminator update.
                        # Discriminator will have hard time classifying the images and Generator
                        # error will be low.
                        # The result is that neither Discriminator nor Generator are learning.
                        probs_dis_real: Tensor = self.dis(real_imgs)
                        probs_dis_fake: Tensor = self.dis(fake_imgs_dis)

                        probs_dis = torch.cat([probs_dis_real, probs_dis_fake])
                        targets_dis = torch.cat(
                            [
                                torch.ones((batch_size, 1), dtype=torch.float, device=self.device),
                                torch.zeros((batch_size, 1), dtype=torch.float, device=self.device),
                            ]
                        )

                        # multiply by 2 since (Goodfellow et al. 2014) sums the losses
                        loss_dis: Tensor = self.loss(probs_dis, targets_dis) * 2

                    # TODO: add gradient scaling
                    self.opt_dis.zero_grad()
                    loss_dis.backward()
                    self.opt_dis.step()

                    # Train generator
                    self.gen.train()
                    self.dis.train()

                    targets_gen = torch.ones((batch_size, 1), dtype=torch.float, device=self.device)

                    with autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled
                    ):
                        noise_gen = self.gen.noise(batch_size).to(self.device)
                        fake_imgs_gen: Tensor = self.gen(noise_gen)
                        probs_gen: Tensor = self.dis(fake_imgs_gen)
                        loss_gen: Tensor = self.loss(probs_gen, targets_gen)

                    # TODO: add gradient scaling
                    self.opt_gen.zero_grad()
                    loss_gen.backward()
                    self.opt_gen.step()

                    # Log metrics
                    loss_gen_v = loss_gen.detach().cpu().item()
                    loss_dis_v = loss_dis.detach().cpu().item()
                    t.set_postfix(loss_gen=f"{loss_gen_v:.2E}", loss_dis=f"{loss_dis_v:.2E}")

                    if global_step % 50 == 0:
                        tensorboard.add_scalar("Loss/Gen", loss_gen_v, global_step)
                        tensorboard.add_scalar("Loss/Dis", loss_dis_v, global_step)

                        weights_gen = ModelStats.weights(self.gen, ["ConvTranspose2d"])
                        weights_dis = ModelStats.weights(self.dis, ["Conv2d"])
                        tensorboard.add_scalars("ConvWeights/Gen", weights_gen, global_step)
                        tensorboard.add_scalars("ConvWeights/Dis", weights_dis, global_step)

                        grads_gen = ModelStats.grads(self.gen, ["ConvTranspose2d"])
                        grads_dis = ModelStats.grads(self.dis, ["Conv2d"])
                        tensorboard.add_scalars("ConvGrads/Gen", grads_gen, global_step)
                        tensorboard.add_scalars("ConvGrads/Dis", grads_dis, global_step)

                        mean_probs = {
                            "gen_fake": probs_gen.mean().detach().cpu(),
                            "dis_real": probs_dis_real.mean().detach().cpu(),
                            "dis_fake": probs_dis_fake.mean().detach().cpu(),
                        }
                        tensorboard.add_scalars("MeanProbs", mean_probs, global_step)

                        with autocast(
                            device_type="cuda", dtype=torch.float16, enabled=self.amp_enabled
                        ):
                            self.gen.eval()
                            with torch.no_grad():
                                fake_imgs_static = self.gen(static_noise).detach().cpu()
                        fake_grid = vutils.make_grid(
                            fake_imgs_static, nrow=8, padding=2, normalize=True
                        )
                        tensorboard.add_image("Images/Fake", fake_grid, global_step)

        tensorboard.close()


def run() -> None:
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    run()
