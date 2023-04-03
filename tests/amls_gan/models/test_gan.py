import torch

from amls_gan.models.gan import Discriminator, Generator


class TestGenerator:
    def test_noise_forward_unflatten(self) -> None:
        net = Generator(noise_dim=100, image_shape=(1, 28, 28))

        z = net.noise(5)
        assert z.shape == (5, 100)
        assert z.dtype == torch.float

        x = net(z)
        assert x.shape == (5, 1 * 28 * 28)
        assert x.dtype == torch.float

        x_img = net.unflatten(x)
        assert x_img.shape == (5, 1, 28, 28)
        assert x.dtype == torch.float

    def test_unflatten_one_image(self) -> None:
        net = Generator(noise_dim=100, image_shape=(1, 28, 28))

        x_flat = torch.rand((1 * 28 * 28))

        x = net.unflatten(x_flat)
        assert x.shape == (1, 28, 28)
        assert x.dtype == torch.float

    def test_init_weights(self) -> None:
        net = Generator(noise_dim=100, image_shape=(1, 28, 28))
        net.init_weights_()


class TestDiscriminator:
    def test_flatten_forward(self) -> None:
        net = Discriminator((1, 28, 28))

        x = torch.randn((5, 1, 28, 28))
        x_flat = net.flatten(x)

        assert x_flat.shape == (5, 1 * 28 * 28)
        assert x_flat.dtype == torch.float

        prob = net(x_flat)

        assert prob.shape == (5, 1)
        assert prob.dtype == torch.float

    def test_init_weights(self) -> None:
        net = Discriminator((1, 28, 28))
        net.init_weights_()
