# GAN

## Training

Clone the repo and execute the following from its root:
```
$ pip install .
$ make download.cifar10
$ make fit ACCELERATOR=[cpu|cuda|mps]
```

To see the progress in the Tensorboard during training:
```
make tensorboard
```
