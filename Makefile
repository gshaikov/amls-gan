datetime := $(shell date -Iseconds)

export RUN_DIR := $(shell pwd)/runs/$(datetime)
export DATASETS_DIR := $(HOME)/datasets

export ACCELERATOR ?= mps
export EPOCHS ?= 100

download.%:
	python -m amls_gan.datasets.$*

fit:
	python -m amls_gan.trainer

tensorboard:
	tensorboard --logdir=$(dir $(RUN_DIR)) --port=0

clean:
	rm -rf $(dir $(RUN_DIR))*

test:
	pytest tests

lint:
	black --check amls_gan tests
	isort --check amls_gan tests
	flake8 amls_gan tests
