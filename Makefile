datetime := $(shell date -Iseconds)

export RUN_DIR := $(shell pwd)/runs/$(datetime)
export DATASETS_DIR := $(HOME)/datasets

download.%:
	python -m amls_gan.datasets.$*

fit:
	python -m amls_gan.trainer

clean:
	rm -rf $(dir $(RUN_DIR))*

test:
	pytest tests

lint:
	black --check amls_gan tests
	isort --check amls_gan tests
	flake8 amls_gan tests
