datetime := $(shell date -Iseconds)

export RUN_DIR := $(shell pwd)/runs/$(datetime)
export DATASETS_DIR := $(HOME)/datasets

download.%:
	python -m amls_gan.datasets.$*

fit:
	python -m amls_gan.trainer

clean:
	rm -rf $(dir $(RUN_DIR))*
