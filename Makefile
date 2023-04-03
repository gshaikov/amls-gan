download.%:
	python -m amls_gan.datasets.$*

fit:
	python -m amls_gan.trainer
