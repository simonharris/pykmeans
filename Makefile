init:
	pip install -qr requirements.txt

test:
	@python -m unittest discover

experiments:
	@python experiments.py -d ${DS} -k ${K} -a ${A}
