init:
	git pull
	pip install -qr requirements.txt

test:
	@python -m unittest discover

lint:
	pylint initialisations

