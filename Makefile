init:
	git pull
	pip install -Ur requirements.txt

test:
	@python -m unittest discover

lint:
	pylint initialisations

