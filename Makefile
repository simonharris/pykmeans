init:
	pip install -qr requirements.txt

test:
	python -m unittest discover
	
