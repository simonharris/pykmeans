default:
	@echo "No default make target"

init:
	git pull
	pip install -Ur requirements.txt

test:
	@python -m unittest discover

lint:
	pylint initialisations

lintall:
	find . -iname "*.py" | xargs pylint

exp:
	@python runner.py

expclean:
	rm -rf _output/out*
