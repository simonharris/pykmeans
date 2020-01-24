default:
	@echo "No default make target"

init:
	git pull
	pip install -Ur requirements.txt

test:
	@python3 -m unittest discover

lint:
	pylint initialisations

lintall:
	find . -iname "*.py" | grep -v "_deprecated" | xargs pylint

exp:
	python3 runner.py

expclean:
	rm -rf _output/out*
