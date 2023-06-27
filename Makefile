install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=main --cov=mylib test_*.py

format:	
	black *.py 

lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py mylib/*.py

refactor: 
	format lint

debug:
	python -m pytest -vv --pdb

deploy:
	#deploy goes here

all: install lint test format deploy