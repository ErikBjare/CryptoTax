run:
	python3 -m cryptotax.main

swedish-taxes:
	python3 -m cryptotax.swedish_taxes

install:
	pip3 install . -r requirements.txt

get_data:
	python3 download_data.py

test:
	python3 -m pytest -v cryptotax/*.py

check:
	mypy --ignore-missing-imports cryptotax *.py
	flake8 --ignore=E225,E265,E402,E501,F401,W391
