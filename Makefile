run:
	python3 -m cryptotax.main

swedish-taxes:
	python3 -m cryptotax.swedish_taxes

avanza:
	python3 -m cryptotax.avanza

install:
	pip3 install . -r requirements.txt

get_data:
	python3 download_data.py

test:
	env $$(cat private.env) pipenv run python3 -m pytest -v cryptotax/main.py cryptotax/download_data.py cryptotax/openfigi.py

check:
	mypy --ignore-missing-imports cryptotax *.py
	flake8 --ignore=E225,E265,E402,E501,F401,W391
