run:
	pipenv run python3 -m cryptotax.main

swedish-taxes:
	pipenv run python3 -m cryptotax.swedish_taxes

avanza:
	pipenv run python3 -m cryptotax.avanza

install:
	pipenv install . -r requirements.txt

get_data:
	python3 download_data.py

test:
	env $$(cat private.env) pipenv run python3 -m pytest -v cryptotax/main.py cryptotax/download_data.py cryptotax/openfigi.py cryptotax/avanza_api.py cryptotax/avanza.py

check:
	pipenv run mypy --ignore-missing-imports cryptotax
	pipenv run flake8 --ignore=E225,E265,E402,E501,F401,W391 cryptotax
