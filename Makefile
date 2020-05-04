run:
	poetry run python3 -m cryptotax.main

swedish-taxes:
	poetry run python3 -m cryptotax.swedish_taxes

avanza:
	poetry run python3 -m cryptotax.avanza

install:
	poetry install

get_data:
	poetry run python3 -m cryptotax.download_data

test:
	env $$(cat private.env) poetry run python3 -m pytest -v cryptotax/main.py cryptotax/download_data.py cryptotax/openfigi.py cryptotax/avanza_api.py cryptotax/avanza.py

check:
	poetry run mypy --ignore-missing-imports cryptotax
	poetry run flake8 --ignore=E225,E265,E402,E501,F401,W391 cryptotax
