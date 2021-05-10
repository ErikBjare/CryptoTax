run:
	poetry run python3 -m cryptotax.main

install:
	poetry install

get_data:
	poetry run python3 -m cryptotax.download_data

test:
	env $$(cat private.env) poetry run pytest

check:
	poetry run mypy --ignore-missing-imports cryptotax
	poetry run flake8 --ignore=E225,E265,E402,E501,F401,W391,W503 cryptotax
