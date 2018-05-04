run:
	python3 main.py

swedish-taxes:
	python3 swedish_taxes.py

install:
	pip3 install -r requirements.txt

get_data:
	python3 download_data.py

test:
	python3 -m pytest -v *.py

check:
	mypy --ignore-missing-imports *.py
	flake8 --ignore=E225,E265,E402,E501
