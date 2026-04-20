.PHONY: setup lint test run-clip run-ui

setup:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

run-clip:
	python main.py

run-ui:
	streamlit run streamlit_app.py
