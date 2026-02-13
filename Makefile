.PHONY: install lint format test jupyter clean

install:
	poetry install

lint:
	poetry run ruff check labs/
	poetry run black --check labs/

format:
	poetry run black labs/
	poetry run ruff check --fix labs/

test:
	poetry run pytest -v

jupyter:
	poetry run jupyter lab --notebook-dir=labs

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
