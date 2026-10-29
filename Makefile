.PHONY: help install install-dev test lint format clean evaluate run-all

help:
	@echo "Available commands:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development and testing dependencies"
	@echo "  make test          Run unit test suite"
	@echo "  make lint          Run linter checks"
	@echo "  make format        Auto-format code with black and ruff"
	@echo "  make evaluate      Run connotation evaluation against lexicon"
	@echo "  make clean         Remove build artifacts and cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,training]"

test:
	PYTHONPATH=src python3 -m unittest discover tests -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

evaluate:
	PYTHONPATH=src python3 -m nlp_connotation.cli evaluate --lexicon data/lexicons/my_Lexicon.csv --output-dir data/results

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
