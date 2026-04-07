.PHONY: install sort bin reduce train serve test lint typecheck

install:
	pip install -e ".[dev]"
	pre-commit install

sort:
	PYTHONPATH=. python flows/training_flow.py --stage sort

bin:
	PYTHONPATH=. python flows/training_flow.py --stage bin

reduce:
	PYTHONPATH=. python flows/training_flow.py --stage reduce

train:
	PYTHONPATH=. python flows/training_flow.py --stage all

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ --cov=src --cov-report=term-missing -v

lint:
	ruff check src/ tests/

typecheck:
	mypy src/
