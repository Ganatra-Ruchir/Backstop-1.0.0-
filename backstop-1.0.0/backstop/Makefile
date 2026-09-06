.PHONY: help install data eda compare train evaluate figures serve test lint all

help:
	@echo "make install    create a virtualenv and install everything"
	@echo "make data       download and verify the dataset (144 MB)"
	@echo "make all        the full pipeline: eda -> compare -> train -> figures"
	@echo "make serve      run the scoring API on :8000"
	@echo "make test       run the test suite"
	@echo "make lint       ruff"

install:
	python -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements-dev.txt

data:
	python scripts/download_data.py

eda:
	python scripts/eda.py

compare:
	python scripts/compare_models.py

train:
	python scripts/train.py

evaluate:
	python scripts/operating_points.py
	python scripts/error_analysis.py
	python scripts/leakage_demo.py
	python scripts/drift_report.py
	python scripts/explain_model.py

figures:
	python scripts/make_figures.py

serve:
	uvicorn backstop.serving.app:app --host 0.0.0.0 --port 8000 --app-dir src

test:
	pytest tests/ -q

test-fast:
	pytest tests/ -q -m "not slow"

lint:
	ruff check .

bench:
	python scripts/benchmark_service.py

all: eda compare train evaluate figures
