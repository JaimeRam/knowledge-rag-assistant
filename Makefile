.PHONY: help setup install services services-down ingest run test clean

help:
	@echo "Digimon RAG Assistant - Available commands:"
	@echo "  make setup         - Create venv and install dependencies"
	@echo "  make install       - Install dependencies in existing venv"
	@echo "  make services      - Start Docker infrastructure services"
	@echo "  make services-down - Stop Docker infrastructure services"
	@echo "  make ingest        - Run data ingestion pipeline"
	@echo "  make run           - Start FastAPI application"
	@echo "  make test          - Run unit test suite (no external services needed)"
	@echo "  make clean         - Remove venv and __pycache__"

setup:
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Run: source venv/bin/activate"

install:
	pip install -r requirements.txt

services:
	cd docker && docker compose up -d

services-down:
	cd docker && docker compose down

ingest:
	python -m app.ingestion.ingest

run:
	python -m app.api.main

test:
	pytest tests/ -v --ignore=tests/test_ingestion.py

clean:
	rm -rf venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
