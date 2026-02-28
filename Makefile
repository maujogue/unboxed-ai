.PHONY: help install sync test pytest unittest orthank-test lint format typecheck check clean docker-up docker-down docker-logs pipeline run-frontend frontend-build

help: ## Show available commands
	@echo "Common commands:"
	@echo "  make install      - Sync dependencies (including dev)"
	@echo "  make test         - Run full pytest suite"
	@echo "  make pytest       - Run full pytest suite (alias)"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Run ruff formatter"
	@echo "  make typecheck    - Run type checker"
	@echo "  make check        - Run lint + typecheck + tests"
	@echo "  make docker-up    - Start pgvector Postgres (docker compose up -d)"
	@echo "  make docker-down  - Stop pgvector Postgres"
	@echo "  make docker-logs  - Show pgvector container logs"
	@echo "  make pipeline     - Run SEG extraction pipeline + generate HTML report"
	@echo "  make run-frontend  - Run experiences frontend (Orthanc + reports) on port 8000"
	@echo "  make frontend-build - Build React frontend for production"

install: ## Install and sync project dependencies
	uv sync --dev

test: pytest ## Default test target

pytest: ## Run pytest suite
	uv run pytest -q

lint: ## Run linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .

typecheck: ## Run type checking
	uv run ty check

check: ## Run lint, typecheck and tests
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

docker-up: ## Start pgvector Postgres
	docker compose up -d

docker-down: ## Stop pgvector Postgres
	docker compose down

docker-logs: ## Show pgvector container logs
	docker compose logs -f pgvector

pipeline: ## Extract SEG from bundled registry, upload to Orthanc, generate HTML report
	uv run orthanc-pipeline --output results/ --report results/rapport.html

run-frontend: ## Run experiences frontend (Orthanc studies + reports)
	uv run uvicorn unboxed_ai.experiences_api:app --reload --host 0.0.0.0 --port 8000

frontend-build: ## Build React frontend for production
	cd frontend && npm install && npm run build
