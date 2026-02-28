.PHONY: help install sync test pytest unittest orthank-test lint format typecheck check clean docker-up docker-down docker-logs

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
