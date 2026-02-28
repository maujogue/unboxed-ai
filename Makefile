.PHONY: help install sync test pytest unittest orthank-test lint format typecheck check clean

help: ## Show available commands
	@echo "Common commands:"
	@echo "  make install      - Sync dependencies (including dev)"
	@echo "  make test         - Run full pytest suite"
	@echo "  make pytest       - Run full pytest suite (alias)"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Run ruff formatter"
	@echo "  make typecheck    - Run type checker"
	@echo "  make check        - Run lint + typecheck + tests"

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
