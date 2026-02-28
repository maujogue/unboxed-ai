# Project Agent Workflow

This project follows these implementation and tooling conventions:

- Use `LangGraph` for graph orchestration and agents.
- Use `langchain-mistralai` for LLM and embeddings integrations.
- Use `langfuse` for LLM tracing and observability.
- Use `pgvector` (via Docker Compose) as the embeddings/vector store backend.
- Use `uv` for dependency management and command execution.
- Use `pytest` for tests.
- Use `ruff` for linting and formatting.
- Use `ty` for type checking.
- Use the `Makefile` as the primary command interface.

## Standard Commands

Prefer these project commands:

- `make install` to sync dependencies.
- `make test` (or `make pytest`) to run tests.
- `make lint` to run Ruff checks.
- `make format` to format with Ruff.
- `make typecheck` to run `ty`.
- `make check` to run lint + type-check + tests.

## Quality Gate (Required)

After each change, run quality checks before considering work done:

1. `make format` (when code style may have changed)
2. `make check`

If any step fails, fix issues and re-run until all checks pass.
