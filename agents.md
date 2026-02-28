# Project Agent Workflow

This project follows these implementation and tooling conventions:

- Use `LangGraph` for graph orchestration and agents.
- Use `langchain-mistralai` for LLM and embeddings integrations.
- Use `langfuse` for LLM tracing and observability.
- Use `pgvector` (via Docker Compose) as the embeddings/vector store backend.
- Use `SQLAlchemy` for sync Postgres engine/session and pgvector extension setup.
- Use `uv` for dependency management and command execution.
- Use `pytest` for tests.
- Use `ruff` for linting and formatting.
- Use `ty` for type checking.
- Use the `Makefile` as the primary command interface.
- **After each run**: run linting, type checking, and tests (`make format` then `make check`); see Quality Gate below.

## Constants (`src/unboxed_ai/lib/Constants.py`)

Central env-driven configuration:

- **Orthanc**: `ORTHANC_URL`, `ORTHANC_USERNAME`, `ORTHANC_PASSWORD`; derived endpoints for instances, studies, archive.
- **Mistral**: `MISTRAL_API_KEY_ENV` (name of the env var), `MISTRAL_API_KEY`; `DEFAULT_MISTRAL_MODEL`, `DEFAULT_MISTRAL_TEMPERATURE`.
- **PostgreSQL / pgvector**: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`. If `PGVECTOR_CONNECTION` is set it is used as-is; otherwise the connection URL is built from the POSTGRES_* vars. `PGVECTOR_COLLECTION_NAME` (default `DEFAULT_PGVECTOR_COLLECTION_NAME` = `unboxed_ai`).
- **Timeouts**: `DEFAULT_TIMEOUT_SECONDS`, `DOWNLOAD_TIMEOUT_SECONDS`, `DEFAULT_WORK_PATH`.

## Services (`src/unboxed_ai/lib/Services.py`)

Single registry for app-level clients; all properties are `@cached_property` (lazy, cached):

- **orthanc** – `OrthancClient` (base URL, auth, timeouts from Constants).
- **mistral_llm** – `ChatMistralAI` (model, API key, temperature from Constants).
- **mistral_embeddings** – `MistralAIEmbeddings` (API key from Constants).
- **pgvector** – `langchain_postgres.PGVector` (embeddings = mistral_embeddings, connection and collection name from Constants). Use for low-level vector store access.
- **sqlalchemy_engine** – Sync SQLAlchemy engine from `create_engine_from_constants()` (same URL as pgvector, `pool_pre_ping=True`).
- **vector_store** – `VectorStoreService` wrapping `pgvector` + `sqlalchemy_engine`: extension setup, ingestion, retrieval.
- **langfuse_callbacks** – LangChain callbacks for Langfuse tracing.

## PostgreSQL + pgvector

- **Docker Compose** (`docker-compose.yml`): service `pgvector` (image `pgvector/pgvector:pg16`), port 5432, volume `pgvector_data`, healthcheck via `pg_isready`. Start with `make docker-up`; stop with `make docker-down`; logs with `make docker-logs`.
- **VectorStoreService** (`src/unboxed_ai/lib/vector_store.py`):
  - `ensure_pgvector_ready()` – runs `CREATE EXTENSION IF NOT EXISTS vector` using the SQLAlchemy engine.
  - `ingest_documents(documents, ids=None)` – add LangChain documents; returns list of ids.
  - `ingest_texts(texts, metadatas=None, ids=None)` – add raw texts (optional metadata/ids); returns list of ids.
  - `get_retriever(k=4, search_type="similarity")` – returns a LangChain retriever over the store.
  - `similarity_search(query, k=4)` – returns top-k documents for a query.
- **SQLAlchemy**: `create_engine_from_constants()` builds the sync engine; the service uses it for extension setup and keeps a `sessionmaker` (sync, no autocommit/autoflush). The vector store itself is `langchain_postgres.PGVector`; SQLAlchemy is used only for extension and future DB operations.

## Public API (`src/unboxed_ai/lib/__init__.py`)

Exports: `Constants`, `OrthancClient`, `Services`, `VectorStoreService`, `create_engine_from_constants`, `show_dicom`, `flush_langfuse`, `get_langfuse_langchain_callbacks`.

## Example and env

- **Example**: `examples/pgvector_ingest_and_retrieve.py` – loads env, waits for DB, ingests sample texts, runs similarity search and prints results. Run after `make docker-up` and with `MISTRAL_API_KEY` set.
- **Env**: Copy `.env.example` and set at least `MISTRAL_API_KEY` (and Langfuse keys if tracing). Optional: `POSTGRES_*` / `PGVECTOR_CONNECTION`, `PGVECTOR_COLLECTION_NAME` (defaults match docker-compose).

## Standard Commands

Prefer these project commands:

- `make install` – sync dependencies.
- `make test` (or `make pytest`) – run tests (unit tests only; integration tests are marked and deselected by default).
- `make lint` – Ruff checks.
- `make format` – Ruff format.
- `make typecheck` – ty.
- `make check` – lint + typecheck + tests.
- `make docker-up` – start pgvector Postgres.
- `make docker-down` – stop pgvector Postgres.
- `make docker-logs` – pgvector container logs.

## Quality Gate (Required)

**After each run or change**, the agent must run linting, type checking, and tests before considering work done:

1. `make format` when code style may have changed.
2. `make check` — this runs **lint** (Ruff), **typecheck** (ty), and **tests** (pytest). All must pass.

If any step fails, fix issues and re-run until all checks pass. Do not leave with failing lint or typecheck.
