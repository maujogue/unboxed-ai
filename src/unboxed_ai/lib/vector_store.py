from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .Constants import Constants

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_postgres import PGVector


def create_engine_from_constants() -> Engine:
    """Build a sync SQLAlchemy engine from Constants (PGVECTOR_CONNECTION or POSTGRES_*)."""
    from sqlalchemy import create_engine

    return create_engine(
        Constants.PGVECTOR_CONNECTION,
        pool_pre_ping=True,
    )


class VectorStoreService:
    """Orchestrates pgvector extension setup, embeddings ingestion, and document retrieval."""

    def __init__(self, store: PGVector, engine: Engine) -> None:
        self._store = store
        self._engine = engine
        self._session_factory = sessionmaker(
            bind=engine, autocommit=False, autoflush=False
        )

    def ensure_pgvector_ready(self) -> None:
        """Ensure the pgvector extension exists in the database."""
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    def ingest_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Ingest LangChain documents into the vector store. Returns document ids."""
        return self._store.add_documents(documents, ids=ids)

    def ingest_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Ingest raw texts (and optional metadata) into the vector store. Returns document ids."""
        return self._store.add_texts(texts, metadatas=metadatas, ids=ids)

    def get_retriever(
        self,
        k: int = 4,
        search_type: str = "similarity",
    ) -> BaseRetriever:
        """Return a retriever over the vector store with the given k and search_type."""
        return self._store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Run a similarity search and return the top k documents."""
        return self._store.similarity_search(query, k=k)
