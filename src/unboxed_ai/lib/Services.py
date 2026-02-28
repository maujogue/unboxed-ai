from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .Constants import Constants
from .langfuse_client import get_langfuse_langchain_callbacks

if TYPE_CHECKING:
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    from langfuse.langchain import CallbackHandler


class Services:
    """Central registry for app-level clients and integrations."""

    @cached_property
    def orthanc(self):
        from .OrthancClient import OrthancClient

        return OrthancClient(
            base_url=Constants.ORTHANC_URL,
            username=Constants.ORTHANC_USERNAME,
            password=Constants.ORTHANC_PASSWORD,
            timeout=Constants.DEFAULT_TIMEOUT_SECONDS,
        )

    @cached_property
    def mistral_llm(self) -> "ChatMistralAI":
        from langchain_mistralai import ChatMistralAI

        api_key = Constants.MISTRAL_API_KEY
        if not api_key:
            raise RuntimeError(
                f"{Constants.MISTRAL_API_KEY_ENV} is required to create mistral_llm."
            )
        return ChatMistralAI(
            model=Constants.DEFAULT_MISTRAL_MODEL,
            api_key=api_key,
            temperature=Constants.DEFAULT_MISTRAL_TEMPERATURE,
        )

    @cached_property
    def mistral_embeddings(self) -> "MistralAIEmbeddings":
        from langchain_mistralai import MistralAIEmbeddings

        api_key = Constants.MISTRAL_API_KEY
        if not api_key:
            raise RuntimeError(
                f"{Constants.MISTRAL_API_KEY_ENV} is required to create mistral_embeddings."
            )
        return MistralAIEmbeddings(api_key=api_key)

    @cached_property
    def pgvector(self):
        try:
            from langchain_postgres import PGVector  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "PGVector service requires `langchain-postgres` to be installed."
            ) from exc
        return PGVector(
            embeddings=self.mistral_embeddings,
            connection=Constants.PGVECTOR_CONNECTION,
            collection_name=Constants.DEFAULT_PGVECTOR_COLLECTION_NAME,
        )

    @cached_property
    def langfuse_callbacks(self) -> list["CallbackHandler"]:
        return get_langfuse_langchain_callbacks()
