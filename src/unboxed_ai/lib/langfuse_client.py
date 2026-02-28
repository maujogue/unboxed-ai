import os

from langfuse import get_client
from langfuse.langchain import CallbackHandler


def is_langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_langchain_callbacks() -> list[CallbackHandler]:
    if not is_langfuse_enabled():
        return []

    # Lazily initialize the shared client only when tracing is configured.
    get_client()
    return [CallbackHandler()]


def flush_langfuse() -> None:
    if not is_langfuse_enabled():
        return

    get_client().flush()
