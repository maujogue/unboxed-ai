from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Constants:
    """Central place for library-wide constants."""

    DEFAULT_WORK_PATH = "downloads/orthanc_studies"
    DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
    DEFAULT_MISTRAL_TEMPERATURE = 0.0
    DEFAULT_PGVECTOR_COLLECTION_NAME = "unboxed_ai"

    DEFAULT_TIMEOUT_SECONDS = 5
    DOWNLOAD_TIMEOUT_SECONDS = 120

    ORTHANC_URL = os.getenv("ORTHANC_URL", "https://orthanc.unboxed-2026.ovh")
    ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "unboxed")
    ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "unboxed2026")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

    REPORTS_PATH = os.getenv("REPORTS_PATH", "assets/new-reports.ods")
    SEGMENTATION_PATH = os.getenv("SEGMENTATION_PATH", "nodules_export.json")
    # PostgreSQL / pgvector (use PGVECTOR_CONNECTION to override built URL)
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    _pgvector_connection = os.getenv("PGVECTOR_CONNECTION")
    PGVECTOR_CONNECTION = (
        _pgvector_connection
        if _pgvector_connection
        else f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    PGVECTOR_COLLECTION_NAME = os.getenv(
        "PGVECTOR_COLLECTION_NAME", DEFAULT_PGVECTOR_COLLECTION_NAME
    )

    ORTHANC_INSTANCES_ENDPOINT = f"{ORTHANC_URL.rstrip('/')}/instances"
    ORTHANC_STUDIES_ENDPOINT = f"{ORTHANC_URL.rstrip('/')}/studies"
    ORTHANC_STUDY_DETAIL_ENDPOINT = f"{ORTHANC_URL.rstrip('/')}/studies/{{study_id}}"
    ORTHANC_STUDY_ARCHIVE_ENDPOINT = (
        f"{ORTHANC_URL.rstrip('/')}/studies/{{study_id}}/archive"
    )


constants = Constants()
