from __future__ import annotations

import os


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
    PGVECTOR_CONNECTION = os.getenv(
        "PGVECTOR_CONNECTION", "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )

    ORTHANC_INSTANCES_ENDPOINT = f"{ORTHANC_URL.rstrip('/')}/instances"
    ORTHANC_STUDIES_ENDPOINT = f"{ORTHANC_URL.rstrip('/')}/studies"
    ORTHANC_STUDY_DETAIL_ENDPOINT = f"{ORTHANC_URL.rstrip('/')}/studies/{{study_id}}"
    ORTHANC_STUDY_ARCHIVE_ENDPOINT = (
        f"{ORTHANC_URL.rstrip('/')}/studies/{{study_id}}/archive"
    )
