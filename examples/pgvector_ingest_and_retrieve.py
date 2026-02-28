from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from unboxed_ai.lib import Services


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest sample texts into pgvector and run a similarity search."
    )
    parser.add_argument(
        "--query",
        default="imaging and radiology",
        help="Query string for similarity search (default: 'imaging and radiology').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to return (default: 4).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=5,
        help="Max seconds to wait for Postgres to be ready (default: 5).",
    )
    return parser.parse_args()


def wait_for_db(services: Services, timeout_seconds: int) -> bool:
    """Return True if the database is reachable and pgvector is ready."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            services.vector_store.ensure_pgvector_ready()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    load_dotenv()
    args = parse_args()

    services = Services()
    if not wait_for_db(services, args.wait_seconds):
        print(
            "pgvector_ingest_and_retrieve: Postgres/pgvector not ready in time.",
            file=sys.stderr,
        )
        return 1

    store = services.vector_store
    sample_texts = [
        "CT scans are used for detailed imaging of bones and soft tissues.",
        "MRI is preferred for brain and spinal cord imaging.",
        "Radiology reports document findings from medical imaging.",
        "DICOM is the standard format for storing medical images.",
    ]
    ids = store.ingest_texts(sample_texts)
    print(f"Ingested {len(ids)} document(s).")

    docs = store.similarity_search(args.query, k=args.k)
    print(f"Top {len(docs)} result(s) for query '{args.query}':")
    for i, doc in enumerate(docs, start=1):
        content = (
            doc.page_content[:77] + "..."
            if len(doc.page_content) > 80
            else doc.page_content
        )
        print(f"  {i}. {content}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
