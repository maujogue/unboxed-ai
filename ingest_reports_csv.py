#!/usr/bin/env python3
"""Ingest assets/processed_reports.csv into a Postgres 'reports' table."""

from __future__ import annotations

import csv
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from unboxed_ai.lib.vector_store import create_engine_from_constants  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "assets" / "processed_reports.csv"


def main() -> None:
    engine = create_engine_from_constants()

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS reports"))
        conn.execute(
            text("""
            CREATE TABLE reports (
                id SERIAL PRIMARY KEY,
                type TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                experience_id TEXT NOT NULL,
                report_description TEXT,
                is_validated BOOLEAN NOT NULL
            )
            """)
        )
        conn.commit()

        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            is_validated = row["is_validated"].strip().lower() == "true"
            conn.execute(
                text("""
                INSERT INTO reports (type, patient_id, experience_id, report_description, is_validated)
                VALUES (:type, :patient_id, :experience_id, :report_description, :is_validated)
                """),
                {
                    "type": row["type"].strip(),
                    "patient_id": row["patient_id"].strip(),
                    "experience_id": row["experience_id"].strip(),
                    "report_description": (row["report_description"] or "").strip(),
                    "is_validated": is_validated,
                },
            )
        conn.commit()

    print(f"Ingested {len(rows)} rows into reports table.")


if __name__ == "__main__":
    main()
