"""FastAPI app: list Orthanc experiences (studies) with associated reports from DB."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

load_dotenv()

from unboxed_ai.lib import OrthancClient  # noqa: E402
from unboxed_ai.lib.Constants import Constants  # noqa: E402
from unboxed_ai.lib.vector_store import create_engine_from_constants  # noqa: E402
from unboxed_ai.report_generation import (  # noqa: E402
    excel_to_df,
    fetch_studies_from_orthanc,
    generate_final_report,
    merge_on_accession,
)

app = FastAPI(title="Unboxed AI Experiences", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_orthanc_client() -> OrthancClient:
    return OrthancClient(
        base_url=Constants.ORTHANC_URL,
        username=Constants.ORTHANC_USERNAME,
        password=Constants.ORTHANC_PASSWORD,
        timeout=Constants.DEFAULT_TIMEOUT_SECONDS,
    )


def _fetch_reports() -> list[dict[str, Any]]:
    """Load all reports from the reports table. experience_id = DICOM AccessionNumber."""
    engine = create_engine_from_constants()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT id, type, patient_id, experience_id, report_description, is_validated
                FROM reports
                ORDER BY patient_id, experience_id
                """
            )
        )
        rows = result.fetchall()
    return [
        {
            "id": r[0],
            "type": str(r[1]) if r[1] is not None else "",
            "patient_id": str(r[2]) if r[2] is not None else "",
            "experience_id": str(r[3]) if r[3] is not None else "",
            "report_description": r[4],
            "is_validated": r[5],
        }
        for r in rows
    ]


def _compute_segmentation_export(accession: str | None = None) -> Path:
    """Generate segmentation export JSON via export_nodules_json.py.

    If accession is set, only that accession is processed and merged into the
    existing output file (avoids loading everything).
    """
    root = Path(__file__).resolve().parent.parent.parent
    script_path = root / "export_nodules_json.py"
    output_path = Path(Constants.SEGMENTATION_PATH)
    if not output_path.is_absolute():
        output_path = root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        "--output",
        str(output_path),
        "--thoracic-only",
    ]
    if accession and str(accession).strip():
        cmd.extend(["--accession", str(accession).strip()])
    subprocess.run(cmd, check=True, cwd=str(root))
    return output_path


def _get_nodule_images_dir() -> Path:
    """Return the project-root path for nodule export images."""
    root = Path(__file__).resolve().parent.parent.parent
    out = Path(Constants.NODULE_IMAGES_DIR)
    return root / out if not out.is_absolute() else out


def _run_nodule_images_export(accession: str | None = None) -> None:
    """Run generate_nodule_images.py to (re)generate finding images.

    If accession is set, only that accession is processed. On failure, log and continue.
    """
    root = Path(__file__).resolve().parent.parent.parent
    script_path = root / "generate_nodule_images.py"
    output_dir = _get_nodule_images_dir()
    if not script_path.exists():
        logging.warning(
            "generate_nodule_images.py not found, skipping nodule images export"
        )
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script_path), "--output", str(output_dir)]
        if accession and str(accession).strip():
            cmd.extend(["--accession", str(accession).strip()])
        subprocess.run(cmd, check=True, cwd=str(root))
    except subprocess.CalledProcessError as e:
        logging.warning(
            "Nodule images export failed: %s; report will still be generated", e
        )
    except Exception as e:
        logging.warning(
            "Nodule images export error: %s; report will still be generated", e
        )


def _list_nodule_images(patient_id: str, accession_id: str) -> list[dict[str, str]]:
    """List nodule image entries for a patient and accession. Returns [{url, filename}, ...]."""
    pid = (patient_id or "").strip()
    if ".." in pid or "/" in pid or "\\" in pid:
        return []
    base = _get_nodule_images_dir()
    patient_dir = base / pid
    if not patient_dir.is_dir():
        return []
    accession_prefix = (accession_id or "").strip()
    images: list[dict[str, str]] = []
    for path in sorted(patient_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        name = path.name
        if accession_prefix and not name.startswith(accession_prefix + "_"):
            continue
        url = f"/api/nodule-images/static/{pid}/{name}"
        images.append({"url": url, "filename": name})
    return images


@app.get("/api/nodule-images")
def get_nodule_images(
    patient_id: str = "",
    accession_id: str = "",
) -> dict[str, Any]:
    """List nodule finding image URLs for a patient and accession."""
    return {"images": _list_nodule_images(patient_id, accession_id)}


@app.get("/api/nodule-images/static/{patient_id}/{filename}")
def serve_nodule_image(patient_id: str, filename: str) -> FileResponse:
    """Serve a single nodule image file (path-safe, no traversal)."""
    pid = (patient_id or "").strip()
    fname = (filename or "").strip()
    if not pid or not fname:
        raise HTTPException(status_code=400, detail="patient_id and filename required")
    if ".." in pid or "/" in pid or "\\" in pid:
        raise HTTPException(status_code=400, detail="invalid patient_id")
    if ".." in fname or "/" in fname or "\\" in fname:
        raise HTTPException(status_code=400, detail="invalid filename")
    base = _get_nodule_images_dir()
    path = base / pid / fname
    if not path.is_file() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/experiences")
def get_experiences() -> list[dict[str, Any]]:
    """List all Orthanc studies (experiences) with associated reports from the database."""
    try:
        client = _get_orthanc_client()
        studies = client.list_studies(limit=None)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Orthanc error: {e!s}") from e

    try:
        reports = _fetch_reports()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database error: {e!s}") from e

    # experience_id in DB = DICOM AccessionNumber from Orthanc; match on (patient_id, experience_id) <-> (patient, accession)
    def _norm(s: Any) -> str:
        return str(s).strip() if s is not None else ""

    reports_by_experience: dict[tuple[str, str], list[dict[str, Any]]] = {}
    reports_by_accession: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        pid, eid = _norm(r["patient_id"]), _norm(r["experience_id"])
        key = (pid, eid)
        reports_by_experience.setdefault(key, []).append(r)
        reports_by_accession.setdefault(eid, []).append(r)

    result: list[dict[str, Any]] = []
    for s in studies:
        accession = _norm(s.get("accession"))
        patient = _norm(s.get("patient"))
        key = (patient, accession)
        linked_reports = reports_by_experience.get(key, [])
        if not linked_reports and accession:
            linked_reports = reports_by_accession.get(accession, [])
        result.append(
            {
                "id": s["id"],
                "patient": patient,
                "patient_id": patient,
                "patient_name": _norm(s.get("patient_name", "")),
                "accession": accession,
                "accession_id": accession,  # alias for frontend compatibility
                "description": s.get("description", "-"),
                "date": s.get("date", "-"),
                "modality": s.get("modality", "-"),
                "reports": linked_reports,
            }
        )
    return result


class GenerateReportRequest(BaseModel):
    patient_id: str
    experience_id: str | None = (
        None  # accession number for which to generate the report
    )


@app.post("/api/reports/generate")
def generate_report(req: GenerateReportRequest) -> dict[str, Any]:
    patient_id = str(req.patient_id or "").strip()
    experience_id = (
        str(req.experience_id or "").strip() if req.experience_id is not None else ""
    )
    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not experience_id:
        raise HTTPException(
            status_code=400,
            detail="experience_id (accession number) is required. Pass the accession of the experience to generate a report for.",
        )

    try:
        reports = _fetch_reports()
        validated_accession_numbers = {
            str(r["experience_id"])
            for r in reports
            if str(r.get("patient_id", "")).strip() == patient_id
            and r.get("is_validated") is True
        }
        reports_df = excel_to_df(Constants.REPORTS_PATH)
        orthanc_df = fetch_studies_from_orthanc()
        merged_df = merge_on_accession(reports_df, orthanc_df)
        segmentation_path = _compute_segmentation_export(accession=experience_id)
        _run_nodule_images_export(accession=experience_id)
        report_text = generate_final_report(
            merged_df,
            segmentation_algo_res_path=str(segmentation_path),
            patient_id=patient_id,
            report_accession_number=experience_id,
            output_file="",
            use_judge=True,
            validated_accession_numbers=validated_accession_numbers,
        )
        nodule_images = _list_nodule_images(patient_id, experience_id)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Report generation error: {e!s}"
        ) from e

    return {
        "patient_id": patient_id,
        "experience_id": experience_id,
        "report": report_text,
        "nodule_images": nodule_images,
    }


class SaveReportRequest(BaseModel):
    patient_id: str
    experience_id: str
    report_description: str
    report_type: str = "Generated"
    is_validated: bool = True


@app.post("/api/reports/save")
def save_report(req: SaveReportRequest) -> dict[str, Any]:
    patient_id = str(req.patient_id).strip()
    experience_id = str(req.experience_id).strip()
    report_description = str(req.report_description or "").strip()
    report_type = str(req.report_type or "Generated").strip() or "Generated"

    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")
    if not experience_id:
        raise HTTPException(status_code=400, detail="experience_id is required")
    if not report_description:
        raise HTTPException(status_code=400, detail="report_description is required")

    engine = create_engine_from_constants()
    try:
        with engine.connect() as conn:
            existing = conn.execute(
                text(
                    """
                    SELECT id
                    FROM reports
                    WHERE patient_id = :patient_id
                      AND experience_id = :experience_id
                      AND type = :report_type
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ),
                {
                    "patient_id": patient_id,
                    "experience_id": experience_id,
                    "report_type": report_type,
                },
            ).fetchone()

            if existing:
                report_id = int(existing[0])
                conn.execute(
                    text(
                        """
                        UPDATE reports
                        SET report_description = :report_description,
                            is_validated = :is_validated
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": report_id,
                        "report_description": report_description,
                        "is_validated": req.is_validated,
                    },
                )
            else:
                inserted = conn.execute(
                    text(
                        """
                        INSERT INTO reports (
                            type, patient_id, experience_id, report_description, is_validated
                        )
                        VALUES (
                            :report_type, :patient_id, :experience_id, :report_description, :is_validated
                        )
                        RETURNING id
                        """
                    ),
                    {
                        "report_type": report_type,
                        "patient_id": patient_id,
                        "experience_id": experience_id,
                        "report_description": report_description,
                        "is_validated": req.is_validated,
                    },
                ).fetchone()
                if not inserted:
                    raise HTTPException(status_code=500, detail="Failed to save report")
                report_id = int(inserted[0])

            conn.commit()
            row = conn.execute(
                text(
                    """
                    SELECT id, type, patient_id, experience_id, report_description, is_validated
                    FROM reports
                    WHERE id = :id
                    """
                ),
                {"id": report_id},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="Saved report not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database error: {e!s}") from e

    return {
        "report": {
            "id": row[0],
            "type": str(row[1]) if row[1] is not None else "",
            "patient_id": str(row[2]) if row[2] is not None else "",
            "experience_id": str(row[3]) if row[3] is not None else "",
            "report_description": row[4],
            "is_validated": row[5],
        }
    }


def _setup_frontend_routes() -> None:
    """Serve React build from frontend/dist, or fallback to vanilla frontend."""
    root = Path(__file__).resolve().parent.parent.parent
    nodule_dir = _get_nodule_images_dir()
    nodule_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/api/nodule-images/static",
        StaticFiles(directory=str(nodule_dir)),
        name="nodule_images_static",
    )
    dist_path = root / "frontend" / "dist"
    vanilla_path = root / "frontend.vanilla"

    if dist_path.exists():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True))
    elif (vanilla_path / "index.html").exists():

        @app.get("/")
        def _index() -> FileResponse:
            return FileResponse(vanilla_path / "index.html")
    else:

        @app.get("/")
        def _index() -> FileResponse:
            return FileResponse(root / "frontend" / "index.html")


_setup_frontend_routes()
