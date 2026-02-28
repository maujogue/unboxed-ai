"""
Pipeline: Orthanc CT series → dcm-seg-nodules → Orthanc SEG upload.

Usage:
    # Process all series in Orthanc
    uv run python -m unboxed_ai.orthanc_pipeline

    # Process a specific series
    uv run python -m unboxed_ai.orthanc_pipeline --series <orthanc-series-id>
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

from dcm_seg_nodules import extract_seg

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_env() -> dict[str, str]:
    """Load config from environment (or .env file if python-dotenv is available)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return {
        "url": os.environ.get("ORTHANC_URL", "http://localhost:8042").rstrip("/"),
        "user": os.environ.get("ORTHANC_USER", ""),
        "password": os.environ.get("ORTHANC_PASSWORD", ""),
    }


class OrthancClient:
    def __init__(self, url: str, user: str = "", password: str = "") -> None:
        self.url = url
        self.auth = HTTPBasicAuth(user, password) if user else None

    def _get(self, path: str, **kwargs) -> requests.Response:
        r = requests.get(f"{self.url}{path}", auth=self.auth, **kwargs)
        r.raise_for_status()
        return r

    def _post(self, path: str, **kwargs) -> requests.Response:
        r = requests.post(f"{self.url}{path}", auth=self.auth, **kwargs)
        r.raise_for_status()
        return r

    def list_series(self) -> list[str]:
        """Return all series IDs in Orthanc."""
        return self._get("/series").json()

    def get_series_info(self, series_id: str) -> dict:
        return self._get(f"/series/{series_id}").json()

    def download_instance(self, instance_id: str, dest: Path) -> Path:
        """Download a single DICOM instance file."""
        response = self._get(f"/instances/{instance_id}/file", stream=True)
        filepath = dest / f"{instance_id}.dcm"
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    def upload_file(self, filepath: Path) -> dict:
        """Upload a DICOM file to Orthanc. Returns the Orthanc response."""
        with open(filepath, "rb") as f:
            response = self._post("/instances", data=f.read(),
                                  headers={"Content-Type": "application/dicom"})
        return response.json()


def process_series(client: OrthancClient, series_id: str, output_dir: Path) -> bool:
    """
    Download a CT series from Orthanc, extract the matching SEG, and re-upload it.
    Returns True on success, False if no matching SEG was found.
    """
    info = client.get_series_info(series_id)
    modality = info.get("MainDicomTags", {}).get("Modality", "?")
    description = info.get("MainDicomTags", {}).get("SeriesDescription", "")
    patient_id = info.get("ParentPatient", series_id)[:8]

    logger.info("Series %s | Modality=%s | %s", series_id[:8], modality, description)

    # Only process CT series
    if modality != "CT":
        logger.info("  → Skipping (not CT)")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        series_dir = Path(tmpdir) / patient_id / "original"
        series_dir.mkdir(parents=True)

        # Download all instances
        instance_ids = info.get("Instances", [])
        logger.info("  Downloading %d instances...", len(instance_ids))
        for instance_id in instance_ids:
            client.download_instance(instance_id, series_dir)

        # Extract SEG from bundled registry
        try:
            seg_path, exam_info = extract_seg(
                Path(tmpdir) / patient_id,
                output_dir=output_dir,
            )
        except FileNotFoundError as e:
            logger.warning("  → No matching SEG: %s", e)
            return False

        logger.info("  SEG extracted to %s", seg_path)
        if exam_info:
            logger.info("  Exam info:\n%s", exam_info)

        # Upload SEG back to Orthanc
        result = client.upload_file(seg_path)
        status = result.get("Status", "?")
        instance_id_uploaded = result.get("ID", "?")
        logger.info("  Uploaded to Orthanc: %s (status: %s)", instance_id_uploaded, status)

    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download CT series from Orthanc, extract SEG, re-upload."
    )
    parser.add_argument(
        "--series",
        help="Orthanc series ID to process (default: all CT series).",
    )
    parser.add_argument(
        "--output", default="results",
        help="Local directory for intermediate SEG files (default: results/).",
    )
    args = parser.parse_args(argv)

    cfg = _load_env()
    logger.info("Connecting to Orthanc at %s", cfg["url"])

    client = OrthancClient(cfg["url"], cfg["user"], cfg["password"])
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Check connectivity
    try:
        client._get("/system")
    except Exception as e:
        logger.error("Cannot reach Orthanc: %s", e)
        return

    if args.series:
        series_ids = [args.series]
    else:
        series_ids = client.list_series()
        logger.info("Found %d series in Orthanc", len(series_ids))

    success, skipped, failed = 0, 0, 0
    for series_id in series_ids:
        try:
            ok = process_series(client, series_id, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error("Error processing series %s: %s", series_id, e)
            failed += 1

    logger.info(
        "\nDone. %d SEG(s) uploaded | %d skipped (no match) | %d error(s)",
        success, skipped, failed,
    )


if __name__ == "__main__":
    main()
