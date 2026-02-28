from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

DEFAULT_ORTHANC_URL = "https://orthanc.unboxed-2026.ovh"
DEFAULT_ORTHANC_AUTH = ("unboxed", "unboxed2026")
DEFAULT_WORK_PATH = "/home/jovyan/work"


class OrthancClient:
    """Convenience wrapper around the Orthanc HTTP API used in the notebook."""

    def __init__(
        self,
        base_url: str = DEFAULT_ORTHANC_URL,
        username: str = DEFAULT_ORTHANC_AUTH[0],
        password: str = DEFAULT_ORTHANC_AUTH[1],
        timeout: int = 5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.timeout = timeout

    def upload_dicom(self, path: str) -> str | None:
        """Upload one DICOM file and return Orthanc instance ID."""
        with open(path, "rb") as stream:
            response = requests.post(
                f"{self.base_url}/instances",
                auth=self.auth,
                data=stream.read(),
                headers={"Content-Type": "application/dicom"},
                timeout=self.timeout,
            )
        if response.status_code == 200:
            return response.json().get("ID")
        return None

    def list_studies(self, limit: int | None = 10) -> list[dict[str, Any]]:
        """List studies and extract key tags."""
        studies_response = requests.get(
            f"{self.base_url}/studies",
            auth=self.auth,
            timeout=self.timeout,
        )
        studies_response.raise_for_status()
        study_ids: list[str] = studies_response.json()

        subset = study_ids if limit is None else study_ids[:limit]
        rows: list[dict[str, Any]] = []
        for study_id in subset:
            detail_response = requests.get(
                f"{self.base_url}/studies/{study_id}",
                auth=self.auth,
                timeout=self.timeout,
            )
            detail_response.raise_for_status()
            info = detail_response.json()
            tags = info.get("MainDicomTags", {})
            rows.append(
                {
                    "id": study_id,
                    "patient": tags.get("PatientID", "-"),
                    "description": tags.get("StudyDescription", "-"),
                    "date": tags.get("StudyDate", "-"),
                    "modality": tags.get("ModalitiesInStudy", "-"),
                    "raw": info,
                }
            )
        return rows

    def upload_dicom_folder(self, folder: str) -> dict[str, Any]:
        """Upload all .dcm files from a folder recursively."""
        files = list(Path(folder).rglob("*.dcm"))
        uploaded: list[str] = []
        failed: list[str] = []
        for file_path in files:
            instance_id = self.upload_dicom(str(file_path))
            if instance_id:
                uploaded.append(str(file_path))
            else:
                failed.append(str(file_path))
        return {
            "folder": folder,
            "total": len(files),
            "uploaded": len(uploaded),
            "failed": len(failed),
            "uploaded_files": uploaded,
            "failed_files": failed,
        }

    def download_study(self, study_id: str, out_dir: str = DEFAULT_WORK_PATH) -> str:
        """Download one Orthanc study archive to a zip file."""
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        target_file = destination / f"study_{study_id[:8]}.zip"
        with requests.get(
            f"{self.base_url}/studies/{study_id}/archive",
            auth=self.auth,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            with open(target_file, "wb") as stream:
                for chunk in response.iter_content(chunk_size=8192):
                    stream.write(chunk)
        return str(target_file)
