"""
Pipeline: Orthanc CT series → dcm-seg-nodules → Orthanc SEG upload + rapport HTML.

Optimisation : au lieu de parcourir toutes les séries Orthanc, on interroge
Orthanc uniquement pour les SeriesInstanceUIDs présents dans le registre bundlé
(au plus 17 requêtes légères au lieu de télécharger 100+ séries).

Usage:
    uv run orthanc-pipeline
    uv run orthanc-pipeline --output results/ --report rapport.html
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import requests

from dcm_seg_nodules import registry as seg_registry

from .lib.Constants import Constants

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orthanc client (minimal, focused on what the pipeline needs)
# ---------------------------------------------------------------------------


class OrthancClient:
    def __init__(self) -> None:
        self.url = Constants.ORTHANC_URL.rstrip("/")
        self.auth = (Constants.ORTHANC_USERNAME, Constants.ORTHANC_PASSWORD)

    def get(self, path: str, **kwargs) -> requests.Response:
        r = requests.get(f"{self.url}{path}", auth=self.auth, **kwargs)
        r.raise_for_status()
        return r

    def post(self, path: str, **kwargs) -> requests.Response:
        r = requests.post(f"{self.url}{path}", auth=self.auth, **kwargs)
        r.raise_for_status()
        return r

    def find_series_by_uid(self, series_instance_uid: str) -> list[str]:
        """Return Orthanc internal series IDs matching a DICOM SeriesInstanceUID."""
        result = self.post(
            "/tools/find",
            json={
                "Level": "Series",
                "Query": {"SeriesInstanceUID": series_instance_uid},
            },
        )
        return result.json()

    def get_series_info(self, series_id: str) -> dict:
        return self.get(f"/series/{series_id}").json()

    def get_patient_info(self, patient_id: str) -> dict:
        return self.get(f"/patients/{patient_id}").json()

    def get_study_info(self, study_id: str) -> dict:
        return self.get(f"/studies/{study_id}").json()

    def download_instance(self, instance_id: str, dest: Path) -> Path:
        r = self.get(f"/instances/{instance_id}/file", stream=True)
        filepath = dest / f"{instance_id}.dcm"
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    def upload_file(self, filepath: Path) -> dict:
        with open(filepath, "rb") as f:
            r = self.post(
                "/instances",
                data=f.read(),
                headers={"Content-Type": "application/dicom"},
            )
        return r.json()


# ---------------------------------------------------------------------------
# Nodule extraction from SEG pixel data
# ---------------------------------------------------------------------------


@dataclass
class Nodule:
    number: int
    diameter: str
    x: float
    y: float
    z: float
    z_min: float
    z_max: float


def extract_nodules(seg_path: Path, info_text: str | None) -> list[Nodule]:
    """Parse a DICOM SEG file and return centroid + diameter for each nodule."""
    ds = pydicom.dcmread(str(seg_path))

    # Parse diameters from info text
    diameters: dict[int, str] = {}
    if info_text:
        for line in info_text.splitlines():
            line = line.strip()
            if line.startswith("- Finding"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    num = int(parts[0].replace("- Finding", "").strip())
                    diameters[num] = parts[1].strip()

    if not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        return []

    frames = ds.PerFrameFunctionalGroupsSequence
    pixel_array = ds.pixel_array

    seg_data: dict[int, list[np.ndarray]] = {}
    seg_z: dict[int, list[float]] = {}

    for i, frame in enumerate(frames):
        seg_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        pos = [float(v) for v in frame.PlanePositionSequence[0].ImagePositionPatient]
        mask = pixel_array[i]

        if not mask.any():
            continue

        if hasattr(frame, "PlaneOrientationSequence"):
            orientation = [
                float(v)
                for v in frame.PlaneOrientationSequence[0].ImageOrientationPatient
            ]
            spacing = [float(v) for v in frame.PixelMeasuresSequence[0].PixelSpacing]
            row_dir = np.array(orientation[:3])
            col_dir = np.array(orientation[3:])
            origin = np.array(pos)
            ys, xs = np.where(mask)
            coords = [
                origin + x * spacing[1] * row_dir + y * spacing[0] * col_dir
                for x, y in zip(xs, ys)
            ]
            seg_data.setdefault(seg_num, []).extend(coords)

        seg_z.setdefault(seg_num, []).append(pos[2])

    nodules = []
    for seg_num in sorted(seg_data.keys()):
        arr = np.array(seg_data[seg_num])
        centroid = arr.mean(axis=0)
        z_vals = seg_z[seg_num]
        nodules.append(
            Nodule(
                number=seg_num,
                diameter=diameters.get(seg_num, "N/A"),
                x=round(float(centroid[0]), 1),
                y=round(float(centroid[1]), 1),
                z=round(float(centroid[2]), 1),
                z_min=round(min(z_vals), 1),
                z_max=round(max(z_vals), 1),
            )
        )
    return nodules


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@dataclass
class SeriesResult:
    patient_id: str
    study_description: str
    accession_number: str
    series_description: str
    orthanc_series_id: str
    uploaded_instance_id: str
    nodules: list[Nodule] = field(default_factory=list)
    info_text: str = ""


def generate_markdown(results: list[SeriesResult], date: str) -> str:
    lines = [
        "# Rapport de segmentation des nodules pulmonaires",
        f"**Date :** {date}  ",
        f"**Séries traitées :** {len(results)}",
        "",
    ]

    for r in results:
        lines += [
            f"---",
            f"## Patient `{r.patient_id}` — {r.study_description}",
            f"**Accession :** {r.accession_number}  ",
            f"**Série :** {r.series_description}  ",
            f"**Orthanc series ID :** `{r.orthanc_series_id}`  ",
            f"**SEG uploadé :** `{r.uploaded_instance_id}`  ",
            "",
        ]

        if r.nodules:
            lines += [
                f"### {len(r.nodules)} nodule(s) détecté(s)",
                "",
                "| # | Diamètre | Centre X (mm) | Centre Y (mm) | Centre Z (mm) | Étendue Z (mm) |",
                "|---|----------|--------------|--------------|--------------|----------------|",
            ]
            for n in r.nodules:
                etendue = round(n.z_max - n.z_min, 1)
                lines.append(
                    f"| {n.number} | {n.diameter} | {n.x} | {n.y} | {n.z} | {etendue} |"
                )
            lines.append("")
        else:
            lines += ["*Aucune donnée de nodule extraite.*", ""]

    return "\n".join(lines)


def markdown_to_html(md: str, title: str = "Rapport nodules") -> str:
    # Conversion Markdown → HTML sans dépendance externe
    import re

    html_lines = []
    in_table = False
    in_list = False

    for line in md.splitlines():
        # Headings
        if line.startswith("### "):
            line = f"<h3>{line[4:]}</h3>"
        elif line.startswith("## "):
            line = f"<h2>{line[3:]}</h2>"
        elif line.startswith("# "):
            line = f"<h1>{line[2:]}</h1>"
        # HR
        elif line.strip() == "---":
            line = "<hr>"
        # Table
        elif line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                # separator row
                if not in_table:
                    in_table = True
                continue
            if not in_table:
                in_table = True
                tag = "th"
                html_lines.append("<table>")
            else:
                tag = "td"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            line = f"<tr>{row}</tr>"
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            # Bold
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            # Inline code
            line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
            # Italic
            line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
            # Empty line → paragraph break
            if line.strip() == "":
                line = "<br>"

        html_lines.append(line)

    if in_table:
        html_lines.append("</table>")

    body = "\n".join(html_lines)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ color: #1a5276; margin-top: 32px; }}
    h3 {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
    th {{ background: #f0f4f8; font-weight: bold; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(output_dir: Path, report_path: Path) -> None:
    client = OrthancClient()

    try:
        client.get("/system")
        logger.info("Connecté à Orthanc : %s", Constants.ORTHANC_URL)
    except Exception as e:
        logger.error("Impossible de joindre Orthanc : %s", e)
        return

    # --- Étape 1 : récupérer les UIDs du registre bundlé -------------------
    registry_entries = seg_registry.list_entries()
    logger.info("%d séries dans le registre bundlé", len(registry_entries))

    # --- Étape 2 : chercher uniquement ces UIDs dans Orthanc ---------------
    # (évite de télécharger toutes les séries)
    matched: list[tuple[str, str]] = []  # (series_instance_uid, orthanc_series_id)
    for series_uid in registry_entries:
        orthanc_ids = client.find_series_by_uid(series_uid)
        for oid in orthanc_ids:
            matched.append((series_uid, oid))
            logger.info("Match : UID ...%s → Orthanc %s", series_uid[-16:], oid[:8])

    logger.info(
        "%d série(s) à traiter (présentes dans Orthanc ET dans le registre)",
        len(matched),
    )

    if not matched:
        logger.warning("Aucune série commune entre Orthanc et le registre.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SeriesResult] = []

    # --- Étape 3 : pour chaque match, copier le SEG et uploader ------------
    for series_uid, orthanc_series_id in matched:
        series_info = client.get_series_info(orthanc_series_id)
        series_tags = series_info.get("MainDicomTags", {})
        series_desc = series_tags.get("SeriesDescription", "")

        study_info = client.get_study_info(series_info["ParentStudy"])
        study_tags = study_info.get("MainDicomTags", {})
        study_desc = study_tags.get("StudyDescription", "")
        accession = study_tags.get("AccessionNumber", "")

        patient_info = client.get_patient_info(study_info["ParentPatient"])
        patient_id = patient_info.get("MainDicomTags", {}).get(
            "PatientID", orthanc_series_id[:8]
        )

        logger.info("Traitement : patient=%s | %s", patient_id, series_desc)

        # Lookup direct dans le registre (pas de téléchargement nécessaire)
        seg_path = seg_registry.lookup(series_uid)
        info_text = seg_registry.lookup_info(series_uid)

        if seg_path is None:
            logger.error("SEG introuvable pour UID %s", series_uid)
            continue

        # Copier le SEG dans output_dir
        dest_dir = output_dir / patient_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"output-seg-{orthanc_series_id[:8]}.dcm"
        shutil.copy2(str(seg_path), str(dest_file))

        # Upload vers Orthanc
        upload_result = client.upload_file(dest_file)
        uploaded_id = upload_result.get("ID", "?")
        status = upload_result.get("Status", "?")
        logger.info("  SEG uploadé : %s (status: %s)", uploaded_id, status)

        # Extraire les nodules depuis le SEG
        nodules = extract_nodules(seg_path, info_text)

        results.append(
            SeriesResult(
                patient_id=patient_id,
                study_description=study_desc,
                accession_number=accession,
                series_description=series_desc,
                orthanc_series_id=orthanc_series_id,
                uploaded_instance_id=uploaded_id,
                nodules=nodules,
                info_text=info_text or "",
            )
        )

    # --- Étape 4 : générer le rapport --------------------------------------
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = generate_markdown(results, date_str)
    html = markdown_to_html(md)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    logger.info("\nRapport généré : %s", report_path)
    logger.info("Done. %d SEG(s) traité(s).", len(results))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extrait les SEG depuis le registre bundlé, les uploade dans Orthanc et génère un rapport HTML."
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Dossier de sortie pour les SEG (défaut: results/)",
    )
    parser.add_argument(
        "--report",
        default="results/rapport.html",
        help="Chemin du rapport HTML (défaut: results/rapport.html)",
    )
    args = parser.parse_args(argv)

    run_pipeline(Path(args.output), Path(args.report))


if __name__ == "__main__":
    main()
