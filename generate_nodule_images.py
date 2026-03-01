"""
Génère une image fusionnée CT + masque SEG pour la coupe la plus caractéristique
de chaque nodule validé dans findings_validation.json.

Source des DICOMs : fichiers locaux (Orthanc n'est pas requis).

Stratégie de sélection de la coupe :
  1. Si findings_validation.json contient image_number pour ce Finding
     → coupe CT avec InstanceNumber = image_number
  2. Sinon → frame SEG dont le masque a la plus grande surface
     (correspond à la coupe au diamètre maximal)

Usage:
    uv run python generate_nodule_images.py
    uv run python generate_nodule_images.py --output export_nodules/
    uv run python generate_nodule_images.py --accession 31981427
    uv run python generate_nodule_images.py --dataset /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from dcm_seg_nodules import registry as seg_registry

DATASET_DIR = Path("/home/corentin/Downloads/dataset_backup(1)/dataset")
FINDINGS_FILE = Path("findings_validation.json")

# ---------------------------------------------------------------------------
# Orthanc helpers (conservés pour référence — désactivés, Orthanc est down)
# ---------------------------------------------------------------------------

# import os
# import requests
# from io import BytesIO
# from dotenv import load_dotenv
# load_dotenv()
# ORTHANC_URL = os.getenv("ORTHANC_URL", "https://orthanc.unboxed-2026.ovh")
# ORTHANC_USER = os.getenv("ORTHANC_USERNAME", "unboxed")
# ORTHANC_PASS = os.getenv("ORTHANC_PASSWORD", "unboxed2026")
# AUTH = (ORTHANC_USER, ORTHANC_PASS) if ORTHANC_USER else None
#
# def _get(path: str, **kwargs) -> requests.Response:
#     r = requests.get(f"{ORTHANC_URL}{path}", auth=AUTH, **kwargs)
#     r.raise_for_status()
#     return r
#
# def _post(path: str, **kwargs) -> requests.Response:
#     r = requests.post(f"{ORTHANC_URL}{path}", auth=AUTH, **kwargs)
#     r.raise_for_status()
#     return r
#
# def find_series_by_uid(series_uid: str) -> list[str]:
#     return _post("/tools/find", json={"Level": "Series", "Query": {"SeriesInstanceUID": series_uid}}).json()
#
# def get_series_instances(series_id: str) -> list[dict]:
#     return _get(f"/series/{series_id}/instances").json()
#
# def get_instance_tags(instance_id: str) -> dict:
#     return _get(f"/instances/{instance_id}").json().get("MainDicomTags", {})
#
# def download_ct_slice(instance_id: str) -> pydicom.Dataset:
#     r = _get(f"/instances/{instance_id}/file")
#     return pydicom.dcmread(BytesIO(r.content))

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def build_accession_to_uid_map() -> dict[str, str]:
    """Build {accession_number: ct_series_uid} from registry info text."""
    result: dict[str, str] = {}
    for ct_uid, meta in seg_registry.list_entries().items():
        info = meta.get("info", "")
        for line in info.splitlines():
            line = line.strip()
            if line.startswith("Accession Number:"):
                acc = line.split(":", 1)[1].strip()
                if acc and acc != "0000":
                    result[acc] = ct_uid
                break
    return result


# ---------------------------------------------------------------------------
# Parsing helpers (conservé pour référence — remplacé par findings_validation.json)
# ---------------------------------------------------------------------------

# def parse_findings(info_text: str | None) -> dict[int, dict]:
#     """Parse all Finding lines from the registry info text.
#     Returns {seg_num: {"diameter": "17mm", "image_number": 39 or None}}.
#     """
#     result: dict[int, dict] = {}
#     if not info_text:
#         return result
#     for line in info_text.splitlines():
#         line = line.strip()
#         if not line.startswith("- Finding"):
#             continue
#         parts = line.split(":", 1)
#         if len(parts) < 2:
#             continue
#         try:
#             num = int(parts[0].replace("- Finding", "").strip())
#         except ValueError:
#             continue
#         detail = parts[1].strip()
#         img_match = re.search(r"\(Image\s+(\d+)\)", detail)
#         diam_match = re.search(r"([\d.]+\s*mm)", detail, re.IGNORECASE)
#         result[num] = {
#             "diameter": diam_match.group(1) if diam_match else "N/A",
#             "image_number": int(img_match.group(1)) if img_match else None,
#         }
#     return result


# ---------------------------------------------------------------------------
# CT index building — local filesystem (remplace build_ct_index Orthanc)
# ---------------------------------------------------------------------------

# Ancienne version Orthanc (conservée en commentaire) :
# def build_ct_index(ct_series_id: str) -> tuple[dict[str, str], dict[int, str]]:
#     uid_to_orthanc: dict[str, str] = {}
#     instnum_to_uid: dict[int, str] = {}
#     instances = get_series_instances(ct_series_id)
#     for inst in instances:
#         tags = get_instance_tags(inst["ID"])
#         sop_uid = tags.get("SOPInstanceUID", "")
#         if not sop_uid:
#             continue
#         uid_to_orthanc[sop_uid] = inst["ID"]
#         inst_num_str = tags.get("InstanceNumber", "")
#         if inst_num_str:
#             try:
#                 instnum_to_uid[int(inst_num_str)] = sop_uid
#             except ValueError:
#                 pass
#     return uid_to_orthanc, instnum_to_uid


def build_local_ct_index(
    ct_series_uid: str,
    patient_id: str,
    dataset_dir: Path,
) -> tuple[dict[str, Path], dict[int, str]]:
    """Index CT slices from the local dataset directory.

    Searches under the patient's folder for DCM files belonging to ct_series_uid.

    Returns:
        uid_to_path: SOPInstanceUID → filepath
        instnum_to_uid: InstanceNumber (int) → SOPInstanceUID
    """
    uid_to_path: dict[str, Path] = {}
    instnum_to_uid: dict[int, str] = {}

    patient_dirs = list(dataset_dir.glob(f"*{patient_id}*"))
    if not patient_dirs:
        print(f"  [WARN] No patient folder found for {patient_id}")
        return uid_to_path, instnum_to_uid

    for patient_dir in patient_dirs:
        for dcm_path in patient_dir.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            except Exception:
                continue
            if getattr(ds, "SeriesInstanceUID", None) != ct_series_uid:
                continue
            sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "")
            if not sop_uid:
                continue
            uid_to_path[sop_uid] = dcm_path
            inst_num_str = getattr(ds, "InstanceNumber", None)
            if inst_num_str is not None:
                try:
                    instnum_to_uid[int(inst_num_str)] = sop_uid
                except (ValueError, TypeError):
                    pass

    return uid_to_path, instnum_to_uid


# ---------------------------------------------------------------------------
# SEG index building
# ---------------------------------------------------------------------------


def build_seg_index(
    seg_dcm: pydicom.Dataset,
) -> tuple[
    dict[tuple[int, str], int],  # (finding_num, ref_sop_uid) → frame_idx
    dict[int, list[tuple[int, int]]],  # finding_num → [(frame_idx, pixel_count)]
    np.ndarray,  # pixel_array (n_frames, H, W)
]:
    """Index SEG frames by (finding_number, referenced_CT_uid) and by area.

    finding_number is derived from the SegmentLabel (e.g. 'Finding.3' → 3),
    falling back to the raw SegmentNumber if no label match is found.
    """
    pixel_array = seg_dcm.pixel_array
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    # Build seg_num → finding_number from DICOM SegmentSequence labels
    seg_num_to_finding: dict[int, int] = {}
    if hasattr(seg_dcm, "SegmentSequence"):
        for seg in seg_dcm.SegmentSequence:
            desc = str(getattr(seg, "SegmentDescription", "") or getattr(seg, "SegmentLabel", ""))
            match = re.search(r"Finding[.\s_]?(\d+)", desc, re.IGNORECASE)
            seg_num_to_finding[int(seg.SegmentNumber)] = (
                int(match.group(1)) if match else int(seg.SegmentNumber)
            )

    seg_frame_by_key: dict[tuple[int, str], int] = {}
    areas_by_seg: dict[int, list[tuple[int, int]]] = {}
    frames = seg_dcm.PerFrameFunctionalGroupsSequence

    for i, frame in enumerate(frames):
        seg_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        finding_num = seg_num_to_finding.get(seg_num, seg_num)
        area = int(pixel_array[i].sum())

        ref_uid: str | None = None
        if hasattr(frame, "DerivationImageSequence"):
            try:
                src = frame.DerivationImageSequence[0].SourceImageSequence[0]
                ref_uid = str(getattr(src, "ReferencedSOPInstanceUID", None))
            except (IndexError, AttributeError):
                pass

        if ref_uid:
            seg_frame_by_key[(finding_num, ref_uid)] = i
        areas_by_seg.setdefault(finding_num, []).append((i, area))

    return seg_frame_by_key, areas_by_seg, pixel_array


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def normalize_ct(ct_dcm: pydicom.Dataset, wl: int = -600, ww: int = 1500) -> np.ndarray:
    """Convert to Hounsfield Units then apply lung windowing + normalize to [0, 1]."""
    slope = float(getattr(ct_dcm, "RescaleSlope", 1))
    intercept = float(getattr(ct_dcm, "RescaleIntercept", 0))
    hu = ct_dcm.pixel_array.astype(np.float32) * slope + intercept
    low, high = wl - ww // 2, wl + ww // 2
    clipped = np.clip(hu, low, high)
    return (clipped - low) / (high - low)


def save_overlay(
    ct_norm: np.ndarray,
    seg_mask: np.ndarray | None,
    out_path: Path,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ct_norm, cmap="gray")
    if seg_mask is not None and seg_mask.any():
        red = np.zeros((*seg_mask.shape, 4), dtype=np.float32)
        red[seg_mask > 0] = [1.0, 0.0, 0.0, 0.5]
        ax.imshow(red)
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-accession processing (piloté par findings_validation.json)
# ---------------------------------------------------------------------------

# Ancienne version (Orthanc + registre info text) conservée en commentaire :
# def process_entry(ct_series_uid, seg_path, info_text, output_dir):
#     orthanc_ids = find_series_by_uid(ct_series_uid)
#     if not orthanc_ids: ...
#     uid_to_orthanc, instnum_to_uid = build_ct_index(ct_series_id)
#     findings = parse_findings(info_text)
#     for seg_num, finding in sorted(findings.items()):
#         ct_dcm = download_ct_slice(ct_orthanc_id)
#         ...


def process_accession(
    entry: dict,
    ct_series_uid: str,
    seg_path: Path,
    output_dir: Path,
    dataset_dir: Path,
) -> None:
    patient_id = entry["patient_id"]
    accession = entry["accession_number"]
    ok_findings = entry.get("ok_findings", [])

    if not ok_findings:
        print(f"  [SKIP] No ok_findings for {accession}")
        return

    print(f"  Patient {patient_id} | Accession {accession} | {len(ok_findings)} finding(s)")

    # Build CT index from local files
    print("  Indexing CT slices from local dataset...")
    uid_to_path, instnum_to_uid = build_local_ct_index(ct_series_uid, patient_id, dataset_dir)
    if not uid_to_path:
        print(f"  [SKIP] No CT slices found for series UID ...{ct_series_uid[-16:]}")
        return
    print(f"  CT index: {len(uid_to_path)} slices")

    # Load SEG and build index
    seg_dcm = pydicom.dcmread(str(seg_path))
    seg_frame_by_key, areas_by_seg, pixel_array = build_seg_index(seg_dcm)

    for finding in ok_findings:
        seg_num = finding["finding_number"]
        description = finding.get("description", "")

        # Toujours utiliser la frame SEG dont l'aire est maximale
        # (= coupe au diamètre le plus grand).
        seg_frames = areas_by_seg.get(seg_num, [])
        if not seg_frames:
            print(f"  [SKIP] Finding{seg_num}: no SEG frames found")
            continue
        best_frame_idx, best_area = max(seg_frames, key=lambda x: x[1])
        if best_area == 0:
            print(f"  [SKIP] Finding{seg_num}: all SEG frames are empty")
            continue

        frame_obj = seg_dcm.PerFrameFunctionalGroupsSequence[best_frame_idx]
        try:
            src = frame_obj.DerivationImageSequence[0].SourceImageSequence[0]
            target_ct_uid = str(src.ReferencedSOPInstanceUID)
        except (IndexError, AttributeError):
            print(f"  [SKIP] Finding{seg_num}: could not read referenced CT UID")
            continue

        ct_path = uid_to_path.get(target_ct_uid)
        if ct_path is None:
            print(f"  [SKIP] Finding{seg_num}: CT slice not found locally")
            continue

        ct_dcm = pydicom.dcmread(str(ct_path))
        actual_image_number = int(getattr(ct_dcm, "InstanceNumber", 0))
        ct_norm = normalize_ct(ct_dcm)
        seg_mask = pixel_array[best_frame_idx]

        safe_desc = re.sub(r"[^\w\-]", "_", description)[:50].strip("_")
        out_path = (
            output_dir
            / patient_id
            / f"{accession}_finding{seg_num}_{safe_desc}.png"
        )
        save_overlay(ct_norm, seg_mask, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        default="export_nodules",
        help="Dossier de sortie (défaut: export_nodules/)",
    )
    parser.add_argument(
        "--accession",
        type=str,
        default=None,
        metavar="ACC",
        help="Process only this accession number (skip others).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="DIR",
        help=f"Path to local DICOM dataset (défaut: {DATASET_DIR})",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else DATASET_DIR
    output_dir = Path(args.output)
    accession_filter = (args.accession or "").strip() or None

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    # Ancienne vérification Orthanc (remplacée par lecture locale) :
    # try:
    #     _get("/system")
    #     print(f"Connected to Orthanc: {ORTHANC_URL}")
    # except Exception as e:
    #     print(f"Cannot reach Orthanc: {e}")
    #     return

    # Load findings_validation.json — source principale des findings validés
    findings: list[dict] = json.loads(FINDINGS_FILE.read_text())
    validated = [
        e for e in findings
        if e.get("status") == "validated" and e.get("ok_findings")
    ]
    print(
        f"{len(validated)} validated accession(s) with ok_findings "
        f"(out of {len(findings)} total)\n"
    )

    # Build accession → ct_series_uid mapping from registry
    acc_to_uid = build_accession_to_uid_map()
    print(f"Registry: {len(acc_to_uid)} accession → UID mappings\n")

    if accession_filter:
        print(f"Filtering by accession: {accession_filter!r}\n")

    # Ancienne boucle (sur le registre) :
    # for ct_series_uid in entries:
    #     seg_path = seg_registry.lookup(ct_series_uid)
    #     info_text = seg_registry.lookup_info(ct_series_uid)
    #     process_entry(ct_series_uid, Path(seg_path), info_text, output_dir)

    for entry in validated:
        accession = entry["accession_number"]
        patient_id = entry["patient_id"]

        if accession_filter and accession != accession_filter:
            continue

        ct_series_uid = acc_to_uid.get(accession)
        if ct_series_uid is None:
            print(f"[SKIP] Accession {accession}: not in SEG registry")
            continue

        seg_path = seg_registry.lookup(ct_series_uid)
        if seg_path is None:
            print(f"[SKIP] Accession {accession}: SEG file not found in registry")
            continue

        print(f"Processing: Patient {patient_id} | Accession {accession}")
        try:
            process_accession(
                entry,
                ct_series_uid,
                Path(seg_path),
                output_dir,
                dataset_dir,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
