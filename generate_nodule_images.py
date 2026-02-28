"""
Génère une image fusionnée CT + masque SEG pour la coupe la plus caractéristique
de chaque nodule (Finding) dans le registre bundlé.

Stratégie de sélection de la coupe :
  1. Si le texte du registre contient "(Image X)" pour ce Finding
     → coupe CT avec InstanceNumber = X dans la série
  2. Sinon → frame SEG dont le masque a la plus grande surface
     (correspond à la coupe au diamètre maximal)

Usage:
    uv run python generate_nodule_images.py
    uv run python generate_nodule_images.py --output export_nodules/
    uv run python generate_nodule_images.py --accession 26721665  # only this accession
"""

from __future__ import annotations

import argparse
import os
import re
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import requests
from dotenv import load_dotenv

from dcm_seg_nodules import registry as seg_registry

load_dotenv()

ORTHANC_URL = os.getenv("ORTHANC_URL", "https://orthanc.unboxed-2026.ovh")
ORTHANC_USER = os.getenv("ORTHANC_USERNAME", "unboxed")
ORTHANC_PASS = os.getenv("ORTHANC_PASSWORD", "unboxed2026")
AUTH = (ORTHANC_USER, ORTHANC_PASS) if ORTHANC_USER else None


# ---------------------------------------------------------------------------
# Orthanc helpers
# ---------------------------------------------------------------------------


def _get(path: str, **kwargs) -> requests.Response:
    r = requests.get(f"{ORTHANC_URL}{path}", auth=AUTH, **kwargs)
    r.raise_for_status()
    return r


def _post(path: str, **kwargs) -> requests.Response:
    r = requests.post(f"{ORTHANC_URL}{path}", auth=AUTH, **kwargs)
    r.raise_for_status()
    return r


def find_series_by_uid(series_uid: str) -> list[str]:
    return _post(
        "/tools/find",
        json={
            "Level": "Series",
            "Query": {"SeriesInstanceUID": series_uid},
        },
    ).json()


def get_series_instances(series_id: str) -> list[dict]:
    return _get(f"/series/{series_id}/instances").json()


def get_instance_tags(instance_id: str) -> dict:
    return _get(f"/instances/{instance_id}").json().get("MainDicomTags", {})


def download_ct_slice(instance_id: str) -> pydicom.Dataset:
    r = _get(f"/instances/{instance_id}/file")
    return pydicom.dcmread(BytesIO(r.content))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_findings(info_text: str | None) -> dict[int, dict]:
    """Parse all Finding lines from the registry info text.

    Returns {seg_num: {"diameter": "17mm", "image_number": 39 or None}}.
    Example input: "- Finding1: diameter 17mm (Image 39)."
    """
    result: dict[int, dict] = {}
    if not info_text:
        return result
    for line in info_text.splitlines():
        line = line.strip()
        if not line.startswith("- Finding"):
            continue
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        try:
            num = int(parts[0].replace("- Finding", "").strip())
        except ValueError:
            continue
        detail = parts[1].strip()
        img_match = re.search(r"\(Image\s+(\d+)\)", detail)
        diam_match = re.search(r"([\d.]+\s*mm)", detail, re.IGNORECASE)
        result[num] = {
            "diameter": diam_match.group(1) if diam_match else "N/A",
            "image_number": int(img_match.group(1)) if img_match else None,
        }
    return result


# ---------------------------------------------------------------------------
# CT index building
# ---------------------------------------------------------------------------


def build_ct_index(ct_series_id: str) -> tuple[dict[str, str], dict[int, str]]:
    """Index CT instances from an Orthanc series.

    Returns:
        uid_to_orthanc: SOPInstanceUID → Orthanc instance ID
        instnum_to_uid: InstanceNumber (int) → SOPInstanceUID
    """
    uid_to_orthanc: dict[str, str] = {}
    instnum_to_uid: dict[int, str] = {}

    instances = get_series_instances(ct_series_id)
    for inst in instances:
        tags = get_instance_tags(inst["ID"])
        sop_uid = tags.get("SOPInstanceUID", "")
        if not sop_uid:
            continue
        uid_to_orthanc[sop_uid] = inst["ID"]
        inst_num_str = tags.get("InstanceNumber", "")
        if inst_num_str:
            try:
                instnum_to_uid[int(inst_num_str)] = sop_uid
            except ValueError:
                pass

    return uid_to_orthanc, instnum_to_uid


# ---------------------------------------------------------------------------
# SEG index building
# ---------------------------------------------------------------------------


def build_seg_index(
    seg_dcm: pydicom.Dataset,
) -> tuple[
    dict[tuple[int, str], int],  # (seg_num, ref_sop_uid) → frame_idx
    dict[int, list[tuple[int, int]]],  # seg_num → [(frame_idx, pixel_count)]
    np.ndarray,  # pixel_array (n_frames, H, W)
]:
    """Index SEG frames by (segment_number, referenced_CT_uid) and by area."""
    pixel_array = seg_dcm.pixel_array
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    seg_frame_by_key: dict[tuple[int, str], int] = {}
    areas_by_seg: dict[int, list[tuple[int, int]]] = {}
    frames = seg_dcm.PerFrameFunctionalGroupsSequence

    for i, frame in enumerate(frames):
        seg_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        area = int(pixel_array[i].sum())

        ref_uid: str | None = None
        if hasattr(frame, "DerivationImageSequence"):
            try:
                src = frame.DerivationImageSequence[0].SourceImageSequence[0]
                ref_uid = str(getattr(src, "ReferencedSOPInstanceUID", None))
            except (IndexError, AttributeError):
                pass

        if ref_uid:
            seg_frame_by_key[(seg_num, ref_uid)] = i
        areas_by_seg.setdefault(seg_num, []).append((i, area))

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
# Per-series processing
# ---------------------------------------------------------------------------


def process_entry(
    ct_series_uid: str,
    seg_path: Path,
    info_text: str | None,
    output_dir: Path,
) -> None:
    # Find CT series in Orthanc
    orthanc_ids = find_series_by_uid(ct_series_uid)
    if not orthanc_ids:
        print(f"  [SKIP] CT series ...{ct_series_uid[-16:]} not found in Orthanc")
        return
    ct_series_id = orthanc_ids[0]

    # Patient ID from SEG file (fastest: no extra Orthanc call)
    ds_header = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
    patient_id = str(getattr(ds_header, "PatientID", ct_series_uid[-8:]))
    accession_number = (
        str(getattr(ds_header, "AccessionNumber", "NOACC")).strip() or "NOACC"
    )

    print(f"  Patient {patient_id} | CT series: {ct_series_id[:8]}")

    # Parse findings
    findings = parse_findings(info_text)
    if not findings:
        print("  [SKIP] No findings parsed from info text")
        return

    # Load SEG pixel data and build index
    seg_dcm = pydicom.dcmread(str(seg_path))
    seg_frame_by_key, areas_by_seg, pixel_array = build_seg_index(seg_dcm)

    # Build CT index
    print("  Indexing CT instances...")
    uid_to_orthanc, instnum_to_uid = build_ct_index(ct_series_id)
    print(f"  CT index: {len(uid_to_orthanc)} instances")

    for seg_num, finding in sorted(findings.items()):
        diameter = finding["diameter"]
        image_number = finding["image_number"]

        target_ct_uid: str | None = None
        frame_idx: int | None = None

        # Strategy 1: explicit image number from info text
        if image_number is not None:
            target_ct_uid = instnum_to_uid.get(image_number)
            if target_ct_uid is None:
                print(
                    f"  [WARN] Finding{seg_num}: Image {image_number} not in CT index, falling back to max-area"
                )
            else:
                frame_idx = seg_frame_by_key.get((seg_num, target_ct_uid))

        # Strategy 2: SEG frame with the largest mask (max diameter cross-section)
        if target_ct_uid is None:
            seg_frames = areas_by_seg.get(seg_num, [])
            if not seg_frames:
                print(f"  [SKIP] Finding{seg_num}: no SEG frames found")
                continue
            best_frame_idx, best_area = max(seg_frames, key=lambda x: x[1])
            frame_idx = best_frame_idx
            frame_obj = seg_dcm.PerFrameFunctionalGroupsSequence[best_frame_idx]
            try:
                src = frame_obj.DerivationImageSequence[0].SourceImageSequence[0]
                target_ct_uid = str(src.ReferencedSOPInstanceUID)
            except (IndexError, AttributeError):
                print(f"  [SKIP] Finding{seg_num}: could not read referenced CT UID")
                continue

        ct_orthanc_id = uid_to_orthanc.get(target_ct_uid)
        if ct_orthanc_id is None:
            print(f"  [SKIP] Finding{seg_num}: CT instance not in Orthanc")
            continue

        # Download CT slice and render
        ct_dcm = download_ct_slice(ct_orthanc_id)
        ct_norm = normalize_ct(ct_dcm)
        seg_mask = pixel_array[frame_idx] if frame_idx is not None else None

        safe_diam = diameter.replace(" ", "").replace("/", "-")
        out_path = (
            output_dir
            / patient_id
            / f"{accession_number}_finding{seg_num}_{safe_diam}.png"
        )
        title = f"Patient {patient_id} | Acc {accession_number} | Finding {seg_num} | Ø {diameter}"
        save_overlay(ct_norm, seg_mask, out_path, title=title)


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
    args = parser.parse_args()

    output_dir = Path(args.output)
    accession_filter = (args.accession or "").strip() or None

    try:
        _get("/system")
        print(f"Connected to Orthanc: {ORTHANC_URL}")
    except Exception as e:
        print(f"Cannot reach Orthanc: {e}")
        return

    entries = seg_registry.list_entries()
    print(f"{len(entries)} SEG(s) in registry\n")
    if accession_filter:
        print(f"Filtering by accession: {accession_filter!r}\n")

    for ct_series_uid in entries:
        seg_path = seg_registry.lookup(ct_series_uid)
        info_text = seg_registry.lookup_info(ct_series_uid)

        if seg_path is None:
            print(f"[WARN] SEG not found for UID ...{ct_series_uid[-16:]}")
            continue

        if accession_filter:
            ds_header = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
            acc = str(getattr(ds_header, "AccessionNumber", "") or "").strip()
            if acc != accession_filter:
                continue

        print(f"Processing: ...{ct_series_uid[-16:]}")
        try:
            process_entry(ct_series_uid, Path(seg_path), info_text, output_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
