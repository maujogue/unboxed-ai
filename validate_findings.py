#!/usr/bin/env python3
"""
validate_findings.py

Cross-references pulmonary lesions from clinical reports (ODS) against
the actual DICOM SEG files (ground truth) stored in dcm_seg_nodules.

Ground truth = DICOM SEG files (segment descriptions + covered slices).
The registry info text is NOT used as ground truth (it can be incomplete).

Flags generated per finding in the report:
  OK             – (FX) tag + SEG has a segment for FindingX + image covered
  IMAGE_MISMATCH – (FX) tag + SEG has the segment but doesn't cover the image
  UNCERTAIN      – (FX) tag + SEG has NO segment for FindingX (or no SEG at all)
  TO_CHECK       – pulmonary lesion mentioned without any (FX) tag

Output:
  findings_validation.json  (machine-readable, includes image flags)
  Console summary
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
import requests
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

load_dotenv()

# ──────────────────────────── Config ───────────────────────────────────────
ODS_PATH = Path("assets/new-reports.ods")
OUTPUT_PATH = Path("findings_validation.json")
SEG_DATA_DIR = Path(".venv/lib/python3.13/site-packages/dcm_seg_nodules/data")

ORTHANC_URL = os.getenv("ORTHANC_URL", "https://orthanc.unboxed-2026.ovh").rstrip("/")
ORTHANC_AUTH = (
    os.getenv("ORTHANC_USERNAME", "unboxed"),
    os.getenv("ORTHANC_PASSWORD", "unboxed2026"),
)


# ──────────────────────────── Pydantic schemas ──────────────────────────────
class ExtractedFinding(BaseModel):
    finding_number: Optional[int] = Field(
        default=None,
        description="Integer X from the tag (FX) if present, null if no (FX) tag",
    )
    image_number: Optional[int] = Field(
        default=None,
        description="Image/slice number mentioned near this lesion (e.g. 'Image 9' → 9), null if none",
    )
    description: str = Field(
        description="Brief verbatim description of the lesion as written in the report"
    )
    is_pulmonary: bool = Field(
        description="True if the lesion is clearly located in lung parenchyma"
    )
    is_non_target: bool = Field(
        description="True if the text explicitly labels this lesion as 'NON-TARGET lesion'"
    )


class ReportParseResult(BaseModel):
    findings: list[ExtractedFinding] = Field(
        description="ALL pulmonary lesions found in the report (tagged or not, excluding NON-TARGET)"
    )


# ──────────────────────────── Registry helpers ──────────────────────────────
def build_acc_to_seg_map() -> dict[str, list[Path]]:
    """Return {accession_number: [seg_file_paths...]} from dcm_seg_nodules registry."""
    from dcm_seg_nodules import registry as seg_registry

    entries = seg_registry.list_entries()
    result: dict[str, list[Path]] = {}
    for val in entries.values():
        info = val.get("info", "")
        acc_match = re.search(r"Accession Number:\s*(\S+)", info)
        if not acc_match:
            continue
        acc_num = acc_match.group(1)
        seg_file = SEG_DATA_DIR / val["seg_file"]
        if seg_file.exists():
            result.setdefault(acc_num, []).append(seg_file)
    return result


# ──────────────────────────── Orthanc SEG fallback ─────────────────────────


def find_seg_in_orthanc(acc_num: str, tmp_dir: Path) -> list[Path]:
    """Download SEG series from Orthanc for a given AccessionNumber.

    Used as fallback when the accession is absent from the dcm_seg_nodules registry
    (e.g. manual annotations uploaded directly to Orthanc).
    """
    try:
        r = requests.post(
            f"{ORTHANC_URL}/tools/find",
            json={"Level": "Study", "Query": {"AccessionNumber": acc_num}},
            auth=ORTHANC_AUTH,
            timeout=10,
        )
        study_ids = r.json()
    except Exception as e:
        print(f"    Warning: Orthanc query failed for AccNum={acc_num}: {e}")
        return []

    seg_paths: list[Path] = []
    for study_id in study_ids:
        try:
            study_info = requests.get(
                f"{ORTHANC_URL}/studies/{study_id}", auth=ORTHANC_AUTH, timeout=10
            ).json()
            for series_id in study_info.get("Series", []):
                series_info = requests.get(
                    f"{ORTHANC_URL}/series/{series_id}", auth=ORTHANC_AUTH, timeout=10
                ).json()
                if series_info.get("MainDicomTags", {}).get("Modality") != "SEG":
                    continue
                for inst_id in series_info.get("Instances", []):
                    r = requests.get(
                        f"{ORTHANC_URL}/instances/{inst_id}/file",
                        auth=ORTHANC_AUTH,
                        timeout=30,
                    )
                    dest = tmp_dir / f"orthanc_seg_{inst_id}.dcm"
                    dest.write_bytes(r.content)
                    seg_paths.append(dest)
        except Exception as e:
            print(f"    Warning: error fetching SEG from Orthanc study {study_id}: {e}")

    return seg_paths


# ──────────────────────────── SEG DICOM analysis ───────────────────────────
def _orthanc_instance_number(sop_uid: str) -> Optional[int]:
    """Query Orthanc to get InstanceNumber for a given SOPInstanceUID."""
    try:
        r = requests.post(
            f"{ORTHANC_URL}/tools/find",
            json={"Level": "Instance", "Query": {"SOPInstanceUID": sop_uid}},
            auth=ORTHANC_AUTH,
            timeout=10,
        )
        ids = r.json()
        if not ids:
            return None
        tags = requests.get(
            f"{ORTHANC_URL}/instances/{ids[0]}/tags?simplify",
            auth=ORTHANC_AUTH,
            timeout=10,
        ).json()
        val = tags.get("InstanceNumber")
        return int(val) if val is not None else None
    except Exception:
        return None


def get_seg_coverage(seg_paths: list[Path]) -> dict[int, list[int]]:
    """
    For a list of SEG DICOM files (same AccessionNumber, possibly multiple series),
    return {finding_number: [image_instance_numbers...]} where finding_number comes
    from the segment description (e.g. 'Finding.3' → 3).

    Only frames with non-zero pixel data are counted.
    """
    result: dict[int, list[int]] = {}

    for seg_path in seg_paths:
        ds = pydicom.dcmread(str(seg_path))
        pixel_array = ds.pixel_array  # (n_frames, H, W)
        frames = ds.PerFrameFunctionalGroupsSequence

        # Segment number → finding number from description
        seg_num_to_finding: dict[int, int] = {}
        for seg in ds.SegmentSequence:
            desc = seg.SegmentDescription
            match = re.search(r"Finding[.\s_]?(\d+)", desc, re.IGNORECASE)
            if match:
                seg_num_to_finding[int(seg.SegmentNumber)] = int(match.group(1))

        # Collect active (non-zero) SOPInstanceUIDs per finding
        finding_to_uids: dict[int, set[str]] = defaultdict(set)
        for i, frame in enumerate(frames):
            if not np.any(pixel_array[i]):
                continue
            seg_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
            finding_num = seg_num_to_finding.get(seg_num)
            if finding_num is None:
                continue
            sop = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            finding_to_uids[finding_num].add(sop)

        # Resolve SOPInstanceUIDs → InstanceNumbers via Orthanc
        for finding_num, uids in finding_to_uids.items():
            nums: list[int] = []
            for sop in uids:
                n = _orthanc_instance_number(sop)
                if n is not None:
                    nums.append(n)
            if finding_num in result:
                result[finding_num] = sorted(set(result[finding_num]) | set(nums))
            else:
                result[finding_num] = sorted(nums)

    return result


# ──────────────────────────── LLM extraction ───────────────────────────────
SYSTEM_PROMPT = """\
You are a radiology report parser specialized in thoracic oncology CT scans.

Your task: extract ALL pulmonary lesions/nodules from the report that are NOT explicitly \
labeled "NON-TARGET lesion". Include BOTH:
  - Lesions carrying a finding tag such as (F1), (F2), (F3), etc.
  - Lesions described WITHOUT any (FX) tag.

For each lesion return:
  - finding_number  : integer X from tag (FX) if present, else null
  - image_number    : CT image/slice number mentioned near this lesion
                      (e.g. "Image 9", "image-9", "Figure 25", "(image 125)" → 9/9/25/125),
                      null if no image number is mentioned
  - description     : brief verbatim description from the report
  - is_pulmonary    : true if clearly in lung parenchyma (nodule, mass, condensation in a lobe/segment)
  - is_non_target   : true if explicitly labeled "NON-TARGET lesion"

Exclude: NON-TARGET lesions, non-pulmonary lesions (liver, bone, adrenal, lymph nodes), normal structures.
If no qualifying lesions exist, return an empty findings list.\
"""


def extract_findings_llm(llm: ChatMistralAI, report_text: str) -> ReportParseResult:
    structured = llm.with_structured_output(ReportParseResult)
    return structured.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Report:\n\n{report_text}"),
    ])


# ──────────────────────────── Per-accession processing ─────────────────────


def process_accession(
    acc_num: str,
    patient_id: str,
    report_text: str,
    acc_to_seg: dict[str, list[Path]],
    llm: ChatMistralAI,
    tmp_dir: Path,
) -> dict:
    """Validate findings for a single accession number.

    Returns a result dict with keys: patient_id, accession_number, seg_coverage,
    status, ok_findings, uncertain_findings, image_mismatch_findings,
    to_check_findings, llm_extracted_findings.
    """
    seg_paths = acc_to_seg.get(acc_num, [])
    seg_source = "registry"

    if not seg_paths:
        orthanc_segs = find_seg_in_orthanc(acc_num, tmp_dir)
        if orthanc_segs:
            seg_paths = orthanc_segs
            seg_source = "orthanc"
            print(f"    Fallback Orthanc: {len(orthanc_segs)} SEG(s) trouvé(s) pour AccNum={acc_num}")

    # ── LLM report parsing ─────────────────────────────────────────────────
    print(f"  Parsing report AccNum={acc_num} (PatientID={patient_id}) …")
    parse_result = extract_findings_llm(llm, report_text)
    relevant = [f for f in parse_result.findings if f.is_pulmonary and not f.is_non_target]

    # ── SEG coverage (Orthanc queries) ─────────────────────────────────────
    seg_coverage: Optional[dict[int, list[int]]] = None
    if seg_paths:
        print(f"    Loading SEG ({seg_source}) + querying Orthanc for AccNum={acc_num} …")
        seg_coverage = get_seg_coverage(seg_paths)
        print(f"    SEG findings: { {k: v for k, v in seg_coverage.items()} }")

    # ── Classification ──────────────────────────────────────────────────────
    ok: list[dict] = []
    uncertain: list[dict] = []
    image_mismatch: list[dict] = []
    to_check: list[dict] = []

    for finding in relevant:
        fn = finding.finding_number
        img = finding.image_number
        base = {
            "finding_number": fn,
            "image_number": img,
            "description": finding.description,
        }

        if fn is None:
            to_check.append({**base, "flag": "TO_CHECK", "reason": "no_fx_tag_in_report"})
        elif seg_coverage is None:
            uncertain.append({**base, "flag": "UNCERTAIN", "reason": "no_seg_file_for_accession"})
        elif fn not in seg_coverage:
            uncertain.append({
                **base,
                "flag": "UNCERTAIN",
                "reason": f"F{fn}_not_in_seg_segments_{sorted(seg_coverage.keys())}",
            })
        elif img is not None and img not in seg_coverage[fn]:
            image_mismatch.append({
                **base,
                "flag": "IMAGE_MISMATCH",
                "reason": f"Image_{img}_not_in_seg_coverage_{seg_coverage[fn]}",
            })
        else:
            ok.append({**base, "flag": "OK"})

    if uncertain:
        status = "uncertain"
    elif image_mismatch:
        status = "image_mismatch"
    elif to_check:
        status = "to_check"
    elif ok:
        status = "validated"
    else:
        status = "no_pulmonary_findings"

    return {
        "patient_id": patient_id,
        "accession_number": acc_num,
        "seg_coverage": seg_coverage,
        "status": status,
        "ok_findings": ok,
        "uncertain_findings": uncertain,
        "image_mismatch_findings": image_mismatch,
        "to_check_findings": to_check,
        "llm_extracted_findings": [
            {
                "finding_number": f.finding_number,
                "image_number": f.image_number,
                "description": f.description,
                "is_pulmonary": f.is_pulmonary,
                "is_non_target": f.is_non_target,
            }
            for f in parse_result.findings
        ],
    }


def print_summary(results: list[dict]) -> None:
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        acc = r["accession_number"]
        pid = r["patient_id"]
        cov = r["seg_coverage"]
        cov_str = str(sorted(cov.keys())) if cov else "NO SEG"

        if r["uncertain_findings"]:
            print(f"\n[UNCERTAIN]       PatientID={pid}  AccNum={acc}")
            print(f"  SEG findings: {cov_str}")
            for u in r["uncertain_findings"]:
                tag = f"F{u['finding_number']}" if u["finding_number"] else "(no tag)"
                img = f" @ Image {u['image_number']}" if u["image_number"] else ""
                print(f"  ⚠  {tag}{img}: {u['description']}")
                print(f"     → {u['reason']}")

        if r["image_mismatch_findings"]:
            print(f"\n[IMAGE MISMATCH]  PatientID={pid}  AccNum={acc}")
            print(f"  SEG findings: {cov_str}")
            for m in r["image_mismatch_findings"]:
                print(f"  ⚡  F{m['finding_number']} @ Image {m['image_number']}: {m['description']}")
                print(f"     → {m['reason']}")

        if r["to_check_findings"] and not r["uncertain_findings"] and not r["image_mismatch_findings"]:
            print(f"\n[TO CHECK]        PatientID={pid}  AccNum={acc}")
            print(f"  SEG findings: {cov_str}")
            for t in r["to_check_findings"]:
                img = f" @ Image {t['image_number']}" if t["image_number"] else ""
                print(f"  ?  (no tag){img}: {t['description']}")

        if not r["uncertain_findings"] and not r["image_mismatch_findings"] and not r["to_check_findings"]:
            if r["status"] == "validated":
                details = ", ".join(
                    f"F{f['finding_number']}@img{f['image_number']}" if f["image_number"]
                    else f"F{f['finding_number']}"
                    for f in r["ok_findings"]
                )
                print(f"[OK]              PatientID={pid}  AccNum={acc}  → {details}")
            else:
                print(f"[NO FINDINGS]     PatientID={pid}  AccNum={acc}")

    counts = {
        "uncertain": sum(1 for r in results if r["uncertain_findings"]),
        "image_mismatch": sum(1 for r in results if r["image_mismatch_findings"] and not r["uncertain_findings"]),
        "to_check": sum(1 for r in results if r["to_check_findings"] and not r["uncertain_findings"] and not r["image_mismatch_findings"]),
        "ok": sum(1 for r in results if r["status"] == "validated"),
        "no_findings": sum(1 for r in results if r["status"] == "no_pulmonary_findings"),
    }
    print(f"\nTotal: {counts['ok']} OK | {counts['uncertain']} uncertain | "
          f"{counts['image_mismatch']} image-mismatch | {counts['to_check']} to-check | "
          f"{counts['no_findings']} no pulmonary findings")


# ──────────────────────────── Main ─────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate pulmonary findings against DICOM SEG.")
    parser.add_argument("--accession", metavar="ACC_NUM", help="Process a single accession number only")
    parser.add_argument("--patient", metavar="PATIENT_ID", help="Process all accessions for a single patient")
    args = parser.parse_args()

    df = pd.read_excel(ODS_PATH, engine="odf")
    acc_to_seg = build_acc_to_seg_map()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")

    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key, temperature=0.0)  # type: ignore[call-arg]

    # ── Filter rows ────────────────────────────────────────────────────────
    if args.accession:
        df = df[df["AccessionNumber"].astype(str).str.strip() == args.accession.strip()]
        if df.empty:
            print(f"AccessionNumber {args.accession!r} not found in {ODS_PATH}")
            return
    elif args.patient:
        df = df[df["PatientID"].astype(str) == args.patient]
        if df.empty:
            print(f"PatientID {args.patient!r} not found in {ODS_PATH}")
            return

    results = []
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)
        for _, row in df.iterrows():
            acc_num = str(int(row["AccessionNumber"]))
            patient_id = str(row["PatientID"])
            report_text = str(row["Clinical information data (Pseudo reports)"])
            result = process_accession(acc_num, patient_id, report_text, acc_to_seg, llm, tmp_dir)
            results.append(result)

    # ── Save ───────────────────────────────────────────────────────────────
    if args.accession:
        # Single accession: update existing JSON rather than overwrite everything
        existing = []
        if OUTPUT_PATH.exists():
            existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        existing = [r for r in existing if r["accession_number"] != args.accession]
        existing.extend(results)
        results_to_save = existing
    else:
        results_to_save = results

    OUTPUT_PATH.write_text(json.dumps(results_to_save, indent=2, ensure_ascii=False))
    print(f"\nResults saved → {OUTPUT_PATH}\n")

    print_summary(results)


if __name__ == "__main__":
    main()
