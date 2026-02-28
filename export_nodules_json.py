"""
Export JSON de la classe Nodule pour tous les DICOM de segmentation du registre bundlé.
Sortie : nodules_export.json
"""

import json
from dataclasses import asdict
from pathlib import Path

import pydicom

from dcm_seg_nodules import registry as seg_registry
from src.unboxed_ai.orthanc_pipeline import extract_nodules

OUTPUT_FILE = "nodules_export.json"

entries = seg_registry.list_entries()
print(f"{len(entries)} SEG(s) trouvé(s) dans le registre.")

export = []

for ct_series_uid in entries:
    seg_path = seg_registry.lookup(ct_series_uid)
    info_text = seg_registry.lookup_info(ct_series_uid)

    if seg_path is None:
        print(f"  [WARN] SEG introuvable pour UID {ct_series_uid}")
        continue

    # Lire les métadonnées patient depuis le fichier DICOM
    ds = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
    patient_id = str(getattr(ds, "PatientID", "unknown"))
    accession = str(getattr(ds, "AccessionNumber", ""))

    nodules = extract_nodules(Path(seg_path), info_text)

    export.append({
        "patient_id": patient_id,
        "accession_number": accession,
        "seg_file": str(seg_path),
        "ct_series_uid": ct_series_uid,
        "nodule_count": len(nodules),
        "nodules": [asdict(n) for n in nodules],
    })

    print(f"  Patient {patient_id} | AccessionNumber {accession} | {len(nodules)} nodule(s)")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(export, f, indent=2, ensure_ascii=False)

print(f"\nExport terminé → {OUTPUT_FILE} ({len(export)} séries)")
