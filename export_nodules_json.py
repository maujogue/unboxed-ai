"""
Export JSON de la classe Nodule pour tous les DICOM de segmentation du registre bundlé.
Sortie : nodules_export.json

Détection de la région anatomique
----------------------------------
Il n'existe pas de tag DICOM par frame indiquant la région anatomique.
BodyPartExamined vaut 'CHEST' même pour les scans TAP → inutilisable.

Ce script utilise deux sources complémentaires :

1. StudyDescription → si le scan est thorax-seul (pas "ABDOMEN"/"PELVIS"),
   tous les nodules sont automatiquement marqués thoraciques.

2. Pour les scans TAP (Thorax-Abdomen-Pelvis), un seuil Z est calculé :
   zone thoracique = Z >= Z_ct_max - thoracic_depth
   où Z_ct_max est la position DICOM la plus supérieure (apex pulmonaire).
   Le thorax adulte faisant ~20-25 cm, le défaut de 250 mm est conservateur
   et couvre la quasi-totalité des cas cliniques.

Options:
  --output FILE            Fichier de sortie (défaut: nodules_export.json)
  --thoracic-only          N'inclure que les nodules dans la zone thoracique
  --thoracic-depth FLOAT   Profondeur (mm) depuis l'apex du scan délimitant
                           la zone thoracique pour les scans TAP (défaut: 250 mm)
"""

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import pydicom

from dcm_seg_nodules import registry as seg_registry
from src.unboxed_ai.orthanc_pipeline import extract_nodules

_NON_THORACIC_RE = re.compile(r"\b(ABDOMEN|ABDOM|PELVI[S]?)\b", re.IGNORECASE)
_THORACIC_RE = re.compile(
    r"\b(THORAX|THOR[AÀÂ]X|T[OÒÓ]RAX|CHEST|PULMON)\b", re.IGNORECASE
)


def scan_is_thorax_only(study_description: str) -> bool:
    """True si la StudyDescription indique un scan thorax exclusif.

    "TC TÒRAX"                         → True
    "TC TÒRAX, TC ABDOMEN, TC PELVIS"  → False
    ""                                 → False (inconnu : filtre Z par sécurité)

    Note : BodyPartExamined='CHEST' est présent même sur les scans TAP → ignoré.
    """
    if not study_description:
        return False
    return bool(_THORACIC_RE.search(study_description)) and not bool(
        _NON_THORACIC_RE.search(study_description)
    )


def nodule_is_thoracic(
    z_min: float, z_ct_max: float, thoracic_depth: float, thorax_only: bool
) -> bool:
    """True si le nodule est dans la zone thoracique.

    Pour les scans thorax-seul (thorax_only=True), tous les nodules sont thoraciques.
    Pour les scans TAP, on vérifie que le bord inférieur du nodule (z_min) est dans
    les `thoracic_depth` mm depuis l'apex du scanner (z_ct_max).
    """
    if thorax_only:
        return True
    if math.isnan(z_ct_max):
        return True  # pas de référence Z : on conserve par défaut
    return z_min >= (z_ct_max - thoracic_depth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", default="nodules_export.json", help="Fichier JSON de sortie"
    )
    parser.add_argument(
        "--thoracic-only",
        action="store_true",
        default=False,
        help="N'exporter que les nodules situés dans la zone thoracique",
    )
    parser.add_argument(
        "--thoracic-depth",
        type=float,
        default=250.0,
        metavar="MM",
        help=(
            "Pour les scans TAP : profondeur en mm depuis l'apex (Z_ct_max) "
            "délimitant la zone thoracique. Ignoré pour les scans thorax-seul "
            "détectés via StudyDescription. (défaut: 250 mm)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = seg_registry.list_entries()
    print(f"{len(entries)} SEG(s) trouvé(s) dans le registre.")
    if args.thoracic_only:
        print(
            f"Filtre thoracique activé (profondeur TAP: {args.thoracic_depth} mm). "
            "Les scans thorax-seul sont détectés automatiquement via StudyDescription."
        )

    export = []

    for ct_series_uid in entries:
        seg_path = seg_registry.lookup(ct_series_uid)
        info_text = seg_registry.lookup_info(ct_series_uid)

        if seg_path is None:
            print(f"  [WARN] SEG introuvable pour UID {ct_series_uid}")
            continue

        ds = pydicom.dcmread(str(seg_path), stop_before_pixels=True)
        patient_id = str(getattr(ds, "PatientID", "unknown"))
        accession = str(getattr(ds, "AccessionNumber", ""))
        study_desc = str(getattr(ds, "StudyDescription", ""))

        thorax_only = scan_is_thorax_only(study_desc)
        scan_type = "thorax-seul" if thorax_only else "TAP/multi-région"

        nodules, z_ct_max = extract_nodules(Path(seg_path), info_text)

        z_thoracic_min = (
            None
            if thorax_only or math.isnan(z_ct_max)
            else (z_ct_max - args.thoracic_depth)
        )

        nodule_dicts = []
        skipped = 0
        for n in nodules:
            thoracic = nodule_is_thoracic(
                n.z_min, z_ct_max, args.thoracic_depth, thorax_only
            )
            if args.thoracic_only and not thoracic:
                skipped += 1
                continue
            d = asdict(n)
            d["is_thoracic"] = thoracic
            nodule_dicts.append(d)

        export.append(
            {
                "patient_id": patient_id,
                "accession_number": accession,
                "seg_file": str(seg_path),
                "ct_series_uid": ct_series_uid,
                "study_description": study_desc,
                "scan_type": scan_type,
                "z_ct_max": round(z_ct_max, 1) if not math.isnan(z_ct_max) else None,
                "z_thoracic_min": round(z_thoracic_min, 1)
                if z_thoracic_min is not None
                else None,
                "nodule_count": len(nodule_dicts),
                "nodules": nodule_dicts,
            }
        )

        msg = f"  Patient {patient_id} | {scan_type} | {len(nodule_dicts)} nodule(s)"
        if skipped:
            msg += f" ({skipped} hors zone thoracique exclus)"
        print(msg)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"\nExport terminé → {output_path} ({len(export)} séries)")


if __name__ == "__main__":
    main()
