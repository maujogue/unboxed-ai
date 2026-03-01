"""
Interface radiologiste — Synthèse agentic de l'évolution des lésions.

Pipeline :
  1. Agent récupération données  (ODS + findings_validation + nodules_export + images locales)
  2. LLM judge                    (LUNGRAD vs RECIST)
  3. LLM synthèse évolution       (Mistral, basé sur rapports validés + mesures SEG)
  4. Rendu HTML                   (images CT+SEG embarquées, section lésions SEG non rapportées)

Orthanc est simulé localement tant que le serveur est down.

Usage:
    uv run python frontend/front.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Generator

import gradio as gr
import matplotlib
import pandas as pd
import pydicom
from dotenv import load_dotenv
from mistralai import Mistral
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
REPORTS_ODS = ROOT / "assets" / "new-reports.ods"
FINDINGS_FILE = ROOT / "findings_validation.json"
NODULES_FILE = ROOT / "nodules_export.json"
IMAGES_DIR = ROOT / "export_nodules"
SEG_VALIDATION_FILE = ROOT / "seg_validation.json"

DATASET_DIR = Path("/home/corentin/Downloads/dataset_backup(1)/dataset")

MISTRAL_MODEL = "mistral-large-latest"

_mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

class LesionMeasurement(BaseModel):
    date: str
    diameter_mm: float
    delta_mm: float | None = None      # calculé par Python
    interpretation: str = ""           # fourni par le LLM

class Lesion(BaseModel):
    id: str                            # "F1", "F2", ...
    localisation: str                  # estimée depuis x/y/z
    relation_spatiale: str = ""        # position relative aux autres lésions (distance, direction)
    measurements: list[LesionMeasurement]

class EvolutionReport(BaseModel):
    lesions: list[Lesion]
    autres_observations: str = ""


# Cache module-level pour contourner les problèmes de propagation des gr.State
# depuis un générateur vers d'autres boutons (comportement connu de Gradio).
_pipeline_cache: dict = {"patient_data": {}, "report_structure": "lungrad"}


# ---------------------------------------------------------------------------
# Helper — résolution de la date d'examen depuis l'arborescence locale
# ---------------------------------------------------------------------------


def estimate_lobe(x: float | None, y: float | None, z: float | None,
                  z_thoracic_min: float = 0.0, z_ct_max: float = 1.0) -> str:
    """Estime le lobe pulmonaire depuis les coordonnées DICOM (x, y, z en mm)."""
    if x is None or z is None:
        return ""
    thorax_h = z_ct_max - z_thoracic_min
    if thorax_h <= 0:
        return ""
    z_rel = (z - z_thoracic_min) / thorax_h  # 0 = bas, 1 = haut
    side = "gauche" if x > 0 else "droit"
    if side == "droit":
        if z_rel > 0.65:
            lobe = "lobe supérieur"
        elif z_rel > 0.35:
            lobe = "lobe moyen" if (y is not None and y < 0) else "lobe inférieur"
        else:
            lobe = "lobe inférieur"
    else:
        lobe = "lobe supérieur" if z_rel > 0.55 else "lobe inférieur"
    return f"{lobe} {side}"


def compute_spatial_relations(findings_by_num: dict[int, dict]) -> dict[int, str]:
    """
    Calcule la position relative de chaque lésion par rapport aux autres
    depuis les coordonnées de centroïde (x, y, z). Retourne {finding_number: description}.
    """
    nums = [n for n, f in findings_by_num.items() if f.get("x") is not None]
    if len(nums) < 2:
        return {}

    result: dict[int, str] = {}
    for num in nums:
        f = findings_by_num[num]
        cx, cy, cz = f["x"], f["y"], f["z"]
        others = []
        for other_num in nums:
            if other_num == num:
                continue
            o = findings_by_num[other_num]
            dx = o["x"] - cx
            dy = o["y"] - cy
            dz = o["z"] - cz
            dist = round((dx**2 + dy**2 + dz**2) ** 0.5)
            # Direction dominante
            dirs = []
            if abs(dz) >= 10:
                dirs.append("supérieur" if dz > 0 else "inférieur")
            if abs(dx) >= 10:
                dirs.append("médial" if (cx > 0 and dx < 0) or (cx < 0 and dx > 0) else "latéral")
            direction = "/".join(dirs) if dirs else "adjacent"
            others.append(f"F{other_num} à {dist} mm ({direction})")
        result[num] = ", ".join(others)
    return result


def _get_study_date_from_dicom(patient_id: str, accession: str) -> str:
    """Lit StudyDate depuis les métadonnées DICOM. Fallback sur la méthode par dossier."""
    for patient_dir in DATASET_DIR.glob(f"*{patient_id}*"):
        for acc_dir in patient_dir.rglob(f"{accession}*"):
            if not acc_dir.is_dir():
                continue
            for dcm_file in acc_dir.rglob("*.dcm"):
                try:
                    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                    raw = str(getattr(ds, "StudyDate", "") or getattr(ds, "AcquisitionDate", ""))
                    if len(raw) == 8 and raw.isdigit():
                        return f"{raw[6:8]}/{raw[4:6]}/{raw[:4]}"
                except Exception:
                    continue
    return _get_study_date(patient_id, accession)


def _get_study_date(patient_id: str, accession: str) -> str:
    """
    Retourne la date d'examen (DD/MM/YYYY) depuis findings_validation.json (study_date),
    en fallback sur la structure du dossier DICOM local.
    """
    # 1. Lire depuis findings_validation.json (portable, pas besoin du dataset local)
    try:
        findings = json.loads(FINDINGS_FILE.read_text())
        for e in findings:
            if e.get("patient_id") == patient_id and str(e.get("accession_number")) == str(accession):
                d = e.get("study_date")
                if d:
                    return d
    except Exception:
        pass
    # 2. Fallback : structure du dossier DICOM local
    for patient_dir in DATASET_DIR.glob(f"*{patient_id}*"):
        for acc_dir in patient_dir.rglob(f"{accession}*"):
            if not acc_dir.is_dir():
                continue
            date_folder = acc_dir.parent.parent.name
            if len(date_folder) >= 8 and date_folder[:8].isdigit():
                d = date_folder[:8]  # YYYYMMDD
                return f"{d[6:8]}/{d[4:6]}/{d[0:4]}"
    return accession  # fallback final


# ---------------------------------------------------------------------------
# SEG validation helpers
# ---------------------------------------------------------------------------


def _load_seg_validations() -> dict:
    if SEG_VALIDATION_FILE.exists():
        return json.loads(SEG_VALIDATION_FILE.read_text())
    return {}


def _get_or_create_seg_image(
    patient_id: str,
    accession: str,
    seg_num: int,
    image_numbers: list,
    nodule_entry: dict,
) -> str | None:
    """Return a CT+SEG overlay image for a SEG-only segment, generating it if missing."""
    out_dir = IMAGES_DIR / patient_id
    cached = list(out_dir.glob(f"{accession}_seg{seg_num}_unlabeled.png"))
    if cached:
        return str(cached[0])

    seg_file = nodule_entry.get("seg_file")
    ct_series_uid = nodule_entry.get("ct_series_uid")
    if not seg_file or not ct_series_uid:
        return None

    seg_path = Path(seg_file)
    if not seg_path.exists():
        # Le chemin peut être codé en dur sur une autre machine — chercher par nom
        # dans l'installation locale du package dcm_seg_nodules.
        try:
            import dcm_seg_nodules  # noqa: PLC0415
            local_path = Path(dcm_seg_nodules.__file__).parent / "data" / seg_path.name
            if local_path.exists():
                seg_path = local_path
            else:
                return None
        except Exception:
            return None

    try:
        sys.path.insert(0, str(ROOT))
        from generate_nodule_images import (  # noqa: PLC0415
            build_local_ct_index,
            build_seg_index,
            normalize_ct,
            save_overlay,
        )

        uid_to_path, instnum_to_uid = build_local_ct_index(ct_series_uid, patient_id, DATASET_DIR)
        if not uid_to_path:
            return None

        seg_dcm = pydicom.dcmread(str(seg_path))
        seg_frame_by_key, areas_by_seg, pixel_array = build_seg_index(seg_dcm)

        # Toujours utiliser la frame SEG dont l'aire est maximale.
        seg_frames = areas_by_seg.get(seg_num, [])
        if not seg_frames:
            return None
        best_frame_idx, best_area = max(seg_frames, key=lambda x: x[1])
        if best_area == 0:
            return None

        frame_obj = seg_dcm.PerFrameFunctionalGroupsSequence[best_frame_idx]
        try:
            src = frame_obj.DerivationImageSequence[0].SourceImageSequence[0]
            target_ct_uid = str(src.ReferencedSOPInstanceUID)
        except (IndexError, AttributeError):
            return None

        ct_path = uid_to_path.get(target_ct_uid or "")
        if not ct_path:
            return None

        ct_dcm = pydicom.dcmread(str(ct_path))
        ct_norm = normalize_ct(ct_dcm)
        seg_mask = pixel_array[best_frame_idx]

        out_path = out_dir / f"{accession}_seg{seg_num}_unlabeled.png"
        save_overlay(ct_norm, seg_mask, out_path)
        return str(out_path)
    except Exception as e:
        print(f"[WARN] Could not generate seg image for seg{seg_num}: {e}")
        return None


def _build_seg_rows(patient_id: str, visits: list) -> list:
    """Build enriched SEG-only rows with image paths for validation.

    Includes segments from ALL visits, including the most recent one.
    Rows from the latest visit are flagged with is_latest_visit=True so the
    UI can warn the radiologist that these segments belong to the exam whose
    report will be generated.
    """
    all_nodules: list[dict] = json.loads(NODULES_FILE.read_text())
    nodules_by_acc = {
        str(e["accession_number"]): e
        for e in all_nodules
        if e["patient_id"] == patient_id
    }
    rows = []
    for v in visits:
        acc = v["accession"]
        is_latest = bool(v.get("is_latest"))
        label = v.get("date") or acc
        nodule_entry = nodules_by_acc.get(acc, {})
        for s in v.get("seg_only_segments", []):
            seg_num = s["seg_number"]
            image_path = _get_or_create_seg_image(
                patient_id, acc, seg_num, s["image_numbers"], nodule_entry
            )
            rows.append({
                "patient_id": patient_id,
                "accession": acc,
                "label": label,
                "seg_number": seg_num,
                "image_numbers": s["image_numbers"],
                "image_path": image_path,
                "is_latest_visit": is_latest,
            })
    return rows


def _seg_show(seg_rows: list, idx: int, decisions: dict) -> tuple:
    """Return display values for a given segment index.

    Returns 5 values: (seg_validation_col, progress, img, info, seg_controls_col)
    """
    if not seg_rows or idx >= len(seg_rows):
        return gr.update(visible=False), "", None, "", gr.update(visible=False)

    row = seg_rows[idx]
    patient_id = row["patient_id"]
    key = f"{patient_id}_{row['accession']}_{row['seg_number']}"
    n = len(seg_rows)

    progress = f"**Segment {idx + 1} / {n}** — {row['label']}"
    latest_flag = " ⭐ _Examen le plus récent — rapport à générer_" if row.get("is_latest_visit") else ""
    info = (
        f"**Segment SEG n°{row['seg_number']}**{latest_flag}\n\n"
        f"Coupes couvertes : {', '.join(str(i) for i in row['image_numbers'])}"
    )
    if key in decisions:
        marker = "✅ Oui" if decisions[key] else "❌ Non"
        info += f"\n\n_Décision actuelle : {marker}_"

    img_path = row.get("image_path")
    img = None
    if img_path and Path(img_path).exists():
        try:
            img = plt.imread(img_path)
        except Exception:
            img = None

    return gr.update(visible=True), progress, img, info, gr.update(visible=True)


def _start_seg_validation(seg_rows: list, patient_id: str) -> tuple:
    seg_rows = _pipeline_cache.get("seg_rows", seg_rows or [])
    existing = _load_seg_validations()
    decisions = {
        f"{r['patient_id']}_{r['accession']}_{r['seg_number']}": existing[
            f"{r['patient_id']}_{r['accession']}_{r['seg_number']}"
        ]
        for r in seg_rows
        if f"{r['patient_id']}_{r['accession']}_{r['seg_number']}" in existing
    }
    if not seg_rows:
        return gr.update(visible=False), "", None, "_Aucun segment SEG sans rapport._", 0, decisions, gr.update(visible=False)
    col_upd, progress, img, info, ctrl_upd = _seg_show(seg_rows, 0, decisions)
    return col_upd, progress, img, info, 0, decisions, ctrl_upd


def _on_validate(is_real: bool, seg_rows: list, idx: int, decisions: dict) -> tuple:
    if not seg_rows or idx >= len(seg_rows):
        return gr.update(visible=False), "", None, "", idx, decisions, gr.update(visible=False)
    row = seg_rows[idx]
    key = f"{row['patient_id']}_{row['accession']}_{row['seg_number']}"
    decisions = {**decisions, key: is_real}
    next_idx = idx + 1
    if next_idx >= len(seg_rows):
        # Tous les segments traités : masquer image + boutons Oui/Non
        return (
            gr.update(visible=True),
            f"**{len(seg_rows)} / {len(seg_rows)}** — Tous les segments traités ✓",
            None,
            "_Tous les segments ont été évalués. Cliquez sur **Sauvegarder** pour enregistrer._",
            next_idx,
            decisions,
            gr.update(visible=False),  # seg_controls_col
        )
    col_upd, progress, img, info, ctrl_upd = _seg_show(seg_rows, next_idx, decisions)
    return col_upd, progress, img, info, next_idx, decisions, ctrl_upd


def _on_oui(seg_rows: list, idx: int, decisions: dict) -> tuple:
    return _on_validate(True, seg_rows, idx, decisions)


def _on_non(seg_rows: list, idx: int, decisions: dict) -> tuple:
    return _on_validate(False, seg_rows, idx, decisions)


def _save_seg(seg_rows: list, decisions: dict) -> str:
    if not decisions:
        return "⚠️ Aucune décision à sauvegarder."
    validations = _load_seg_validations()
    validations.update(decisions)
    SEG_VALIDATION_FILE.write_text(json.dumps(validations, ensure_ascii=False, indent=2))
    return f"✅ {len(decisions)} décision(s) sauvegardée(s) dans `seg_validation.json`."


def _save_and_generate(
    seg_rows: list, decisions: dict, patient_data: dict, report_structure: str
):
    """Sauvegarde les validations SEG puis génère le rapport de la visite la plus récente.

    Générateur :
      - 1er yield : cache la zone de validation, affiche le spinner de génération
      - 2e yield : affiche le rapport final
    Retourne 6 valeurs : seg_save_status, seg_section_col, seg_validation_col,
                         new_report_col, new_report_status_md, new_report_html_out
    """
    save_msg = _save_seg(seg_rows, decisions)
    if not patient_data:
        patient_data = _pipeline_cache.get("patient_data", {})
    if not report_structure:
        report_structure = _pipeline_cache.get("report_structure", "lungrad")

    loading_text = "## ⏳ Génération du rapport final en cours…\n_Le LLM analyse l'historique des visites et rédige le compte-rendu radiologique. Cette étape prend généralement 15 à 30 secondes._"

    # 1er yield : message de chargement bien visible au-dessus des boutons
    yield (
        save_msg,                       # seg_save_status
        loading_text,                   # seg_loading_md
        gr.update(visible=True),        # seg_section_col
        gr.update(visible=True),        # seg_validation_col
        gr.update(visible=False),       # seg_controls_col
        gr.update(visible=False),       # new_report_col
        gr.update(visible=False),       # gen_final_report_btn
        "", "",                         # new_report_status_md, new_report_html_out
    )

    try:
        col_upd, report_status, report_html = _generate_latest_report(patient_data, report_structure)
    except Exception as e:
        yield f"❌ Erreur : {e}", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", ""
        return

    # 2e yield : rapport prêt → cacher la section validation, afficher le rapport
    yield save_msg, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), col_upd, gr.update(visible=False), report_status, report_html


def _reset_seg(seg_rows: list, patient_id: str) -> tuple:
    """Supprime toutes les validations du patient courant et repart du début."""
    validations = _load_seg_validations()
    keys_to_delete = [
        f"{r['patient_id']}_{r['accession']}_{r['seg_number']}" for r in seg_rows
    ]
    for k in keys_to_delete:
        validations.pop(k, None)
    SEG_VALIDATION_FILE.write_text(json.dumps(validations, ensure_ascii=False, indent=2))

    # Repart du premier segment, décisions vides
    if not seg_rows:
        return gr.update(visible=False), "", None, "", 0, {}, gr.update(visible=False), "🗑️ Rien à réinitialiser."
    col_upd, progress, img, info, ctrl_upd = _seg_show(seg_rows, 0, {})
    return col_upd, progress, img, info, 0, {}, ctrl_upd, f"🗑️ {len(keys_to_delete)} validation(s) supprimée(s)."

# ---------------------------------------------------------------------------
# Agent 1 — Récupération des données patient
# ---------------------------------------------------------------------------


def agent_retrieve_patient_data(patient_id: str) -> dict:
    """
    Simule la récupération depuis Orthanc :
      - Rapports cliniques (ODS → simulé depuis SR Orthanc)
      - Résultats de segmentation (findings_validation.json → simulé depuis DICOM-SEG Orthanc)
      - Mesures des nodules (nodules_export.json)
      - Images CT+SEG overlay (export_nodules/)
    """
    # Rapports cliniques (depuis ODS — en prod : Orthanc /studies → SR DICOM)
    df = pd.read_excel(str(REPORTS_ODS), engine="odf")
    patient_reports = df[df["PatientID"] == patient_id].copy()

    # Findings validés
    all_findings: list[dict] = json.loads(FINDINGS_FILE.read_text())
    patient_findings = [e for e in all_findings if e["patient_id"] == patient_id]

    # Mesures nodules (depuis nodules_export.json — en prod : pixel data DICOM-SEG Orthanc)
    all_nodules: list[dict] = json.loads(NODULES_FILE.read_text())
    nodules_by_acc = {
        str(e["accession_number"]): e
        for e in all_nodules
        if e["patient_id"] == patient_id
    }

    # Construire les visites consolidées
    visits: list[dict] = []
    for entry in sorted(patient_findings, key=lambda e: e["accession_number"]):
        acc = str(entry["accession_number"])
        ok = entry.get("ok_findings", [])
        uncertain = entry.get("uncertain_findings", [])
        mismatch = entry.get("image_mismatch_findings", [])
        llm_extracted = entry.get("llm_extracted_findings", [])
        seg_cov = entry.get("seg_coverage") or {}
        ok_nums = {f["finding_number"] for f in ok}

        # Texte du rapport clinique pour cet accession
        report_row = patient_reports[
            patient_reports["AccessionNumber"].astype(str) == acc
        ]
        report_text = (
            report_row["Clinical information data (Pseudo reports)"].iloc[0]
            if not report_row.empty
            else None
        )

        # Mesures depuis nodules_export
        nodule_entry = nodules_by_acc.get(acc, {})
        nodule_measures = {n["number"]: n for n in nodule_entry.get("nodules", [])}
        z_thoracic_min = float(nodule_entry.get("z_thoracic_min") or 0.0)
        z_ct_max = float(nodule_entry.get("z_ct_max") or 0.0)

        # Relations spatiales entre lésions pour cet accession
        findings_coords = {
            num: {"x": m.get("x"), "y": m.get("y"), "z": m.get("z")}
            for num, m in nodule_measures.items()
            if num in ok_nums
        }
        spatial_relations = compute_spatial_relations(findings_coords)

        # Findings validés avec images + mesures
        validated = []
        for f in ok:
            num = f["finding_number"]
            imgs = sorted(
                (IMAGES_DIR / patient_id).glob(f"{acc}_finding{num}_*.png")
            )
            measure = nodule_measures.get(num, {})
            lobe = estimate_lobe(
                measure.get("x"), measure.get("y"), measure.get("z"),
                z_thoracic_min, z_ct_max,
            )
            validated.append({
                "finding_number": num,
                "description": f["description"],
                "image_number": f.get("image_number"),
                "diameter": measure.get("diameter", "N/A"),
                "lobe": lobe,
                "relation_spatiale": spatial_relations.get(num, ""),
                "position": {
                    "x": measure.get("x"),
                    "y": measure.get("y"),
                    "z": measure.get("z"),
                },
                "image_path": str(imgs[0]) if imgs else None,
            })

        # Findings dans le rapport mais sans correspondance SEG
        pulmonary_not_validated = [
            f for f in llm_extracted
            if f.get("is_pulmonary") and f["finding_number"] not in ok_nums
        ]
        report_only = []
        for f in pulmonary_not_validated:
            report_only.append({"finding_number": f["finding_number"], "description": f["description"], "reason": "non apparié dans le SEG"})
        for f in uncertain:
            if f["finding_number"] not in ok_nums:
                reason_map = {"no_seg_file_for_accession": "pas de fichier SEG pour cet accession"}
                report_only.append({"finding_number": f["finding_number"], "description": f["description"], "reason": reason_map.get(f.get("reason", ""), f.get("reason", "inconnu"))})
        for f in mismatch:
            if f["finding_number"] not in ok_nums:
                report_only.append({"finding_number": f["finding_number"], "description": f["description"], "reason": "décalage du numéro d'image"})

        # Segments SEG sans correspondance rapport
        seg_only = [
            {"seg_number": int(k), "image_numbers": v}
            for k, v in seg_cov.items()
            if int(k) not in ok_nums
        ]

        visits.append({
            "accession": acc,
            "date": _get_study_date_from_dicom(patient_id, acc),
            "status": entry.get("status"),
            "report_text": report_text,
            "validated_findings": validated,
            "report_only_findings": report_only,
            "seg_only_segments": seg_only,
        })

    # Identifier la visite la plus récente : son rapport sera généré par nous,
    # donc elle est exclue de la validation SEG.
    def _parse_visit_date(v: dict) -> tuple:
        d = v.get("date", "")
        try:
            dd, mm, yyyy = d.split("/")
            return (int(yyyy), int(mm), int(dd))
        except (ValueError, AttributeError):
            return (0, 0, 0)

    if visits:
        visits.sort(key=_parse_visit_date)  # ordre chronologique croissant
        latest_idx = max(range(len(visits)), key=lambda i: _parse_visit_date(visits[i]))
        for i, v in enumerate(visits):
            v["is_latest"] = (i == latest_idx)

    return {
        "patient_id": patient_id,
        "visits": visits,
        "n_reports": len(patient_reports),
    }


# ---------------------------------------------------------------------------
# Agent 2 — LLM judge : LUNGRAD vs RECIST
# ---------------------------------------------------------------------------


def agent_judge_structure(patient_id: str, visits: list[dict]) -> tuple[str, str]:
    """Demande au LLM de choisir la structure du rapport (LUNGRAD ou RECIST)."""
    context_parts = []
    for v in visits:
        if v.get("report_text"):
            excerpt = str(v["report_text"])[:600]
            context_parts.append(f"Accession {v['accession']} :\n{excerpt}")

    context = "\n\n".join(context_parts) if context_parts else "(pas de rapports disponibles)"

    prompt = f"""You are a radiology reporting expert. Given the following patient context, decide which report structure is most appropriate:

- LUNGRAD: Lung nodule reporting (screening, nodule characterisation, follow-up of lung nodules).
- RECIST: Response evaluation in solid tumors (target lesions, sum of diameters, treatment response).

Patient ID: {patient_id}

Patient history (excerpts):
{context}

Reply with JSON: {{"report_structure": "lungrad"|"recist", "reasoning": "one sentence"}}"""

    resp = _mistral.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        structure = parsed.get("report_structure", "lungrad")
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        structure, reasoning = "lungrad", "Parse failed; defaulting to LUNGRAD."
    return structure, reasoning


# ---------------------------------------------------------------------------
# Agent 3 — LLM synthèse évolution
# ---------------------------------------------------------------------------

SYSTEM_LUNGRAD = "You are a senior radiologist assistant specialised in lung nodule tracking (LUNGRAD framework). Reply in French."
SYSTEM_RECIST = "You are a senior radiologist assistant specialised in oncology response evaluation (RECIST 1.1). Reply in French."


def agent_synthesise_evolution(
    patient_id: str,
    visits: list[dict],
    report_structure: str,
) -> EvolutionReport:
    """
    Construit un EvolutionReport structuré :
    - Lésions, localisation (Python), relations spatiales (Python), mesures + deltas (Python)
    - Interprétations cliniques par mesure (LLM)
    - Autres observations non nodulaires (LLM)
    """
    system = SYSTEM_LUNGRAD if report_structure == "lungrad" else SYSTEM_RECIST

    validated_visits = [v for v in visits if v["validated_findings"]]
    if not validated_visits:
        return EvolutionReport(lesions=[], autres_observations="Aucune lésion validée.")

    # ── Construire les lésions depuis les données Python ─────────────────────
    # On garde la localisation + relation_spatiale du dernier examen disponible
    lesions_by_id: dict[str, Lesion] = {}
    for v in validated_visits:
        date = v.get("date") or v["accession"]
        for f in v["validated_findings"]:
            fid = f"F{f['finding_number']}"
            m = re.search(r"([\d.]+)", f.get("diameter", ""))
            diameter_mm = float(m.group(1)) if m else 0.0
            if fid not in lesions_by_id:
                lesions_by_id[fid] = Lesion(
                    id=fid,
                    localisation=f.get("lobe", ""),
                    relation_spatiale=f.get("relation_spatiale", ""),
                    measurements=[],
                )
            lesions_by_id[fid].measurements.append(
                LesionMeasurement(date=date, diameter_mm=diameter_mm)
            )

    # ── Calcul des deltas en Python ──────────────────────────────────────────
    for lesion in lesions_by_id.values():
        for i, meas in enumerate(lesion.measurements):
            if i > 0:
                meas.delta_mm = round(meas.diameter_mm - lesion.measurements[i - 1].diameter_mm, 1)

    # ── Contexte pour le LLM (interprétations seulement) ────────────────────
    context_lines = []
    for fid, lesion in sorted(lesions_by_id.items()):
        lines = [f"{fid} ({lesion.localisation}):"]
        for meas in lesion.measurements:
            delta_str = (
                "" if meas.delta_mm is None
                else f" ({'+' if meas.delta_mm >= 0 else ''}{meas.delta_mm} mm)"
            )
            lines.append(f"  - {meas.date} : {meas.diameter_mm:.1f} mm{delta_str}")
        context_lines.append("\n".join(lines))
    context = "\n\n".join(context_lines)

    # Extraits de rapports cliniques pour les autres observations
    report_excerpts = "\n".join(
        f"- {v.get('date') or v['accession']} : {str(v['report_text'])[:300]}"
        for v in validated_visits if v.get("report_text")
    )

    prompt = f"""Patient {patient_id} — mesures de segmentation algorithmique (diamètres fiables) :

{context}

Extraits des rapports cliniques (pour identifier les observations non nodulaires uniquement) :
{report_excerpts or "(aucun)"}

Réponds UNIQUEMENT avec ce JSON :
{{
  "interpretations": {{
    "F1": {{"<date>": "<interprétation courte selon {report_structure.upper()}, max 12 mots>"}},
    "F2": {{...}}
  }},
  "autres_observations": "<findings non nodulaires : épanchement, adénopathies, etc. Vide si aucun.>"
}}

Règles :
- N'utilise JAMAIS les diamètres des rapports cliniques — uniquement ceux fournis ci-dessus.
- Interprétation en français, concise, vocabulaire radiologique.
- N'utilise pas d'astérisques."""

    resp = _mistral.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(resp.choices[0].message.content or "{}")
        interps = parsed.get("interpretations", {})
        for fid, lesion in lesions_by_id.items():
            fid_interps = interps.get(fid, {})
            for meas in lesion.measurements:
                meas.interpretation = fid_interps.get(meas.date, "")
        autres = parsed.get("autres_observations", "")
    except Exception:
        autres = ""

    return EvolutionReport(lesions=list(lesions_by_id.values()), autres_observations=autres)


# ---------------------------------------------------------------------------
# Rendu HTML final
# ---------------------------------------------------------------------------


def _b64_img(path: str | None) -> str:
    if not path or not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:420px;border-radius:6px;flex-shrink:0">'


_TH = "padding:9px 14px;border:1px solid #d1d5db;text-align:left;font-weight:600"
_TD = "padding:8px 14px;border:1px solid #e5e7eb;vertical-align:top"


def _build_timeline_chart(patient_data: dict) -> str:
    """Génère un graphe matplotlib d'évolution des lésions; retourne une balise <img> base64 ou ''."""
    import io as _io

    visits = patient_data.get("visits", [])

    # Séries par numéro de lésion : {num: [(date_str, diam_mm), ...]}
    series: dict[int, list[tuple[str, float]]] = {}
    for v in visits:
        date_label = v.get("date") or v["accession"]
        for f in v.get("validated_findings", []):
            num = f["finding_number"]
            m = re.search(r"([\d.]+)", f.get("diameter", ""))
            if not m:
                continue
            series.setdefault(num, []).append((date_label, float(m.group(1))))

    if not series:
        return ""

    # Toutes les dates uniques dans l'ordre chronologique (ordre des visites, déjà trié)
    seen: set[str] = set()
    all_dates: list[str] = []
    for v in visits:
        d = v.get("date") or v["accession"]
        if d not in seen:
            seen.add(d)
            all_dates.append(d)
    date_to_x = {d: i for i, d in enumerate(all_dates)}

    def _fmt(d: str) -> str:
        if len(d) == 8 and d.isdigit():
            return f"{d[6:8]}/{d[4:6]}/{d[:4]}"
        return d

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#f0f9ff")
    ax.set_facecolor("#f0f9ff")

    palette = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]

    for i, (num, points) in enumerate(sorted(series.items())):
        color = palette[i % len(palette)]
        xs = [date_to_x[p[0]] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2.2,
                markersize=8, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.1f} mm", (x, y),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=color, fontweight="bold",
            )
        # Label de la courbe à la dernière valeur
        if points:
            last_x, last_y = date_to_x[points[-1][0]], points[-1][1]
            ax.annotate(
                f"F{num}", (last_x, last_y),
                textcoords="offset points", xytext=(8, 0),
                va="center", fontsize=10, color=color, fontweight="bold",
            )

    ax.set_xticks(range(len(all_dates)))
    ax.set_xticklabels([_fmt(d) for d in all_dates], fontsize=9)
    ax.set_ylabel("Diamètre (mm)", fontsize=10)
    ax.set_title("Évolution temporelle des lésions (diamètre SEG)",
                 fontsize=12, fontweight="bold", color="#1e3a5f", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#93c5fd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(all_dates) > 1:
        ax.set_xlim(-0.3, len(all_dates) - 0.7)

    y_vals = [p[1] for pts in series.values() for p in pts]
    margin = max((max(y_vals) - min(y_vals)) * 0.25, 5)
    ax.set_ylim(max(0, min(y_vals) - margin), max(y_vals) + margin + 8)

    plt.tight_layout()
    buf = _io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;max-width:860px;border-radius:8px;margin-top:4px">'
    )


_TH = "padding:9px 14px;border:1px solid #d1d5db;text-align:left;font-weight:600;background:#eff6ff;color:#1e3a5f"
_TD = "padding:8px 14px;border:1px solid #e5e7eb;vertical-align:top;font-size:13px"
_TD_DELTA_POS = _TD + ";color:#dc2626;font-weight:600"
_TD_DELTA_NEG = _TD + ";color:#16a34a;font-weight:600"
_TD_DELTA_NEU = _TD + ";color:#6b7280"


def build_html_report(
    patient_data: dict,
    structure: str,
    structure_reasoning: str,
    evolution: EvolutionReport,
) -> str:
    patient_id = patient_data["patient_id"]
    visits = list(reversed(patient_data["visits"]))  # plus récent en premier

    # ── Section 1 : synthèse structurée par lésion ───────────────────────
    lesion_blocks = ""
    for lesion in evolution.lesions:
        rows = ""
        for meas in lesion.measurements:
            if meas.delta_mm is None:
                delta_cell = f'<td style="{_TD_DELTA_NEU}">—</td>'
            elif meas.delta_mm > 0:
                delta_cell = f'<td style="{_TD_DELTA_POS}">+{meas.delta_mm} mm</td>'
            elif meas.delta_mm < 0:
                delta_cell = f'<td style="{_TD_DELTA_NEG}">{meas.delta_mm} mm</td>'
            else:
                delta_cell = f'<td style="{_TD_DELTA_NEU}">stable</td>'
            rows += f"""<tr>
              <td style="{_TD}">{meas.date}</td>
              <td style="{_TD};font-weight:600">{meas.diameter_mm:.1f} mm</td>
              {delta_cell}
              <td style="{_TD};color:#374151">{meas.interpretation}</td>
            </tr>"""

        relation_html = (
            f'<p style="margin:4px 0 10px;font-size:12px;color:#6b7280">'
            f'↔ {lesion.relation_spatiale}</p>'
        ) if lesion.relation_spatiale else ""

        lesion_blocks += f"""
        <div style="margin-bottom:20px;border:1px solid #e0e7ff;border-radius:8px;overflow:hidden">
          <div style="background:#eff6ff;padding:10px 16px;display:flex;align-items:baseline;gap:10px">
            <strong style="font-size:15px;color:#1e3a5f">{lesion.id}</strong>
            <span style="font-size:13px;color:#4b5563">{lesion.localisation}</span>
          </div>
          {relation_html and f'<div style="padding:0 16px">{relation_html}</div>'}
          <table style="width:100%;border-collapse:collapse">
            <thead><tr>
              <th style="{_TH}">Date</th>
              <th style="{_TH}">Diamètre SEG</th>
              <th style="{_TH}">Évolution</th>
              <th style="{_TH}">Interprétation</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    if not lesion_blocks:
        lesion_blocks = "<p style='color:#9ca3af'>Aucune lésion validée.</p>"

    autres_html = ""
    if evolution.autres_observations:
        autres_html = f"""
      <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
                  padding:14px 18px;margin-bottom:24px">
        <h3 style="margin:0 0 8px;color:#9a3412;font-size:15px">Autres observations</h3>
        <p style="margin:0;font-size:14px;color:#1e293b">{evolution.autres_observations}</p>
      </div>"""

    # ── Section 2 : images des lésions validées par visite ───────────────
    validated_html_parts: list[str] = []
    for v in visits:
        if not v["validated_findings"]:
            continue
        label = v.get("date") or v["accession"]
        cards = ""
        for f in v["validated_findings"]:
            img_tag = _b64_img(f["image_path"])
            if not img_tag:
                continue
            cards += f"""
            <div style="display:flex;gap:14px;align-items:flex-start;
                        margin:10px 0;padding:12px;background:#f9fafb;
                        border-radius:8px;border:1px solid #e5e7eb">
              {img_tag}
              <div style="flex:1;min-width:0">
                <strong>F{f['finding_number']}</strong>
                <span style="font-size:12px;color:#6b7280;margin-left:8px">{f.get('lobe','')}</span><br>
                <span style="font-size:13px;color:#6b7280">
                  Diamètre SEG : {f['diameter']}
                  &nbsp;·&nbsp; Coupe : {f['image_number']}
                </span>
              </div>
            </div>"""
        if cards:
            validated_html_parts.append(f"""
            <div style="margin-bottom:20px">
              <h4 style="color:#1e40af;margin:0 0 6px;font-size:15px">Examen du {label}</h4>
              {cards}
            </div>""")

    validated_html = "\n".join(validated_html_parts) or ""

    structure_badge = (
        "<span style='background:#dbeafe;color:#1d4ed8;padding:2px 8px;"
        f"border-radius:4px;font-size:12px;font-weight:600'>{structure.upper()}</span>"
    )

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                color:#111827;line-height:1.6">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <h2 style="margin:0;color:#1e3a5f">Patient {patient_id}</h2>
        {structure_badge}
      </div>
      <p style="margin:0 0 20px;font-size:13px;color:#6b7280">
        Structure : {structure_reasoning}
      </p>

      <h3 style="color:#1e3a5f;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-bottom:16px">
        🔬 Évolution par lésion
      </h3>
      {lesion_blocks}

      {autres_html}

      {"<h3 style='color:#1e3a5f;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-bottom:14px'>✅ Images des lésions confirmées</h3>" + validated_html if validated_html else ""}
    </div>"""


# ---------------------------------------------------------------------------
# Génération du rapport de la visite la plus récente
# ---------------------------------------------------------------------------


def _render_new_report_html(patient_id: str, visit_label: str, report_text: str, patient_data: dict | None = None) -> str:
    """Rendu HTML du rapport généré pour la visite la plus récente (style vert)."""
    report_html = (
        report_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n\n", "</p><p style='margin:10px 0'>")
        .replace("\n", "<br>")
    )
    report_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", report_html)
    report_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", report_html)

    timeline_img = _build_timeline_chart(patient_data) if patient_data else ""
    timeline_section = f"""
      <h3 style="color:#1e3a5f;border-bottom:1px solid #e5e7eb;
                 padding-bottom:6px;margin-bottom:14px">
        📈 Évolution temporelle des lésions
      </h3>
      <div>{timeline_img}</div>
    """ if timeline_img else ""

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                color:#111827;line-height:1.6">
      <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                  padding:18px 20px;margin-bottom:24px">
        <h3 style="margin:0 0 12px;color:#15803d;font-size:16px">
          📄 Rapport généré — Patient {patient_id} — Examen du {visit_label}
        </h3>
        <div style="font-size:14px;color:#1e293b">
          <p style='margin:10px 0'>{report_html}</p>
        </div>
      </div>
      {timeline_section}
    </div>"""


def _generate_latest_report(patient_data: dict, report_structure: str) -> tuple:
    """
    Génère le rapport de la visite la plus récente (is_latest=True).
    Returns (gr.update(visible=...), status_md, report_html).
    """
    if not patient_data:
        return gr.update(visible=False), "⚠️ Données patient manquantes.", ""

    visits = patient_data.get("visits", [])
    patient_id = patient_data.get("patient_id", "")

    latest_visit = next((v for v in visits if v.get("is_latest")), None)
    if not latest_visit:
        return gr.update(visible=False), "⚠️ Aucune visite récente identifiée.", ""

    # Décisions utilisateur sur les segments SEG-only
    seg_validations = _load_seg_validations()

    # Mesures SEG depuis nodules_export.json
    all_nodules: list[dict] = json.loads(NODULES_FILE.read_text())
    nodules_by_acc = {
        str(e["accession_number"]): e
        for e in all_nodules
        if e["patient_id"] == patient_id
    }

    # Couverture SEG (image_numbers) depuis findings_validation.json
    all_findings: list[dict] = json.loads(FINDINGS_FILE.read_text())
    seg_coverage_by_acc = {
        str(e["accession_number"]): e.get("seg_coverage") or {}
        for e in all_findings
        if e["patient_id"] == patient_id
    }

    # Construire visits_text dans l'ordre chronologique
    visits_text_parts: list[str] = []
    for v in visits:
        date_str = v.get("date") or v["accession"]
        acc = v["accession"]
        nodule_entry = nodules_by_acc.get(acc, {})
        nodule_measures = {
            n["number"]: n
            for n in nodule_entry.get("nodules", [])
        }
        seg_cov = seg_coverage_by_acc.get(acc, {})

        if v.get("is_latest"):
            lines = [f"Date: {date_str} [EXAMEN LE PLUS RÉCENT — rapport à générer]"]
            if v.get("report_text"):
                lines.append(f'Indication clinique: "{v["report_text"]}"')
            seg_segments = v.get("seg_only_segments", [])
            # Garder uniquement les segments validés comme réels (ou non encore évalués)
            real_segments = [
                s for s in seg_segments
                if seg_validations.get(f"{patient_id}_{acc}_{s['seg_number']}") is not False
            ]
            if real_segments:
                lines.append("Résultats de segmentation algorithmique:")
                for s in real_segments:
                    seg_num = s["seg_number"]
                    image_numbers = s["image_numbers"]
                    measure = nodule_measures.get(seg_num, {})
                    diameter = measure.get("diameter", "N/A")
                    lines.append(
                        f"  - Segment SEG {seg_num}: coupes {image_numbers}, diamètre {diameter}"
                    )
            else:
                lines.append("Résultats de segmentation algorithmique: aucune lésion validée pour cet examen.")
        else:
            lines = [f"Date: {date_str}"]
            if v.get("report_text"):
                lines.append(f'Rapport clinique: "{v["report_text"]}"')
            if v.get("validated_findings"):
                lines.append("Lésions validées (rapport + SEG):")
                for f in v["validated_findings"]:
                    num = f["finding_number"]
                    desc = f["description"]
                    diameter = f["diameter"]
                    fallback = [f["image_number"]] if f.get("image_number") else []
                    image_numbers = seg_cov.get(str(num), fallback)
                    lines.append(
                        f'  - F{num}: "{desc}" (diamètre SEG: {diameter}, coupes: {image_numbers})'
                    )
            # Segments SEG validés comme réels par l'utilisateur (sans rapport)
            seg_validated_real = []
            for s in v.get("seg_only_segments", []):
                seg_num = s["seg_number"]
                key = f"{patient_id}_{acc}_{seg_num}"
                if seg_validations.get(key) is True:
                    measure = nodule_measures.get(seg_num, {})
                    diameter = measure.get("diameter", "N/A")
                    seg_validated_real.append((seg_num, s["image_numbers"], diameter))
            if seg_validated_real:
                lines.append("Lésions SEG validées comme réelles (sans rapport):")
                for seg_num, image_numbers, diameter in seg_validated_real:
                    lines.append(
                        f"  - SEG {seg_num}: coupes {image_numbers}, diamètre {diameter}"
                    )

        visits_text_parts.append("\n".join(lines))

    visits_text = "\n\n".join(visits_text_parts)

    system = SYSTEM_LUNGRAD if report_structure == "lungrad" else SYSTEM_RECIST
    prompt = f"""A patient has done several visits and got scanned by a CT scanner.
For each visit, you are given :
1 - The reason of the exam
2 - The result of the reading by a radiologist

For the most recent visit (marked [EXAMEN LE PLUS RÉCENT — rapport à générer]), there is no radiologist report yet.
Your goal is to write this missing report based on the segmentation algorithm findings.

STRICT INSTRUCTIONS :
- Write the report EXCLUSIVELY for the most recent visit.
- Focus STRICTLY on pulmonary parenchyma findings: lung nodules, parenchymal abnormalities, pleural effusion.
- DO NOT mention and IGNORE any mediastinal, abdominal, hepatic, splenic, renal, adrenal, pelvic, or vascular findings.
- You MUST keep the exact same format as the previous radiology reports (structure, lesion naming F1, F2…).
- You MUST base measurements on the segmentation algorithm results only.
- ONLY use the previous reports to identify existing lesion names (F1, F2, ... Fn) and track their evolution.
- Be factual, concise, and directly usable by a radiologist.
- NEVER use asterisks (*) anywhere in the generated text, not as markers, footnotes, or for any other purpose.

PatientID: {patient_id}

Patient history reports:

{visits_text}"""

    try:
        resp = _mistral.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        report_text = resp.choices[0].message.content or ""
    except Exception as e:
        return gr.update(visible=True), f"❌ Erreur lors de la génération : {e}", ""

    visit_label = latest_visit.get("date") or latest_visit["accession"]
    html = _render_new_report_html(patient_id, visit_label, report_text, patient_data)
    return gr.update(visible=True), "✅ Rapport généré.", html


def _auto_generate_if_no_validation(
    seg_rows: list, patient_data: dict, report_structure: str
):
    """Si pas de segments à valider, affiche le bouton de génération du rapport."""
    # Utiliser le cache en priorité (la propagation du gr.State depuis un générateur
    # vers les .then() suivants est connue pour être peu fiable dans Gradio)
    seg_rows = _pipeline_cache.get("seg_rows", seg_rows or [])
    if seg_rows:
        return gr.update(visible=False), gr.update(visible=False), "", ""
    # Pas de segments — montrer le bouton de génération
    return gr.update(visible=True), gr.update(visible=True), "", ""


def _on_generate_final_report(patient_data: dict, report_structure: str):
    """Génère le rapport de la visite la plus récente (déclenché par le bouton)."""
    if not patient_data:
        patient_data = _pipeline_cache.get("patient_data", {})
    if not report_structure:
        report_structure = _pipeline_cache.get("report_structure", "lungrad")

    yield gr.update(visible=False), "⏳ Génération du rapport en cours…", ""

    try:
        _col_upd, report_status, report_html = _generate_latest_report(patient_data, report_structure)
    except Exception as e:
        yield gr.update(visible=True), f"❌ Erreur : {e}", ""
        return

    yield gr.update(visible=False), report_status, report_html


# ---------------------------------------------------------------------------
# Pipeline agentic complet (générateur → streaming des étapes)
# ---------------------------------------------------------------------------


def run_pipeline(patient_id: str) -> Generator[tuple, None, None]:
    """
    Exécute le pipeline agentic et yield
    (status_markdown, html_output, seg_rows, patient_id, patient_data, report_structure,
     + seg_validation_col, seg_progress_md, seg_image, seg_info_md,
       seg_idx_state, seg_decisions_state, seg_controls_col, seg_section_col)
    à chaque étape.
    """
    patient_id = patient_id.strip()
    # 8 no-op updates for the seg validation outputs in intermediate yields
    _N = (gr.update(),) * 8

    if not patient_id:
        yield "Saisissez un ID patient.", "", [], "", {}, "lungrad", *_N
        return

    status_lines: list[str] = []

    def _step(msg: str, icon: str = "⏳") -> str:
        status_lines.append(f"{icon} {msg}")
        return "\n\n".join(status_lines)

    def _seg_outputs(rows: list, pid: str):
        """Compute the 8 seg-validation output values from seg_rows."""
        existing = _load_seg_validations()
        decisions = {
            f"{r['patient_id']}_{r['accession']}_{r['seg_number']}": existing[
                f"{r['patient_id']}_{r['accession']}_{r['seg_number']}"
            ]
            for r in rows
            if f"{r['patient_id']}_{r['accession']}_{r['seg_number']}" in existing
        }
        if not rows:
            return (
                gr.update(visible=False), "", None, "", 0, decisions,
                gr.update(visible=False), gr.update(visible=False),
            )
        col_upd, progress, img, info, ctrl_upd = _seg_show(rows, 0, decisions)
        return col_upd, progress, img, info, 0, decisions, ctrl_upd, gr.update(visible=True)

    # ── Étape 1 : récupération des données ───────────────────────────────
    yield _step("Agent récupération — lecture des données patient…"), "", [], patient_id, {}, "lungrad", *_N

    try:
        patient_data = agent_retrieve_patient_data(patient_id)
    except Exception as e:
        yield _step(f"Erreur lors de la récupération : {e}", "❌"), "", [], patient_id, {}, "lungrad", *_N
        return

    _pipeline_cache["patient_data"] = patient_data
    visits = patient_data["visits"]
    seg_rows = _build_seg_rows(patient_id, visits)
    _pipeline_cache["seg_rows"] = seg_rows
    n_validated = sum(len(v["validated_findings"]) for v in visits)
    n_seg_only = sum(len(v["seg_only_segments"]) for v in visits)

    yield _step(
        f"Données récupérées — {len(visits)} visite(s), "
        f"{n_validated} lésion(s) validée(s), "
        f"{n_seg_only} segment(s) non rapportés.",
        "✅",
    ), "", [], patient_id, patient_data, "lungrad", *_N

    if n_validated == 0:
        yield (
            _step("Aucune lésion validée — synthèse impossible.", "ℹ️"),
            build_html_report(patient_data, "lungrad", "Aucune visite validée.", EvolutionReport(lesions=[])),
            seg_rows, patient_id, patient_data, "lungrad",
            *_seg_outputs(seg_rows, patient_id),
        )
        return

    # ── Étape 2 : LLM judge ──────────────────────────────────────────────
    yield _step("Agent juge — sélection de la structure (LUNGRAD / RECIST)…"), "", [], patient_id, patient_data, "lungrad", *_N

    try:
        structure, reasoning = agent_judge_structure(patient_id, visits)
    except Exception as e:
        yield _step(f"Erreur du juge LLM : {e}", "❌"), "", [], patient_id, patient_data, "lungrad", *_N
        return

    _pipeline_cache["report_structure"] = structure
    yield _step(
        f"Structure choisie : **{structure.upper()}** — {reasoning}",
        "✅",
    ), "", [], patient_id, patient_data, structure, *_N

    # ── Étape 3 : synthèse évolution ─────────────────────────────────────
    yield _step("Agent synthèse — génération de l'analyse d'évolution…"), "", [], patient_id, patient_data, structure, *_N

    try:
        evolution = agent_synthesise_evolution(patient_id, visits, structure)
    except Exception as e:
        yield _step(f"Erreur de la synthèse LLM : {e}", "❌"), "", [], patient_id, patient_data, structure, *_N
        return

    yield _step("Synthèse générée.", "✅"), "", [], patient_id, patient_data, structure, *_N

    # ── Étape 4 : rendu HTML ─────────────────────────────────────────────
    yield _step("Rendu du rapport…"), "", [], patient_id, patient_data, structure, *_N

    html = build_html_report(patient_data, structure, reasoning, evolution)

    yield (
        _step("Rapport prêt.", "✅"), html, seg_rows, patient_id, patient_data, structure,
        *_seg_outputs(seg_rows, patient_id),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def _patient_ids() -> list[str]:
    findings: list[dict] = json.loads(FINDINGS_FILE.read_text())
    return sorted(set(e["patient_id"] for e in findings))


_CSS = """
/* Supprimer les cadres bleus sur la section validation et ses enfants */
[id^="seg-"], [id^="seg-"] .block, [id^="seg-"] > div,
#new-report, #new-report .block, #new-report > div,
.no-border, .no-border .block, .no-border > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
"""

with gr.Blocks(title="Radiologie — Synthèse IA", css=_CSS) as demo:

    gr.Markdown(
        "# Synthèse radiologique des lésions pulmonaires\n"
        "Sélectionnez un patient. Le pipeline agentic récupère les données, "
        "choisit la structure de rapport (LUNGRAD/RECIST) et génère une synthèse de l'évolution."
    )

    with gr.Row():
        patient_input = gr.Dropdown(
            choices=_patient_ids(),
            allow_custom_value=True,
            label="ID Patient",
            scale=4,
        )
        btn = gr.Button("Générer le rapport →", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            status_md = gr.Markdown(
                value="",
                label="Étapes du pipeline",
                show_label=True,
                elem_id="pipeline-status",
            )
        with gr.Column(scale=3):
            report_html = gr.HTML(
                value="<p style='color:#9ca3af;padding:20px'>Le rapport apparaîtra ici.</p>"
            )

    # ── Hidden states ──────────────────────────────────────────────────
    seg_rows_state = gr.State([])
    seg_idx_state = gr.State(0)
    seg_decisions_state = gr.State({})
    patient_state = gr.State("")
    patient_data_state = gr.State({})
    report_structure_state = gr.State("lungrad")

    # ── Section validation SEG sans rapport ────────────────────────────
    with gr.Column(visible=False, elem_id="seg-section", elem_classes=["no-border"]) as seg_section_col:
      gr.Markdown("---")
      gr.Markdown(
          "### 🔍 Validation des segments SEG sans correspondance dans le rapport\n"
          "_Indiquez pour chaque segment s'il correspond à une vraie lésion (Oui) ou à un faux positif (Non)._"
      )

    with gr.Column(visible=False, elem_id="seg-validation", elem_classes=["no-border"]) as seg_validation_col:
        seg_progress_md = gr.Markdown(elem_id="seg-progress")
        with gr.Column(visible=True) as seg_controls_col:
            with gr.Row():
                with gr.Column(scale=2):
                    seg_image = gr.Image(
                        label="Coupe CT + overlay SEG",
                        show_label=True,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    seg_info_md = gr.Markdown()
                    with gr.Row():
                        oui_btn = gr.Button("✅ Oui — lésion réelle", variant="primary")
                        non_btn = gr.Button("❌ Non — faux positif", variant="stop")
        seg_loading_md = gr.Markdown("", elem_id="seg-loading")
        with gr.Row():
            save_seg_btn = gr.Button("💾 Sauvegarder les validations", variant="secondary")
            reset_seg_btn = gr.Button("🗑️ Réinitialiser", variant="secondary")
            seg_save_status = gr.Markdown("")

    with gr.Column(visible=False, elem_id="new-report") as new_report_col:
        gr.Markdown("---")
        gr.Markdown("### 📄 Rapport généré pour la visite la plus récente")
        gen_final_report_btn = gr.Button("Générer le rapport final →", variant="primary", visible=False)
        new_report_status_md = gr.Markdown("")
        new_report_html_out = gr.HTML("")

    # ── Wiring ─────────────────────────────────────────────────────────
    _SEG_OUTPUTS = [seg_validation_col, seg_progress_md, seg_image, seg_info_md, seg_idx_state, seg_decisions_state, seg_controls_col]

    reset_event = btn.click(
        fn=lambda: ("", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", ""),
        outputs=[status_md, report_html, seg_section_col, seg_validation_col, new_report_col, gen_final_report_btn, new_report_status_md, new_report_html_out],
        show_progress="hidden",
    )

    pipeline_event = reset_event.then(
        fn=run_pipeline,
        inputs=patient_input,
        outputs=[
            status_md, report_html, seg_rows_state, patient_state, patient_data_state, report_structure_state,
            seg_validation_col, seg_progress_md, seg_image, seg_info_md, seg_idx_state, seg_decisions_state,
            seg_controls_col, seg_section_col,
        ],
        show_progress="hidden",
    )

    pipeline_event.then(
        fn=_auto_generate_if_no_validation,
        inputs=[seg_rows_state, patient_data_state, report_structure_state],
        outputs=[new_report_col, gen_final_report_btn, new_report_status_md, new_report_html_out],
        show_progress="hidden",
    )

    gen_final_report_btn.click(
        fn=_on_generate_final_report,
        inputs=[patient_data_state, report_structure_state],
        outputs=[gen_final_report_btn, new_report_status_md, new_report_html_out],
        show_progress="hidden",
    )

    oui_btn.click(
        fn=_on_oui,
        inputs=[seg_rows_state, seg_idx_state, seg_decisions_state],
        outputs=_SEG_OUTPUTS,
    )

    non_btn.click(
        fn=_on_non,
        inputs=[seg_rows_state, seg_idx_state, seg_decisions_state],
        outputs=_SEG_OUTPUTS,
    )

    save_seg_btn.click(
        fn=_save_and_generate,
        inputs=[seg_rows_state, seg_decisions_state, patient_data_state, report_structure_state],
        outputs=[seg_save_status, seg_loading_md, seg_section_col, seg_validation_col, seg_controls_col, new_report_col, gen_final_report_btn, new_report_status_md, new_report_html_out],
        show_progress="hidden",
    )

    reset_seg_btn.click(
        fn=_reset_seg,
        inputs=[seg_rows_state, patient_state],
        outputs=[*_SEG_OUTPUTS, seg_save_status],
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Soft(),
    )
