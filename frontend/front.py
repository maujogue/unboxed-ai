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

# Cache module-level pour contourner les problèmes de propagation des gr.State
# depuis un générateur vers d'autres boutons (comportement connu de Gradio).
_pipeline_cache: dict = {"patient_data": {}, "report_structure": "lungrad"}


# ---------------------------------------------------------------------------
# Helper — résolution de la date d'examen depuis l'arborescence locale
# ---------------------------------------------------------------------------


def _get_study_date(patient_id: str, accession: str) -> str:
    """
    Retrouve la date d'examen (DD/MM/YYYY) à partir du nom du dossier d'étude.
    Structure : dataset/{patient}*/{YYYYMMDD_...}/{patient}/  {accession ...}/
    Retourne la chaîne de l'accession en fallback si introuvable.
    """
    for patient_dir in DATASET_DIR.glob(f"*{patient_id}*"):
        for acc_dir in patient_dir.rglob(f"{accession}*"):
            if not acc_dir.is_dir():
                continue
            # date_folder est 2 niveaux au-dessus du dossier accession
            date_folder = acc_dir.parent.parent.name
            if len(date_folder) >= 8 and date_folder[:8].isdigit():
                d = date_folder[:8]  # YYYYMMDD
                return f"{d[6:8]}/{d[4:6]}/{d[0:4]}"
    return accession  # fallback


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
        nodule_measures = {
            n["number"]: n
            for n in nodules_by_acc.get(acc, {}).get("nodules", [])
        }

        # Findings validés avec images + mesures
        validated = []
        for f in ok:
            num = f["finding_number"]
            imgs = sorted(
                (IMAGES_DIR / patient_id).glob(f"{acc}_finding{num}_*.png")
            )
            measure = nodule_measures.get(num, {})
            validated.append({
                "finding_number": num,
                "description": f["description"],
                "image_number": f.get("image_number"),
                "diameter": measure.get("diameter", "N/A"),
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
            "date": _get_study_date(patient_id, acc),
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
) -> str:
    """
    Génère une synthèse de l'évolution des lésions validées (rapport ✓ + SEG ✓).
    Seules les visites avec des findings validés sont incluses dans le contexte.
    """
    system = SYSTEM_LUNGRAD if report_structure == "lungrad" else SYSTEM_RECIST

    validated_visits = [v for v in visits if v["validated_findings"]]
    if not validated_visits:
        return "_Aucune lésion validée pour ce patient._"

    context_lines: list[str] = []
    for v in validated_visits:
        label = v.get("date") or v["accession"]
        findings_desc = "\n".join(
            f"  • F{f['finding_number']} — {f['description']} "
            f"(diamètre SEG : {f['diameter']}, image {f['image_number']})"
            for f in v["validated_findings"]
        )
        # Inclure un extrait du rapport clinique pour contexte
        report_excerpt = ""
        if v.get("report_text"):
            report_excerpt = f"\n  Extrait rapport : {str(v['report_text'])[:400]}…"

        context_lines.append(
            f"Examen du {label} :\n{findings_desc}{report_excerpt}"
        )

    context = "\n\n".join(context_lines)

    prompt = f"""Voici les lésions pulmonaires confirmées (présentes dans le rapport ET dans la segmentation algorithmique) pour le patient {patient_id}, dans l'ordre chronologique :

{context}

Rédige une synthèse médicale structurée de l'évolution de ces lésions :
- Pour chaque lésion récurrente (F1, F2…), décris l'évolution entre les visites (diamètre, stabilité, progression, régression).
- Signale l'apparition de nouvelles lésions ou la disparition d'anciennes.
- Utilise le vocabulaire radiologique approprié en français.
- Sois factuel, concis et directement utilisable par un radiologue.
- Structure ta réponse en paragraphes clairs (une lésion = un paragraphe).
- NE génère PAS d'en-tête de rapport ni de conclusion générale : uniquement l'analyse de l'évolution.
- IMPÉRATIF : pour les diamètres, utilise UNIQUEMENT le diamètre issu de la segmentation algorithmique (champ "diamètre SEG"). N'utilise JAMAIS les diamètres mentionnés dans les extraits de rapport clinique. Ne compare pas les deux valeurs et ne les cite pas ensemble.
- N'utilise JAMAIS d'astérisques (*) dans le texte généré, ni comme marqueur, ni comme note de bas de page, ni pour tout autre usage."""

    resp = _mistral.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


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


def build_html_report(
    patient_data: dict,
    structure: str,
    structure_reasoning: str,
    synthesis: str,
) -> str:
    patient_id = patient_data["patient_id"]
    visits = list(reversed(patient_data["visits"]))  # plus récent en premier

    # ── Section 1 : lésions validées par visite ──────────────────────────
    validated_html_parts: list[str] = []
    for v in visits:
        if not v["validated_findings"]:
            continue
        label = v.get("date") or v["accession"]
        cards = ""
        for f in v["validated_findings"]:
            img_tag = _b64_img(f["image_path"])
            cards += f"""
            <div style="display:flex;gap:14px;align-items:flex-start;
                        margin:10px 0;padding:12px;background:#f9fafb;
                        border-radius:8px;border:1px solid #e5e7eb">
              {img_tag}
              <div style="flex:1;min-width:0">
                <strong>F{f['finding_number']}</strong><br>
                <span style="font-size:13px;color:#6b7280">
                  Diamètre SEG : {f['diameter']}
                  &nbsp;·&nbsp; Coupe : {f['image_number']}
                </span>
              </div>
            </div>"""

        validated_html_parts.append(f"""
        <div style="margin-bottom:20px">
          <h4 style="color:#1e40af;margin:0 0 6px;font-size:15px">
            Examen du {label}
          </h4>
          {cards}
        </div>""")

    if validated_html_parts:
        validated_html = "\n".join(validated_html_parts)
    else:
        validated_html = "<p style='color:#9ca3af'>Aucune lésion validée.</p>"

    # ── Synthèse LLM : markdown → HTML basique ───────────────────────────
    synthesis_html = (
        synthesis
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n\n", "</p><p style='margin:10px 0'>")
        .replace("\n", "<br>")
        .replace("**", "<strong>", 1)
    )
    # Gestion basique du bold **...**
    synthesis_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", synthesis_html)
    synthesis_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", synthesis_html)

    structure_badge = (
        "<span style='background:#dbeafe;color:#1d4ed8;padding:2px 8px;"
        f"border-radius:4px;font-size:12px;font-weight:600'>{structure.upper()}</span>"
    )

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                color:#111827;line-height:1.6">

      <!-- En-tête -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <h2 style="margin:0;color:#1e3a5f">Patient {patient_id}</h2>
        {structure_badge}
      </div>
      <p style="margin:0 0 16px;font-size:13px;color:#6b7280">
        Structure choisie par le LLM judge : {structure_reasoning}
      </p>

      <!-- Synthèse LLM -->
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                  padding:18px 20px;margin-bottom:24px">
        <h3 style="margin:0 0 12px;color:#1d4ed8;font-size:16px">
          🤖 Synthèse de l'évolution — analyse IA
        </h3>
        <div style="font-size:14px;color:#1e293b">
          <p style='margin:10px 0'>{synthesis_html}</p>
        </div>
      </div>

      <!-- Lésions validées avec images -->
      <h3 style="color:#1e3a5f;border-bottom:1px solid #e5e7eb;
                 padding-bottom:6px;margin-bottom:14px">
        ✅ Lésions confirmées
      </h3>
      {validated_html}


    </div>"""


# ---------------------------------------------------------------------------
# Génération du rapport de la visite la plus récente
# ---------------------------------------------------------------------------


def _render_new_report_html(patient_id: str, visit_label: str, report_text: str) -> str:
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
    html = _render_new_report_html(patient_id, visit_label, report_text)
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
            build_html_report(patient_data, "lungrad", "Aucune visite validée.", ""),
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
        synthesis = agent_synthesise_evolution(patient_id, visits, structure)
    except Exception as e:
        yield _step(f"Erreur de la synthèse LLM : {e}", "❌"), "", [], patient_id, patient_data, structure, *_N
        return

    yield _step("Synthèse générée.", "✅"), "", [], patient_id, patient_data, structure, *_N

    # ── Étape 4 : rendu HTML ─────────────────────────────────────────────
    yield _step("Rendu du rapport…"), "", [], patient_id, patient_data, structure, *_N

    html = build_html_report(patient_data, structure, reasoning, synthesis)

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
