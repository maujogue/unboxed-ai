#!/usr/bin/env python3
"""
generate_findings_report.py

Generates an HTML report from findings_validation.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

INPUT = Path("findings_validation.json")
OUTPUT = Path("results/findings_validation.html")

STATUS_LABEL = {
    "validated": ("OK", "#1e8449", "#d5f5e3"),
    "uncertain": ("UNCERTAIN", "#922b21", "#fadbd8"),
    "image_mismatch": ("IMAGE MISMATCH", "#784212", "#fdebd0"),
    "to_check": ("À VÉRIFIER", "#1a5276", "#d6eaf8"),
    "no_pulmonary_findings": ("AUCUNE LÉSION PULMONAIRE", "#6d6d6d", "#f2f3f4"),
}

FLAG_BADGE = {
    "OK": ('<span style="background:#1e8449;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em">✓ OK</span>'),
    "UNCERTAIN": ('<span style="background:#922b21;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em">⚠ UNCERTAIN</span>'),
    "IMAGE_MISMATCH": ('<span style="background:#784212;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em">⚡ IMAGE MISMATCH</span>'),
    "TO_CHECK": ('<span style="background:#1a5276;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em">? À VÉRIFIER</span>'),
}


def seg_coverage_html(coverage: dict | None) -> str:
    if coverage is None:
        return '<span style="color:#922b21"><em>Aucun fichier SEG pour cet examen</em></span>'
    if not coverage:
        return "<em>Aucun segment actif</em>"
    rows = []
    for fn, imgs in sorted(coverage.items()):
        imgs_str = ", ".join(str(i) for i in imgs) if imgs else "—"
        rows.append(f"<tr><td>Finding.{fn}</td><td>{imgs_str}</td></tr>")
    return (
        '<table style="width:auto;margin:4px 0">'
        "<tr><th>Segment</th><th>Images couvertes</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def findings_table_html(findings: list[dict]) -> str:
    if not findings:
        return ""
    rows = []
    for f in findings:
        fn = f"F{f['finding_number']}" if f["finding_number"] is not None else "(sans tag)"
        img = f"Image&nbsp;{f['image_number']}" if f["image_number"] else "—"
        flag = FLAG_BADGE.get(f.get("flag", ""), "")
        reason = f.get("reason", "")
        reason_html = f'<br><small style="color:#777">{reason}</small>' if reason else ""
        rows.append(
            f"<tr><td>{flag}</td><td><strong>{fn}</strong></td>"
            f"<td>{img}</td><td>{f['description']}{reason_html}</td></tr>"
        )
    return (
        '<table style="margin:8px 0">'
        "<tr><th>Flag</th><th>Finding</th><th>Image</th><th>Description</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def render_entry(entry: dict) -> str:
    pid = entry["patient_id"]
    acc = entry["accession_number"]
    status = entry["status"]
    label, color, bg = STATUS_LABEL.get(status, (status, "#333", "#fff"))

    badge = (
        f'<span style="background:{color};color:#fff;padding:3px 12px;'
        f'border-radius:12px;font-size:0.85em;font-weight:bold">{label}</span>'
    )

    all_findings = (
        entry.get("ok_findings", [])
        + entry.get("uncertain_findings", [])
        + entry.get("image_mismatch_findings", [])
        + entry.get("to_check_findings", [])
    )

    html = f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:16px 20px;margin:16px 0;background:{bg}">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
    <h2 style="margin:0;color:#1a5276">Patient&nbsp;<code>{pid}</code></h2>
    <span style="color:#555">AccNum&nbsp;<code>{acc}</code></span>
    {badge}
  </div>
  <div style="margin-bottom:8px">
    <strong>Couverture SEG&nbsp;(algo)&nbsp;:</strong><br>
    {seg_coverage_html(entry.get("seg_coverage"))}
  </div>
  <div>
    <strong>Lésions extraites du rapport&nbsp;:</strong><br>
    {findings_table_html(all_findings) if all_findings else "<em style='color:#777'>Aucune lésion pulmonaire identifiée dans le rapport.</em>"}
  </div>
</div>
"""
    return html


def main() -> None:
    data = json.loads(INPUT.read_text())

    counts = {
        "uncertain": sum(1 for r in data if r["uncertain_findings"]),
        "image_mismatch": sum(1 for r in data if r["image_mismatch_findings"] and not r["uncertain_findings"]),
        "to_check": sum(1 for r in data if r["to_check_findings"] and not r["uncertain_findings"] and not r["image_mismatch_findings"]),
        "ok": sum(1 for r in data if r["status"] == "validated"),
        "no_findings": sum(1 for r in data if r["status"] == "no_pulmonary_findings"),
    }

    summary_html = f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;margin:16px 0">
  <div style="background:#d5f5e3;border-radius:8px;padding:12px 20px;text-align:center">
    <div style="font-size:2em;font-weight:bold;color:#1e8449">{counts['ok']}</div>
    <div style="color:#555;font-size:0.9em">OK</div>
  </div>
  <div style="background:#fadbd8;border-radius:8px;padding:12px 20px;text-align:center">
    <div style="font-size:2em;font-weight:bold;color:#922b21">{counts['uncertain']}</div>
    <div style="color:#555;font-size:0.9em">UNCERTAIN</div>
  </div>
  <div style="background:#fdebd0;border-radius:8px;padding:12px 20px;text-align:center">
    <div style="font-size:2em;font-weight:bold;color:#784212">{counts['image_mismatch']}</div>
    <div style="color:#555;font-size:0.9em">IMAGE MISMATCH</div>
  </div>
  <div style="background:#d6eaf8;border-radius:8px;padding:12px 20px;text-align:center">
    <div style="font-size:2em;font-weight:bold;color:#1a5276">{counts['to_check']}</div>
    <div style="color:#555;font-size:0.9em">À VÉRIFIER</div>
  </div>
  <div style="background:#f2f3f4;border-radius:8px;padding:12px 20px;text-align:center">
    <div style="font-size:2em;font-weight:bold;color:#6d6d6d">{counts['no_findings']}</div>
    <div style="color:#555;font-size:0.9em">RAPPORT VIDE</div>
  </div>
</div>
"""

    entries_html = "\n".join(render_entry(e) for e in data)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Validation des findings — rapports vs algo</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ margin: 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
    th {{ background: #f0f4f8; font-weight: bold; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  </style>
</head>
<body>
<h1>Validation findings — rapports cliniques vs algorithme de segmentation</h1>
<strong>Généré le :</strong> {now} &nbsp;|&nbsp;
<strong>Examens analysés :</strong> {len(data)}<br><br>

<h2 style="border-bottom:1px solid #ddd;padding-bottom:6px">Résumé</h2>
{summary_html}

<h2 style="border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:24px">Détail par examen</h2>
{entries_html}

<hr>
<p style="color:#999;font-size:0.8em">
  <strong>Légende :</strong>
  ✓ OK = finding (FX) présent dans le rapport ET dans le SEG avec image couverte ·
  ⚠ UNCERTAIN = finding (FX) dans le rapport mais absent du SEG (ou aucun SEG pour cet examen) ·
  ⚡ IMAGE MISMATCH = segment SEG existe mais ne couvre pas l'image citée dans le rapport ·
  ? À VÉRIFIER = lésion pulmonaire mentionnée sans tag (FX), pas de cross-référence possible
</p>
</body>
</html>
"""

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Report saved → {OUTPUT}")


if __name__ == "__main__":
    main()
