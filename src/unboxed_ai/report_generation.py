"""
Report generation from Orthanc + Excel: patient lung history summaries.

Optional LLM judge: chooses between LUNGRAD vs RECIST report structure from context,
then routes to the corresponding prompt.

Limits of the judge + routing approach:
- Latency and cost: two LLM calls (judge + report) per patient.
- Judge reliability: the model can pick the wrong structure or vary on similar inputs.
- Binary choice: hybrid cases (e.g. screening then treatment) may need "both" or
  a third option; consider extending ReportStructureKind if needed.
- Audit: log report_structure and reasoning (e.g. in output_file or a sidecar) for traceability.
"""

import os
from typing import Literal

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from mistralai import Mistral
from unboxed_ai.lib.Constants import Constants

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

ORTHANC = "https://orthanc.unboxed-2026.ovh"
AUTH = ("unboxed", "unboxed2026")
def fetch_studies_from_orthanc() :
    studies_ids = requests.get(f"{ORTHANC}/studies", auth=AUTH, timeout=5).json()
    print(f"  📊 {len(studies_ids)} étude(s) DICOM dans Orthanc\n")

    if not studies_ids:
        print("  ℹ️  Le dataset sera chargé avant le hackathon.")
    else:
        rows = []
        print(studies_ids)
        print(len(studies_ids))
        for sid in studies_ids:
            info = requests.get(f"{ORTHANC}/studies/{sid}", auth=AUTH).json()
            t = info.get("MainDicomTags", {})
            rows.append({
                "ID"          : sid[:12] + "…",
                "PatientID"     : t.get("PatientID", "-"),
                "StudyDescription" : t.get("StudyDescription", "-"),
                "StudyDate"        : t.get("StudyDate", "-"),
                "ModalitiesInStudy"    : t.get("ModalitiesInStudy", "-"),
                "AccessionNumber" : t.get("AccessionNumber", "-"),
            })
            print(sid, t.get("PatientID", "-"))
        df = pd.DataFrame(rows)
        return(df)
    
# ---------------------------------------------------------------------------
# LLM judge: LUNGRAD vs RECIST report structure (structured output)
# ---------------------------------------------------------------------------

ReportStructureKind = Literal["lungrad", "recist"]


class ReportStructureChoice(BaseModel):
    """Structured output from the judge LLM."""

    report_structure: ReportStructureKind
    reasoning: str = ""


def judge_report_structure(patient_id: str, visits_text: str) -> ReportStructureChoice:
    """
    Use the LLM as a judge to decide whether LUNGRAD or RECIST is the most
    appropriate report structure for this patient's context. Returns structured
    output (report_structure + optional reasoning).
    """
    judge_prompt = f"""You are a radiology reporting expert. Given the following patient context, decide which report structure is most appropriate:

- **LUNGRAD**: Lung nodule reporting (screening, nodule characterization, follow-up of lung nodules). Use when the main focus is lung nodules / screening.
- **RECIST**: Response evaluation in solid tumors (target lesions, sum of diameters, treatment response). Use when the main focus is treatment response or measurable target lesions.

PatientID: {patient_id}

Patient history / clinical context:

{visits_text}

Return your decision as JSON with:
- "report_structure": either "lungrad" or "recist"
- "reasoning": brief explanation (one sentence)."""

    response = client.chat.parse(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format=ReportStructureChoice,
        temperature=0.1,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        # Fallback if parse failed: default to lungrad
        return ReportStructureChoice(report_structure="lungrad", reasoning="Parse failed; defaulting to LUNGRAD.")
    return parsed


# ---------------------------------------------------------------------------
# Report generation (routed by judge)
# ---------------------------------------------------------------------------

LUNGRAD_SYSTEM = """You are a senior radiologist assistant.
Summarize the patient's lung history in chronological order using a LUNGRAD-oriented structure.
Focus on: lung nodules, size and character, screening context, and nodule follow-up.
Use clear language; reference relevant images per visit. Highlight nodule evolution and any LUNGRAD-relevant findings."""

RECIST_SYSTEM = """You are a senior radiologist assistant.
Summarize the patient's lung history in chronological order using a RECIST-oriented structure.
Focus on: target and non-target lesions, sum of diameters, treatment response (CR/PR/SD/PD), and measurable disease.
Use clear language; reference relevant images per visit. Highlight response trends and RECIST-relevant measurements."""


def build_report_prompt(
    patient_id: str,
    visits_text: str,
    report_structure: ReportStructureKind,
) -> str:
    """Build the main report prompt for the chosen structure (LUNGRAD or RECIST)."""
    system = LUNGRAD_SYSTEM if report_structure == "lungrad" else RECIST_SYSTEM
    return f"""{system}

PatientID: {patient_id}

Patient history reports:

{visits_text}

Return a structured summary in clear language. Include:
- Changes over time
- Key observations relevant to {"lung nodules and LUNGRAD" if report_structure == "lungrad" else "target lesions and RECIST response"}
- Referenced images per visit
- Diagnosis/response trends as appropriate."""


def generate_response(prompt: str) -> str:
    """
    Generates a response from the Mistral API based on the given prompt.
    """
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def excel_to_df(file_path: str) -> pd.DataFrame:
    """
    Opens an Excel file and returns its contents as a pandas DataFrame.
    (Name kept for compatibility — it now returns the DataFrame.)
    """
    df = pd.read_excel(file_path)
    return df


def filter_by_patient(df: pd.DataFrame, patient_id) -> pd.DataFrame:
    """Return rows where the `PatientID` column equals `patient_id`."""
    if 'PatientID' not in df.columns:
        raise KeyError("No 'PatientID' column in dataframe")
    return df[df['PatientID'] == patient_id]


def iterate_accession_numbers(df: pd.DataFrame, patient_id=None):
    """Yield unique `AccessionNumber` values; if `patient_id` provided, filter first."""
    if patient_id is not None:
        df = filter_by_patient(df, patient_id)
    if 'AccessionNumber' not in df.columns:
        raise KeyError("No 'AccessionNumber' column in dataframe")
    for acc in df['AccessionNumber'].dropna().unique():
        yield acc

def merge_on_accession(excel_df: pd.DataFrame, orthanc_df: pd.DataFrame, excel_col: str = 'AccessionNumber', orthanc_col: str = 'AccessionNumber', how: str = 'inner') -> pd.DataFrame:
    """Merge the Excel and Orthanc DataFrames on accession number.

    Returns the merged DataFrame (columns suffixed with `_excel` and `_orthanc` when needed).
    """
    if excel_col not in excel_df.columns:
        raise KeyError(f"Excel dataframe missing column: {excel_col}")
    if orthanc_col not in orthanc_df.columns:
        raise KeyError(f"Orthanc dataframe missing column: {orthanc_col}")

    # Work on copies to avoid mutating caller data
    left = excel_df.copy()
    right = orthanc_df.copy()

    # Coerce both merge keys to string dtype and normalize whitespace.
    # This avoids errors when one side is int/float and the other is str.
    left[excel_col] = left[excel_col].astype("string").str.strip()
    right[orthanc_col] = right[orthanc_col].astype("string").str.strip()

    # Drop rows with missing or empty accession values (can't match)
    left = left[left[excel_col].notna() & (left[excel_col] != "")]
    right = right[right[orthanc_col].notna() & (right[orthanc_col] != "")]

    merged = left.merge(right, left_on=excel_col, right_on=orthanc_col, how=how, suffixes=("_excel", "_orthanc"))
    return merged


def generate_report_on_lungs_only(
    df: pd.DataFrame,
    patient_id: str,
    output_file: str = "history_summary.txt",
    use_judge: bool = True,
) -> str:
    """
    Generates a lung-history report for the patient. If use_judge is True (default),
    an LLM judge first chooses LUNGRAD vs RECIST report structure from context,
    then the main report is generated with the corresponding prompt.
    """
    patient_visits = df[df["PatientID_excel"] == patient_id].sort_values(by="StudyDate")
    visits_text = ""
    for _, row in patient_visits.iterrows():
        visits_text += f"Date: {row['StudyDate']} (format : YYYYMMDD)\n"
        visits_text += f"Clinical information: {row['Clinical information data (Pseudo reports)']}\n"

    if use_judge:
        choice = judge_report_structure(patient_id, visits_text)
        print(f"  Judge: {choice.report_structure} — {choice.reasoning}")
        prompt = build_report_prompt(patient_id, visits_text, choice.report_structure)
    else:
        # Legacy single prompt (no routing)
        prompt = build_report_prompt(patient_id, visits_text, "lungrad")

    print(prompt)
    response = generate_response(prompt)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Response saved to {output_file}")
    return response

if __name__ == "__main__":
    df = excel_to_df(Constants.REPORTS_PATH)
    orth_df = fetch_studies_from_orthanc()
    merged = merge_on_accession(df, orth_df)

    def chat_fn(message, history):
        rapport = generate_report_on_lungs_only(merged, message, output_file="history_report.txt")
        return rapport

    import gradio as gr
    demo = gr.ChatInterface(fn=chat_fn, title="Report on patient history", fill_height=True, save_history=True)
    demo.launch()