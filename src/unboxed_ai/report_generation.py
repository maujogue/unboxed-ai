import pandas as pd
import json
import requests
from dotenv import load_dotenv
import os
import requests, pandas as pd

ORTHANC = "https://orthanc.unboxed-2026.ovh"
AUTH    = ("unboxed", "unboxed2026")
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
    
load_dotenv()
API_KEY=os.getenv("MISTRAL_API_KEY")
from mistralai import Mistral

client = Mistral(api_key=API_KEY)

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

import pandas as pd
import json
    
def generate_report_on_lungs_only(df: pd.DataFrame, patient_id: str, output_file: str = "history_summary.txt"):
    """
    Génère un prompt prêt à envoyer à l'IA pour synthétiser l'historique pulmonaire
    d'un patient à partir du DataFrame enrichi.
    """
    # Filtrer les visites du patient
    patient_visits = df[df["PatientID_excel"] == patient_id].sort_values(by="StudyDate")

    # Construire le texte à inclure dans le prompt
    visits_text = ""
    for _, row in patient_visits.iterrows():
        visits_text += f"Date: {row['StudyDate']} (format : YYYYMMDD)\n"
        visits_text += f"Clinical information: {row['Clinical information data (Pseudo reports)']}\n"

    prompt = f"""
You are a senior radiologist assistant.

Summarize the following patient's lung history in chronological order.
Focus ONLY on lung findings, their evolution, and reference any relevant images.

PatientID: {patient_id}

Patient history reports:

{visits_text}

Return a structured summary in clear language, highlighting:
- Changes over time
- Important lung observations
- Referenced images per visit
- Diagnosis trends (RECIST vs LUNGRAD if available)
"""
    print(prompt)
    response = generate_response(prompt)
    # Sauvegarder le prompt si demandé
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Response sauvegardée dans {output_file}")

    return response

if __name__ == "__main__":
    file_path = r"C:\Users\carl-\Downloads\Liste examen UNBOXED finaliseģe v2 (avec mesures).xlsx"  # Replace with your Excel file path
    df = excel_to_df(file_path)
    orth_df = fetch_studies_from_orthanc()
    merged = merge_on_accession(df, orth_df)

    def chat_fn(message, history):
        rapport = generate_report_on_lungs_only(merged, message, output_file="history_report.txt")
        return rapport

    import gradio as gr
    demo = gr.ChatInterface(fn=chat_fn, title="Report on patient history", fill_height=True, save_history=True)
    demo.launch()