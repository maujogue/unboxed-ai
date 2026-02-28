import requests
import pandas as pd

ORTHANC = "https://orthanc.unboxed-2026.ovh"
AUTH    = ("unboxed", "unboxed2026")

studies_ids = requests.get(f"{ORTHANC}/studies", auth=AUTH, timeout=5).json()
print(f"  📊 {len(studies_ids)} étude(s) DICOM dans Orthanc\n")

if not studies_ids:
    print("  ℹ️  Le dataset sera chargé avant le hackathon.")
else:
    rows = []
    for sid in studies_ids[:10]:
        info = requests.get(f"{ORTHANC}/studies/{sid}", auth=AUTH).json()
        t = info.get("MainDicomTags", {})
        rows.append({
            "ID"          : sid[:12] + "…",
            "Patient"     : t.get("PatientID", "-"),
            "Description" : t.get("StudyDescription", "-"),
            "Date"        : t.get("StudyDate", "-"),
            "Modalité"    : t.get("ModalitiesInStudy", "-"),
        })
    df = pd.DataFrame(rows)
    print(df)