import math
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()

# ---- Config ----
ORTHANC_URL = os.getenv("ORTHANC_URL", "https://orthanc.unboxed-2026.ovh")
ORTHANC_USER = os.getenv("ORTHANC_USERNAME", "")
ORTHANC_PASS = os.getenv("ORTHANC_PASSWORD", "")
AUTH = (ORTHANC_USER, ORTHANC_PASS) if ORTHANC_USER else None

# ---- 1. Lister les patients et prendre le premier ----
patients = requests.get(f"{ORTHANC_URL}/patients", auth=AUTH).json()
patient_id = patients[0]
print(f"Patient : {patient_id}")

studies = requests.get(f"{ORTHANC_URL}/patients/{patient_id}/studies", auth=AUTH).json()
series_list = studies[0]["Series"]

# ---- 2. Lister toutes les séries CT et SEG ----
ct_series_ids = []
seg_series_id = None

for sid in series_list:
    info = requests.get(f"{ORTHANC_URL}/series/{sid}", auth=AUTH).json()
    modality = info.get("MainDicomTags", {}).get("Modality", "")
    print(f"Série {sid} → {modality}")
    if modality == "CT":
        ct_series_ids.append(sid)
    if modality == "SEG" and seg_series_id is None:
        seg_series_id = sid

print(f"\nSEG : {seg_series_id}")

# ---- 3. Télécharger le SEG ----
seg_instances = requests.get(f"{ORTHANC_URL}/series/{seg_series_id}/instances", auth=AUTH).json()
seg_instance_id = seg_instances[0]["ID"]
print(f"Instance SEG : {seg_instance_id}")

r_seg = requests.get(f"{ORTHANC_URL}/instances/{seg_instance_id}/file", auth=AUTH)
seg_dcm = pydicom.dcmread(BytesIO(r_seg.content))

seg_array = seg_dcm.pixel_array.astype(np.float32)
if seg_array.ndim == 2:
    seg_array = seg_array[np.newaxis, ...]  # (1, H, W)

n_frames = len(seg_dcm.PerFrameFunctionalGroupsSequence)
print(f"Nombre de frames SEG : {n_frames}")

# ---- 4. Construire un index SOPInstanceUID → ID Orthanc pour toutes les séries CT ----
print("\nConstruction de l'index CT...")
uid_to_orthanc_id = {}
for sid in ct_series_ids:
    instances = requests.get(f"{ORTHANC_URL}/series/{sid}/instances", auth=AUTH).json()
    for inst in instances:
        info = requests.get(f"{ORTHANC_URL}/instances/{inst['ID']}", auth=AUTH).json()
        sop_uid = info["MainDicomTags"]["SOPInstanceUID"]
        uid_to_orthanc_id[sop_uid] = inst["ID"]
    print(f"  Série {sid} : {len(instances)} instances indexées")

print(f"Index total : {len(uid_to_orthanc_id)} instances CT")

# ---- 5. Pour chaque frame SEG, retrouver la coupe CT correspondante ----
results = []  # liste de (ct_norm, seg_mask, frame_index)

for i in range(n_frames):
    frame_info = seg_dcm.PerFrameFunctionalGroupsSequence[i]
    ref_sop_uid = str(
        frame_info.DerivationImageSequence[0]
        .SourceImageSequence[0]
        .ReferencedSOPInstanceUID
    )

    ct_instance_id = uid_to_orthanc_id.get(ref_sop_uid)
    if ct_instance_id is None:
        print(f"Frame {i + 1}/{n_frames} : UID CT introuvable, ignorée")
        continue

    r_ct = requests.get(f"{ORTHANC_URL}/instances/{ct_instance_id}/file", auth=AUTH)
    ct_dcm = pydicom.dcmread(BytesIO(r_ct.content))
    ct_img = ct_dcm.pixel_array.astype(np.float32)

    # Fenêtrage pulmonaire
    wl, ww = -600, 1500
    ct_clipped = np.clip(ct_img, wl - ww // 2, wl + ww // 2)
    ct_norm = (ct_clipped - ct_clipped.min()) / (ct_clipped.max() - ct_clipped.min())

    seg_mask = seg_array[i]
    results.append((ct_norm, seg_mask, i))
    print(f"Frame {i + 1}/{n_frames} OK")

# ---- 5. Export PNG dans un dossier ----
output_dir = "export_seg"
os.makedirs(output_dir, exist_ok=True)

for ct_norm, seg_mask, frame_i in results:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ct_norm, cmap="gray")
    ax.imshow(np.ma.masked_where(seg_mask == 0, seg_mask), cmap="Reds", alpha=0.5)
    ax.axis("off")
    fig.savefig(f"{output_dir}/frame_{frame_i + 1:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Exporté : frame_{frame_i + 1:03d}.png")

print(f"\nDone — {len(results)} images dans ./{output_dir}/")
