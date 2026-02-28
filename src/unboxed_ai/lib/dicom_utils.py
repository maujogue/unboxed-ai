from __future__ import annotations

from typing import Any


def show_dicom(path: str, show_plot: bool = True) -> dict[str, Any]:
    """
    Load and normalize a DICOM image and optionally render it with matplotlib.

    Requires `pydicom`, `numpy`, and `matplotlib` to be installed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pydicom

    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if show_plot:
        _, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#111111")
        ax.imshow(img, cmap="gray", interpolation="bilinear")
        ax.set_title(
            (
                f"Patient: {getattr(ds, 'PatientID', '?')} | "
                f"Modality: {getattr(ds, 'Modality', '?')} | "
                f"Slice: {getattr(ds, 'InstanceNumber', '?')}"
            ),
            color="white",
            fontsize=10,
            pad=10,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "path": path,
        "shape": tuple(int(dim) for dim in img.shape),
        "patient_id": str(getattr(ds, "PatientID", "N/A")),
        "modality": str(getattr(ds, "Modality", "N/A")),
        "instance_number": str(getattr(ds, "InstanceNumber", "N/A")),
        "pixel_spacing": str(getattr(ds, "PixelSpacing", "N/A")),
        "study_date": str(getattr(ds, "StudyDate", "N/A")),
    }
