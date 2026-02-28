from unboxed_ai.lib import OrthancClient


def test_orthanc_client_exposes_notebook_methods() -> None:
    client = OrthancClient()
    assert hasattr(client, "upload_dicom")
    assert hasattr(client, "upload_dicom_folder")
    assert hasattr(client, "download_study")
    assert hasattr(client, "list_studies")

    studies = client.list_studies()
    print("Studies found:")
    for i, study in enumerate(studies, start=1):
        print(
            f"{i}: id={study.get('id', '-')}, "
            f"patient={study.get('patient', '-')}, "
            f"description={study.get('description', '-')}, "
            f"date={study.get('date', '-')}, "
            f"modality={study.get('modality', '-')}"
        )

if __name__ == "__main__":
    test_orthanc_client_exposes_notebook_methods()