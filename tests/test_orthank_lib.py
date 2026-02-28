import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unboxed_ai.lib import OrthancClient


class TestOrthancClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OrthancClient(
            base_url="http://orthanc.local:8042",
            username="user",
            password="pass",
            timeout=7,
        )

    @patch("unboxed_ai.lib.OrthancClient.requests.get")
    def test_list_studies(self, mock_get: Mock) -> None:
        studies_response = Mock()
        studies_response.raise_for_status.return_value = None
        studies_response.json.return_value = ["study-a"]

        detail_response = Mock()
        detail_response.raise_for_status.return_value = None
        detail_response.json.return_value = {
            "MainDicomTags": {
                "PatientID": "P1",
                "AccessionNumber": "ACC123",
                "StudyDescription": "CT",
                "StudyDate": "20260101",
                "ModalitiesInStudy": "CT",
            }
        }
        mock_get.side_effect = [studies_response, detail_response]

        rows = self.client.list_studies(limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "study-a")
        self.assertEqual(rows[0]["patient"], "P1")
        self.assertEqual(rows[0]["accession"], "ACC123")
        self.assertEqual(rows[0]["modality"], "CT")

    @patch("unboxed_ai.lib.OrthancClient.requests.post")
    def test_upload_dicom_returns_instance_id_when_successful(
        self, mock_post: Mock
    ) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"ID": "instance-1"}
        mock_post.return_value = response

        with tempfile.TemporaryDirectory() as tmp_dir:
            dicom_path = Path(tmp_dir) / "file.dcm"
            dicom_path.write_bytes(b"DICOMDATA")
            instance_id = self.client.upload_dicom(str(dicom_path))

        self.assertEqual(instance_id, "instance-1")
        mock_post.assert_called_once()

    @patch("unboxed_ai.lib.OrthancClient.requests.post")
    def test_upload_dicom_returns_none_on_http_error(self, mock_post: Mock) -> None:
        response = Mock(status_code=500)
        response.json.return_value = {}
        mock_post.return_value = response

        with tempfile.TemporaryDirectory() as tmp_dir:
            dicom_path = Path(tmp_dir) / "file.dcm"
            dicom_path.write_bytes(b"DICOMDATA")
            instance_id = self.client.upload_dicom(str(dicom_path))

        self.assertIsNone(instance_id)

    @patch.object(OrthancClient, "upload_dicom")
    def test_upload_dicom_folder_counts_results(self, mock_upload_dicom: Mock) -> None:
        mock_upload_dicom.side_effect = ["id-1", None]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.dcm").write_bytes(b"a")
            (root / "nested").mkdir()
            (root / "nested" / "b.dcm").write_bytes(b"b")

            result = self.client.upload_dicom_folder(str(root))

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["uploaded"], 1)
        self.assertEqual(result["failed"], 1)

    @patch("unboxed_ai.lib.OrthancClient.requests.get")
    def test_download_study_writes_zip_file(self, mock_get: Mock) -> None:
        response = MagicMock()
        response.iter_content.return_value = [b"abc", b"def"]
        response.raise_for_status.return_value = None
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        mock_get.return_value = response

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = self.client.download_study("study1234", out_dir=tmp_dir)
            content = Path(output).read_bytes()

        self.assertTrue(output.endswith("study_study123.zip"))
        self.assertEqual(content, b"abcdef")


class TestShowDicom(unittest.TestCase):
    def test_show_dicom_without_plot_uses_pydicom_data(self) -> None:
        import numpy as np

        fake_pixels = np.array([[0, 1], [2, 3]])
        fake_ds = SimpleNamespace(
            pixel_array=fake_pixels,
            PatientID="PAT-1",
            Modality="CT",
            InstanceNumber=7,
            PixelSpacing=[0.7, 0.7],
            StudyDate="20260228",
        )

        with patch("pydicom.dcmread", return_value=fake_ds):
            from unboxed_ai.lib import show_dicom

            result = show_dicom("fake.dcm", show_plot=False)

        self.assertEqual(result["shape"], (2, 2))
        self.assertEqual(result["patient_id"], "PAT-1")
        self.assertEqual(result["modality"], "CT")
