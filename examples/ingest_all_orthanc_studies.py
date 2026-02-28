from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from unboxed_ai.lib import Constants, OrthancClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all DICOM studies from a remote Orthanc instance and "
            "save them locally."
        )
    )
    parser.add_argument(
        "--base-url",
        default=Constants.ORTHANC_URL,
        help="Orthanc base URL (default: ORTHANC_URL env var or project default).",
    )
    parser.add_argument(
        "--username",
        default=Constants.ORTHANC_USERNAME,
        help="Orthanc username (default: ORTHANC_USERNAME env var or 'unboxed').",
    )
    parser.add_argument(
        "--password",
        default=Constants.ORTHANC_PASSWORD,
        help="Orthanc password (default: ORTHANC_PASSWORD env var or project default).",
    )
    parser.add_argument(
        "--out-dir",
        default=Constants.DEFAULT_WORK_PATH,
        help=f"Output directory for downloaded studies (default: {Constants.DEFAULT_WORK_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of studies to download.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds for metadata calls (default: 30).",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract each downloaded study zip into a folder.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction (ignored without --extract).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    archives_dir = out_dir / "archives"
    extracted_dir = out_dir / "extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    if args.extract:
        extracted_dir.mkdir(parents=True, exist_ok=True)

    client = OrthancClient(
        base_url=args.base_url,
        username=args.username,
        password=args.password,
        timeout=args.timeout,
    )

    studies = client.list_studies(limit=args.limit)
    total = len(studies)
    if total == 0:
        print("No studies found on remote Orthanc.")
        return 0

    print(f"Found {total} study(ies). Downloading to: {out_dir}")
    downloaded = 0
    failed = 0

    for index, study in enumerate(studies, start=1):
        study_id = str(study["id"])
        try:
            zip_path = Path(client.download_study(study_id, out_dir=str(archives_dir)))
            canonical_zip = archives_dir / f"study_{study_id}.zip"
            if zip_path != canonical_zip:
                shutil.move(str(zip_path), str(canonical_zip))
            else:
                canonical_zip = zip_path

            if args.extract:
                study_target_dir = extracted_dir / study_id
                study_target_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(canonical_zip, "r") as zf:
                    zf.extractall(study_target_dir)
                if not args.keep_zip:
                    canonical_zip.unlink(missing_ok=True)

            downloaded += 1
            print(f"[{index}/{total}] OK    {study_id}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{total}] FAIL  {study_id} -> {exc}")

    print(f"Done. downloaded={downloaded}, failed={failed}, output={out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
