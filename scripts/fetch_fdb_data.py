#!/usr/bin/env python3
"""Download and unpack the FDB-v3 audio (Google Drive link from the v3 README)
and print the path of the directory holding the {example_id}_{speaker_id}/ folders."""

import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

GDRIVE_ID = "1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz"
FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def find_data_root(base: Path) -> Path | None:
    candidates = [base] + [p for p in base.rglob("*") if p.is_dir()]
    for c in candidates:
        kids = [k for k in c.iterdir() if k.is_dir() and FOLDER_RE.match(k.name) and (k / "input.wav").exists()]
        if len(kids) >= 10:
            return c
    return None


def main(dest: str) -> None:
    dest_p = Path(dest).resolve()
    dest_p.mkdir(parents=True, exist_ok=True)
    found = find_data_root(dest_p)
    if found:
        print(found)
        return
    import gdown

    archive = dest_p / "fdb_v3_data_download"
    if not archive.exists():
        gdown.download(id=GDRIVE_ID, output=str(archive), quiet=False)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_p)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest_p)
    else:
        sys.exit(f"unrecognized archive format: {archive}")
    shutil.rmtree(dest_p / "__MACOSX", ignore_errors=True)
    found = find_data_root(dest_p)
    if not found:
        sys.exit(f"could not locate example folders under {dest_p}")
    print(found)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "third_party/fdb_data")
