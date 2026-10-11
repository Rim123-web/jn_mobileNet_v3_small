from __future__ import annotations

import csv
import hashlib
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
MANIFEST = RUN / "validation/file_hash_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


rows = []
for path in sorted(RUN.rglob("*")):
    if path.is_file() and path != MANIFEST:
        rows.append(
            {
                "relative_path": path.relative_to(RUN).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)

print(f"HASH MANIFEST REFRESH PASS: {len(rows)} files")
