"""Regenerate deterministic hashes for vendored Version 3 dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "olikbochon" / "v3_vendor_notices" / "VENDOR_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files(name: str) -> dict[str, str]:
    root = ROOT / "src" / name
    return {
        path.relative_to(ROOT / "src").as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def generate() -> None:
    manifest = {
        "ftfy": {
            "version": "6.0.3",
            "license": "MIT",
            "files": package_files("ftfy"),
        },
        "wcwidth": {
            "version": "0.8.2",
            "license": "MIT",
            "files": package_files("wcwidth"),
        },
    }
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate()
