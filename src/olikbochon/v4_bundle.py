"""Deterministic source-to-notebook runtime bundling for Version 4-A."""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from pathlib import Path


V4_RUNTIME_MODULES = (
    "__init__.py",
    "data_loading.py",
    "metrics.py",
    "modeling.py",
    "submission.py",
    "v4_kaggle.py",
    "v4_wikipedia.py",
)


def canonical_v4_runtime_files(repository_root: Path) -> dict[str, bytes]:
    source = Path(repository_root) / "src" / "olikbochon"
    return {
        f"olikbochon/{name}": (source / name).read_bytes() for name in V4_RUNTIME_MODULES
    }


def build_v4_runtime_archive(repository_root: Path) -> tuple[bytes, str, dict[str, str]]:
    files = canonical_v4_runtime_files(repository_root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    payload = buffer.getvalue()
    return (
        payload,
        hashlib.sha256(payload).hexdigest(),
        {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())},
    )


def encode_v4_runtime_archive(payload: bytes) -> str:
    return base64.b85encode(payload).decode("ascii")


def materialize_v4_runtime_archive(
    encoded_payload: str, expected_sha256: str, target_directory: Path
) -> Path:
    payload = base64.b85decode(encoded_payload.encode("ascii"))
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(
            f"Embedded V4-A runtime digest mismatch: expected {expected_sha256}, "
            f"observed {observed}"
        )
    target = Path(target_directory)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise RuntimeError("Unsafe path in embedded V4-A runtime archive")
        archive.extractall(target)
    return target
