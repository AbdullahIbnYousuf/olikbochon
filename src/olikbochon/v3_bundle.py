"""Deterministic source-to-notebook runtime bundle construction."""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from pathlib import Path


PROJECT_MODULES = (
    "__init__.py",
    "data_loading.py",
    "file_discovery.py",
    "metrics.py",
    "modeling.py",
    "near_duplicates.py",
    "preprocessing.py",
    "submission.py",
    "v3_bundle.py",
    "v3_data.py",
    "v3_default_normalizer.py",
    "v3_kaggle.py",
    "v3_modeling.py",
    "v3_normalizer_constants.py",
    "v3_preprocessing.py",
    "v3_selection.py",
    "v3_training.py",
)


def canonical_runtime_files(repository_root: Path) -> dict[str, bytes]:
    """Collect the exact runtime and attribution files embedded in the notebook."""
    root = Path(repository_root)
    source = root / "src"
    files: dict[str, bytes] = {}
    for name in PROJECT_MODULES:
        files[f"olikbochon/{name}"] = (source / "olikbochon" / name).read_bytes()
    files["olikbochon/bangla_normalizer/NOTICE.md"] = (
        source / "olikbochon" / "bangla_normalizer" / "NOTICE.md"
    ).read_bytes()
    for notice in sorted((source / "olikbochon" / "v3_vendor_notices").glob("*")):
        if notice.is_file():
            files[f"olikbochon/v3_vendor_notices/{notice.name}"] = notice.read_bytes()
    for package in ("ftfy", "wcwidth"):
        package_root = source / package
        for path in sorted(package_root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[f"{package}/{path.relative_to(package_root).as_posix()}"] = path.read_bytes()
    return files


def build_runtime_archive(repository_root: Path) -> tuple[bytes, str, dict[str, str]]:
    """Build a byte-identical ZIP with fixed metadata and per-file hashes."""
    files = canonical_runtime_files(repository_root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    payload = buffer.getvalue()
    return (
        payload,
        hashlib.sha256(payload).hexdigest(),
        {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())},
    )


def encode_runtime_archive(payload: bytes) -> str:
    return base64.b85encode(payload).decode("ascii")


def materialize_runtime_archive(
    encoded_payload: str,
    expected_sha256: str,
    target_directory: Path,
) -> Path:
    """Verify and extract the trusted embedded archive without repository access."""
    payload = base64.b85decode(encoded_payload.encode("ascii"))
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(
            f"Embedded runtime digest mismatch: expected {expected_sha256}, observed {observed}"
        )
    target = Path(target_directory)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise RuntimeError("Unsafe path in embedded runtime archive")
        archive.extractall(target)
    return target
