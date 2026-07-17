from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "src" / "olikbochon" / "v3_modeling.py",
    ROOT / "src" / "olikbochon" / "v3_preprocessing.py",
]


def test_v3_runtime_has_no_remote_loader_or_install_behavior() -> None:
    executable = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES)
    forbidden = (
        "from_pretrained",
        "huggingface.co",
        "hf_hub_download",
        "snapshot_download",
        "pip install",
        "subprocess",
        "requests.",
        "urllib.",
    )
    assert all(token not in executable for token in forbidden)


def test_v3_runtime_never_modifies_tokenizer_or_embeddings() -> None:
    executable = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES)
    forbidden = (
        ".add_tokens(",
        ".add_special_tokens(",
        ".resize_token_embeddings(",
    )
    assert all(token not in executable for token in forbidden)
