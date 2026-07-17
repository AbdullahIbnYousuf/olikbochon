from __future__ import annotations

import ast
from pathlib import Path

import nbformat

from olikbochon.v3_bundle import canonical_runtime_files


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "generated" / "banglabert_v3.ipynb"


def runtime_python_source() -> str:
    files = canonical_runtime_files(ROOT)
    return "\n".join(
        content.decode("utf-8")
        for name, content in files.items()
        if name.endswith(".py")
    )


def runtime_python_sources() -> list[str]:
    return [
        content.decode("utf-8")
        for name, content in canonical_runtime_files(ROOT).items()
        if name.endswith(".py")
    ]


def test_runtime_has_no_network_install_or_rejected_model_behavior() -> None:
    source = runtime_python_source().lower()
    forbidden = (
        "reasat/banglabert",
        "hf_hub_download",
        "snapshot_download",
        "pip install",
        "!pip",
        "wget ",
        "curl ",
        "git clone",
        "openai",
        "requests.",
        "urllib.request",
        ".add_tokens(",
        ".add_special_tokens(",
        ".resize_token_embeddings(",
    )
    assert all(token not in source for token in forbidden)


def test_every_transformers_loader_is_local_only_and_remote_code_disabled() -> None:
    calls = []
    for source in runtime_python_sources():
        tree = ast.parse(source)
        calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        )
    assert calls
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(keywords["local_files_only"]) is True
        assert ast.literal_eval(keywords["trust_remote_code"]) is False
        assert not (call.args and isinstance(call.args[0], ast.Constant))


def test_offline_flags_are_set_before_embedded_runtime_import() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    import_position = code.index("from olikbochon.v3_kaggle import run_kaggle_v3")
    assert code.index('os.environ["HF_HUB_OFFLINE"] = "1"') < import_position
    assert code.index('os.environ["TRANSFORMERS_OFFLINE"] = "1"') < import_position


def test_notebook_does_not_display_test_rows_or_predictions() -> None:
    source = runtime_python_source()
    forbidden = (
        "print(test)",
        "print(test.head",
        "display(test",
        "print(submission)",
        "display(submission",
        "to_clipboard",
    )
    assert all(token not in source for token in forbidden)
