"""Verify all completed V7 artifacts without reading competition test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from olikbochon.data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from olikbochon.v7_protocol import (
    OUTER_FOLDS,
    OUTER_SEEDS,
    metric_record,
    outer_indices,
    stable_training_row_ids,
    validate_common_fold_assignments,
    validate_probability_frame,
)


EXPERIMENTS = ("e0", "e1", "e2", "e3", "e4")
STACK_BASES = ("e0", "e1", "e2")


def _verify_oof(
    name: str,
    directory: Path,
    row_ids: np.ndarray,
    assignment_keys: set[tuple[object, ...]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    oof = pd.read_csv(directory / "oof_predictions.csv")
    validate_probability_frame(oof, row_ids)
    keys = set(oof[["row_id", "seed", "outer_fold"]].itertuples(index=False, name=None))
    if keys != assignment_keys:
        raise RuntimeError(f"{name} OOF rows do not match frozen fold assignments")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if int(summary.get("test_rows_accessed", -1)) != 0:
        raise RuntimeError(f"{name} summary does not prove zero test access")
    aggregate = metric_record(
        oof["label"], oof["probability_label_1"], oof["prediction"], oof["context_present"]
    )
    for metric in ("macro_f1", "f1_label0", "f1_label1", "accuracy", "brier_score"):
        if not np.isclose(
            aggregate[metric], summary["aggregate_repeated_oof"][metric], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"{name} summary does not reproduce {metric}")
    return oof, aggregate


def _verify_stack(
    name: str, directory: Path, folds: pd.DataFrame, row_ids: np.ndarray
) -> int:
    table = pd.read_csv(directory / "stack_training_predictions.csv")
    columns = [f"probability_label_0_{name}", f"probability_label_1_{name}"]
    required = {
        "row_id", "seed", "target_outer_fold", *columns, "base_prediction_provenance"
    }
    if not required.issubset(table.columns):
        raise RuntimeError(f"{name} stack-training matrix has an incomplete schema")
    if set(table["base_prediction_provenance"]) != {"outer_train_inner_grouped_cross_fit"}:
        raise RuntimeError(f"{name} stack-training provenance is not inner cross-fitted")
    values = table[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.allclose(
        values.sum(axis=1), 1.0, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError(f"{name} stack-training probabilities are invalid")
    for seed in OUTER_SEEDS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            train, validation = outer_indices(folds, row_ids, seed=seed, outer_fold=outer_fold)
            selected = table.loc[
                (table["seed"] == seed) & (table["target_outer_fold"] == outer_fold),
                "row_id",
            ].astype(str)
            if selected.duplicated().any() or set(selected) != set(row_ids[train]):
                raise RuntimeError(f"{name} stack rows do not equal outer-training rows")
            if set(selected) & set(row_ids[validation]):
                raise RuntimeError(f"{name} stack contains outer-validation rows")
    return len(table)


def _model_records(name: str, directory: Path) -> list[dict[str, object]]:
    if name == "e2":
        manifest = json.loads((directory / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        return list(manifest["model_artifacts"])
    path = directory / "model_manifest.json"
    return list(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else []


def verify(train_path: Path, artifact_root: Path) -> dict[str, object]:
    frame = load_labeled_json(train_path, expected_sha256=OFFICIAL_SAMPLE_SHA256)
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    folds = pd.read_csv(artifact_root / "folds" / "common_folds.csv")
    validate_common_fold_assignments(folds, row_ids)
    assignment_keys = set(
        folds[["row_id", "seed", "outer_fold"]].itertuples(index=False, name=None)
    )
    results: dict[str, object] = {}
    for name in EXPERIMENTS:
        directory = artifact_root / name
        oof, aggregate = _verify_oof(name, directory, row_ids, assignment_keys)
        stack_rows = _verify_stack(name, directory, folds, row_ids) if name in STACK_BASES else 0
        models = _model_records(name, directory)
        for record in models:
            path = directory / str(record["path"])
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"{name} model checksum mismatch: {record['path']}")
            if float(record["reload_max_abs_probability_difference"]) > 1e-12:
                raise RuntimeError(f"{name} model reload tolerance failed: {record['path']}")
        results[name] = {
            "oof_rows": len(oof),
            "stack_training_rows": stack_rows,
            "models_verified": len(models),
            "macro_f1": aggregate["macro_f1"],
        }
    return {
        "status": "COMPLETE",
        "official_rows": len(frame),
        "seeds": len(OUTER_SEEDS),
        "outer_folds": len(OUTER_SEEDS) * OUTER_FOLDS,
        "experiments": results,
        "test_rows_accessed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/v7"))
    args = parser.parse_args()
    print(json.dumps(verify(args.train_data, args.artifact_root), sort_keys=True))


if __name__ == "__main__":
    main()
