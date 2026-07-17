"""End-to-end offline Kaggle orchestration for the BanglaBERT Version 3 baseline."""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    classification_metrics,
    prediction_collapse_warning,
    predictions_from_label1,
)
from .near_duplicates import (
    audit_near_duplicates,
    enforce_near_duplicate_safety,
    remove_training_side_near_duplicates,
)
from .submission import build_submission, validate_submission, validate_test_frame
from .v3_data import (
    MODEL_REVISION,
    MODEL_WEIGHT_SHA256,
    V3Files,
    audit_partition_safety,
    authenticate_model_directory,
    build_official_groups,
    data_manifest,
    discover_v3_files,
    load_labeled_json,
    make_official_folds,
    safe_discovery_summary,
)
from .v3_default_normalizer import verify_frozen_reference_corpus
from .v3_selection import (
    choose_arm,
    choose_deployment_threshold,
    evaluate_oof,
    prediction_distribution,
    write_safe_summary,
)
from .v3_training import (
    EXPECTED_BASE_MISSING,
    EXPECTED_BASE_UNEXPECTED,
    OPTIONAL_POSITION_IDS_KEY,
    TrainConfig,
    cleanup_cuda,
    encode_frame,
    infer_probabilities,
    is_cuda_oom,
    load_base_classifier,
    load_saved_classifier,
    load_tokenizer,
    run_cross_validation,
    run_with_oom_restart,
    set_all_seeds,
    train_attempt,
)


WORKING = Path("/kaggle/working")
STAGE_A_DIRECTORY = WORKING / "banglabert_v3_stage_a_selected"
FINAL_MODEL_DIRECTORY = WORKING / "banglabert_v3_model"
SUMMARY_PATH = WORKING / "v3_run_summary.json"


def configure_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"


def _environment_summary() -> dict[str, Any]:
    import tokenizers
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Version 3 must not fall back to CPU training")
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "cuda_available": True,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "determinism": (
            "seeded Python/NumPy/PyTorch/CUDA with deterministic cuDNN; some GPU kernels may "
            "remain nondeterministic"
        ),
    }


def _load_and_audit_data(files: V3Files) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    train = load_labeled_json(files.public_train)
    validation = load_labeled_json(files.public_validation)
    official = load_labeled_json(files.official_train)
    frames = {
        "public_train": train,
        "public_validation": validation,
        "official": official,
    }
    exact = audit_partition_safety(frames)
    train_validation = audit_near_duplicates(
        train,
        validation,
        left_name="public_train",
        right_name="public_validation",
    )
    combined_public = pd.concat(
        [
            train.assign(partition="public_train", source_index=np.arange(len(train))),
            validation.assign(
                partition="public_validation", source_index=np.arange(len(validation))
            ),
        ],
        ignore_index=True,
    )
    public_official = audit_near_duplicates(
        combined_public,
        official,
        left_name="combined_public",
        right_name="official",
    )
    enforce_near_duplicate_safety(train_validation, public_official)
    train_clean, validation_clean, removals = remove_training_side_near_duplicates(
        train,
        validation,
        official,
        train_validation,
        public_official,
        combined_public["partition"],
        combined_public["source_index"],
    )
    cleaned = {
        "public_train": train_clean,
        "public_validation": validation_clean,
        "official": official,
    }
    audit = {
        "exact": exact,
        "near_duplicate": {
            "public_train_validation_pairs": train_validation.high_confidence_pairs,
            "public_official_pairs": public_official.high_confidence_pairs,
            "label_conflicts": (
                train_validation.label_conflict_pairs + public_official.label_conflict_pairs
            ),
            "removals": removals,
        },
    }
    return cleaned, audit


def _phase_runner(
    phase: str,
    start_checkpoint: Path,
    tokenizer: Any,
    train_data: Any,
    validation_data: Any,
    config: TrainConfig,
    *,
    base_checkpoint: bool,
    selected_checkpoint_directory: Path | None = None,
    final_checkpoint_directory: Path | None = None,
) -> tuple[Any, Any]:
    def runner(active: TrainConfig, _attempt: int) -> Any:
        for directory in (selected_checkpoint_directory, final_checkpoint_directory):
            if directory is not None and directory.exists():
                shutil.rmtree(directory)
        return train_attempt(
            phase,
            start_checkpoint,
            tokenizer,
            train_data,
            validation_data,
            active,
            base_checkpoint=base_checkpoint,
            selected_checkpoint_directory=selected_checkpoint_directory,
            final_checkpoint_directory=final_checkpoint_directory,
        )

    return run_with_oom_restart(
        phase,
        config,
        runner,
        is_oom=is_cuda_oom,
        cleanup=cleanup_cuda,
        seed_reset=set_all_seeds,
    )


def _copy_attribution(final_directory: Path, base_model_directory: Path) -> None:
    runtime_root = Path(__file__).resolve().parent
    source_notice = runtime_root / "bangla_normalizer" / "NOTICE.md"
    vendor_notices = runtime_root / "v3_vendor_notices"
    shutil.copy2(source_notice, final_directory / "BANGLABERT_NORMALIZER_NOTICE.md")
    shutil.copy2(vendor_notices / "FTFY_LICENSE.txt", final_directory / "FTFY_LICENSE.txt")
    shutil.copy2(vendor_notices / "WCWIDTH_LICENSE.txt", final_directory / "WCWIDTH_LICENSE.txt")
    shutil.copy2(
        base_model_directory / "README.md", final_directory / "BANGLABERT_MODEL_README.md"
    )
    snapshot_info = base_model_directory / "SNAPSHOT_INFO.md"
    if snapshot_info.is_file():
        shutil.copy2(snapshot_info, final_directory / "SNAPSHOT_INFO.md")
    (final_directory / "V3_ATTRIBUTION.md").write_text(
        "# Version 3 attribution\n\n"
        "This private academic-competition artifact fine-tunes the official "
        "csebuetnlp/BanglaBERT snapshot at revision "
        f"{MODEL_REVISION}. BanglaBERT and the official normalizer declare "
        "CC BY-NC-SA 4.0. The normalizer was adapted to its audited default-only "
        "NFKC path for deterministic offline use. See the included official model "
        "README, snapshot information, normalizer notice, and dependency licenses.\n",
        encoding="utf-8",
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_kaggle_v3(input_root: Path = Path("/kaggle/input")) -> dict[str, Any]:
    """Execute all frozen decisions, then train and infer once on Kaggle."""
    configure_offline_environment()
    total_started = perf_counter()
    normalizer = verify_frozen_reference_corpus()
    environment = _environment_summary()
    set_all_seeds(42)

    audit_started = perf_counter()
    files = discover_v3_files(input_root)
    model_authentication = authenticate_model_directory(files.model_directory)
    print(
        json.dumps(
            {
                "environment": environment,
                "normalizer_startup_verification": normalizer,
                "authenticated_model": model_authentication,
                "resolved_inputs": safe_discovery_summary(files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    frames, data_audit = _load_and_audit_data(files)
    manifest = data_manifest(files, frames)
    group_audit = build_official_groups(frames["official"])
    labels = frames["official"]["label"].to_numpy(dtype=np.int64)
    fold_assignment = make_official_folds(labels, group_audit)
    data_audit_seconds = perf_counter() - audit_started

    tokenizer = load_tokenizer(files.model_directory)
    transition_model, transition_info = load_base_classifier(files.model_directory, tokenizer)
    del transition_model
    cleanup_cuda()
    loading_validation = {
        "expected_missing_keys": sorted(EXPECTED_BASE_MISSING),
        "expected_unexpected_keys": sorted(EXPECTED_BASE_UNEXPECTED),
        "observed_missing_keys": sorted(transition_info["missing_keys"]),
        "observed_unexpected_keys": sorted(transition_info["unexpected_keys"]),
        "allowed_optional_unexpected_key": OPTIONAL_POSITION_IDS_KEY,
        "base_transition_optional_position_ids_buffer_appeared": (
            OPTIONAL_POSITION_IDS_KEY in transition_info["unexpected_keys"]
        ),
        "mismatched_key_count": len(transition_info["mismatched_keys"]),
        "error_message_count": len(transition_info["error_msgs"]),
        "passed": True,
    }

    encoded_public_train = encode_frame(frames["public_train"], tokenizer, with_labels=True)
    encoded_public_validation = encode_frame(
        frames["public_validation"], tokenizer, with_labels=True
    )
    encoded_official = encode_frame(frames["official"], tokenizer, with_labels=True)

    stage_config = TrainConfig(epochs=2, learning_rate=2e-5)
    stage_output, stage_restart = _phase_runner(
        "stage_a",
        files.model_directory,
        tokenizer,
        encoded_public_train,
        encoded_public_validation,
        stage_config,
        base_checkpoint=True,
        selected_checkpoint_directory=STAGE_A_DIRECTORY,
    )
    stage_model, stage_loading_info = load_saved_classifier(STAGE_A_DIRECTORY, tokenizer)
    del stage_model
    cleanup_cuda()
    loading_validation["stage_a_reload_optional_position_ids_buffer_appeared"] = (
        OPTIONAL_POSITION_IDS_KEY in stage_loading_info.get("unexpected_keys", [])
    )

    official_config = TrainConfig(epochs=4, learning_rate=1e-5)
    arm_a = run_cross_validation(
        "arm_a",
        files.model_directory,
        tokenizer,
        encoded_official,
        fold_assignment.folds,
        official_config,
        base_checkpoint=True,
    )
    arm_b = run_cross_validation(
        "arm_b",
        STAGE_A_DIRECTORY,
        tokenizer,
        encoded_official,
        fold_assignment.folds,
        official_config,
        base_checkpoint=False,
    )
    context_presence = np.asarray(encoded_official.context_present, dtype=bool)
    arm_a_report = evaluate_oof(
        labels, arm_a.probabilities, list(arm_a.fold_metrics), context_presence
    )
    arm_b_report = evaluate_oof(
        labels, arm_b.probabilities, list(arm_b.fold_metrics), context_presence
    )
    arm_decision = choose_arm(labels, arm_a.probabilities, arm_b.probabilities)
    selected_probabilities = (
        arm_b.probabilities if arm_decision.selected_arm == "arm_b" else arm_a.probabilities
    )
    threshold_decision = choose_deployment_threshold(labels, selected_probabilities)
    threshold_predictions = predictions_from_label1(
        selected_probabilities, threshold_decision.tuned_threshold
    )
    threshold_tuning_report = {
        "label": "Official OOF threshold-tuning estimate",
        "metrics": classification_metrics(labels, threshold_predictions),
        "prediction_distribution": prediction_distribution(threshold_predictions),
    }

    final_start = (
        STAGE_A_DIRECTORY if arm_decision.selected_arm == "arm_b" else files.model_directory
    )
    final_output, final_restart = _phase_runner(
        "final_training",
        final_start,
        tokenizer,
        encoded_official,
        None,
        official_config,
        base_checkpoint=arm_decision.selected_arm == "arm_a",
        final_checkpoint_directory=FINAL_MODEL_DIRECTORY,
    )
    _copy_attribution(FINAL_MODEL_DIRECTORY, files.model_directory)
    final_metadata = {
        "selected_arm": arm_decision.selected_arm,
        "frozen_threshold": threshold_decision.deployed_threshold,
        "model_revision": MODEL_REVISION,
        "original_weight_sha256": MODEL_WEIGHT_SHA256,
        "training_configuration": asdict(official_config),
        "normalizer_reference_revision": normalizer["reference_revision"],
        "normalizer_configuration": "locked default normalize(text), NFKC-last",
        "response_truncation": "384 tokens: ceil head plus floor tail",
        "data_sources": [
            "official dataset samples.json",
            "abidur14004/new-dataset public 4k train and 1k validation",
        ],
        "license": "CC BY-NC-SA 4.0 model/normalizer; private academic competition use",
    }
    (FINAL_MODEL_DIRECTORY / "V3_METADATA.json").write_text(
        json.dumps(final_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_model_size = _directory_size(FINAL_MODEL_DIRECTORY)

    inference_started = perf_counter()
    test = pd.read_csv(files.test)
    validate_test_frame(test)
    sample_submission = pd.read_csv(files.sample_submission)
    if test["id"].duplicated().any() or sample_submission["id"].duplicated().any():
        raise RuntimeError("Version 3 requires unique test and sample-submission IDs")
    encoded_test = encode_frame(test, tokenizer, with_labels=False)
    test_probabilities, inference_peak, final_loading_info = infer_probabilities(
        FINAL_MODEL_DIRECTORY, tokenizer, encoded_test, batch_size=8
    )
    loading_validation["final_reload_optional_position_ids_buffer_appeared"] = (
        OPTIONAL_POSITION_IDS_KEY in final_loading_info.get("unexpected_keys", [])
    )
    deployed_predictions = predictions_from_label1(
        test_probabilities, threshold_decision.deployed_threshold
    )
    submission = build_submission(test["id"], deployed_predictions, sample_submission)
    validate_submission(submission, test["id"], sample_submission)
    submission.to_csv(WORKING / "submission.csv", index=False)
    fixed_created = False
    if threshold_decision.deployed_threshold != 0.5:
        fixed = build_submission(
            test["id"], predictions_from_label1(test_probabilities, 0.5), sample_submission
        )
        validate_submission(fixed, test["id"], sample_submission)
        fixed.to_csv(WORKING / "submission_fixed_050.csv", index=False)
        fixed_created = True
    inference_seconds = perf_counter() - inference_started
    test_collapse_warning = prediction_collapse_warning(
        deployed_predictions, name="final test inference"
    )
    print("schema validation passed")
    print("row-count validation passed")
    print("ID-sequence validation passed")
    print("label-domain validation passed")
    if test_collapse_warning is not None:
        print(test_collapse_warning)

    maximum_memory = max(
        final_output.maximum_allocated_bytes,
        arm_a.maximum_allocated_bytes,
        arm_b.maximum_allocated_bytes,
        inference_peak,
    )
    summary = {
        "model": model_authentication,
        "environment": environment,
        "normalizer": normalizer,
        "data_manifest": manifest,
        "data_audit": data_audit,
        "official_group_audit": {
            "group_count": group_audit.group_count,
            "nontrivial_group_count": group_audit.nontrivial_groups,
            "largest_group": group_audit.largest_group,
            "exact_duplicate_pair_count": group_audit.exact_duplicate_pairs,
            "prompt_context_pair_count": group_audit.prompt_context_pairs,
            "near_duplicate_pair_count": group_audit.near_duplicate_pairs,
        },
        "folds": {
            "strategy": fold_assignment.strategy,
            "class_counts": fold_assignment.fold_class_counts,
            "group_leakage": False,
        },
        "loading_validation": loading_validation,
        "stage_a": {
            "epoch_metrics": stage_output.epoch_history,
            "selected_epoch": stage_output.selected_epoch,
            "oom_restart": asdict(stage_restart),
        },
        "arm_a": {"oof": arm_a_report, "oom_restarts": arm_a.restart_records},
        "arm_b": {"oof": arm_b_report, "oom_restarts": arm_b.restart_records},
        "arm_selection": {
            "label": "Official OOF model-selection estimate",
            "optimistic": True,
            **asdict(arm_decision),
        },
        "threshold_selection": {
            "optimistic": True,
            **asdict(threshold_decision),
            "tuning_report": threshold_tuning_report,
        },
        "final_training": {
            "configuration": asdict(official_config),
            "oom_restart": asdict(final_restart),
            "model_size_bytes": final_model_size,
        },
        "submission": {
            "row_count": int(len(submission)),
            "prediction_distribution": prediction_distribution(deployed_predictions),
            "response_fallback_count": encoded_test.response_fallback_count,
            "fixed_050_created": fixed_created,
            "schema_validation": True,
            "id_sequence_validation": True,
            "label_domain_validation": True,
            "collapse_warning": test_collapse_warning,
        },
        "runtime": {
            "data_audit_seconds": data_audit_seconds,
            "stage_a_seconds": stage_restart.successful_attempt_seconds,
            "arm_a_fold_seconds": [item.successful_attempt_seconds for item in arm_a.restart_records],
            "arm_b_fold_seconds": [item.successful_attempt_seconds for item in arm_b.restart_records],
            "final_training_seconds": final_restart.successful_attempt_seconds,
            "test_inference_seconds": inference_seconds,
            "total_seconds": perf_counter() - total_started,
            "maximum_allocated_gpu_bytes": maximum_memory,
        },
    }
    write_safe_summary(SUMMARY_PATH, summary)
    safe_console = {
        "environment": environment,
        "model_authentication": model_authentication,
        "normalizer": normalizer,
        "dynamic_data_counts": {
            name: int(len(frame)) for name, frame in frames.items()
        },
        "fold_strategy": fold_assignment.strategy,
        "selected_arm": arm_decision.selected_arm,
        "deployed_threshold": threshold_decision.deployed_threshold,
        "submission_row_count": int(len(submission)),
        "test_prediction_distribution": prediction_distribution(deployed_predictions),
        "response_fallback_count": encoded_test.response_fallback_count,
        "total_seconds": summary["runtime"]["total_seconds"],
    }
    print(json.dumps(safe_console, indent=2, sort_keys=True))
    gc.collect()
    return summary
