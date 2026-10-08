from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sml_hpo.data.registry import load_dataset_splits
from sml_hpo.descriptors.probe import (
    PROBE_FEATURE_NAMES,
    extract_probe_descriptor,
)
from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)
from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.tasks.spec import load_task_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--zero-descriptors",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--probe-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--full-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--probe-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--model-seed",
        type=int,
        default=20260802,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--curvature-epsilon",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def select_dataset(splits, split_name: str):
    if split_name == "train":
        return splits.train_dataset

    if split_name == "validation":
        return splits.validation_dataset

    if split_name == "test":
        return splits.test_dataset

    raise ValueError(split_name)


def main() -> None:
    args = parse_args()

    tasks = load_task_manifest(
        args.manifest
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        tasks = tasks[: args.limit]

    if not tasks:
        raise RuntimeError(
            "No tasks were selected"
        )

    dataset_names = {
        task.dataset
        for task in tasks
    }

    split_names = {
        task.split
        for task in tasks
    }

    if len(dataset_names) != 1:
        raise RuntimeError(
            "Manifest mixes datasets"
        )

    if len(split_names) != 1:
        raise RuntimeError(
            "Manifest mixes splits"
        )

    dataset_name = tasks[0].dataset
    split_name = tasks[0].split

    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable"
            )

        torch.cuda.set_device(
            device.index or 0
        )

    splits = load_dataset_splits(
        dataset_name=dataset_name,
        root=args.data_root,
        image_size=84,
        split_seed=args.split_seed,
        download=False,
    )

    dataset = select_dataset(
        splits,
        split_name,
    )

    print("Manifest:", args.manifest)
    print("Dataset:", dataset_name)
    print("Split:", split_name)
    print("Tasks:", len(tasks))
    print("Probe steps:", args.probe_steps)
    print(
        "Probe validation episodes:",
        args.validation_episodes,
    )
    print("Device:", device)
    print("Probe dimensions:", 20)

    rows: list[dict[str, object]] = []

    run_start = time.perf_counter()

    for task_index, task in enumerate(
        tasks
    ):
        sampler = EpisodeSampler(
            dataset=dataset,
            allowed_classes=task.class_ids,
            n_way=task.n_way,
            n_shot=task.n_shot,
            n_query=task.n_query,
        )

        result = extract_probe_descriptor(
            task=task,
            sampler=sampler,
            device=device,
            probe_steps=args.probe_steps,
            validation_episodes=(
                args.validation_episodes
            ),
            model_seed=args.model_seed,
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
            curvature_relative_epsilon=(
                args.curvature_epsilon
            ),
        )

        row: dict[str, object] = {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "split": task.split,
            "task_seed": task.task_seed,
            "probe_steps": (
                result.probe_steps
            ),
            "probe_validation_episodes": (
                result.validation_episodes
            ),
            "probe_elapsed_seconds": (
                result.elapsed_seconds
            ),
        }

        for feature_name, value in zip(
            PROBE_FEATURE_NAMES,
            result.descriptor,
            strict=True,
        ):
            row[feature_name] = float(
                value
            )

        rows.append(row)

        print(
            f"[{task_index + 1:4d}/"
            f"{len(tasks):4d}] "
            f"{task.task_id} | "
            f"probe_time="
            f"{result.elapsed_seconds:.3f}s | "
            f"val_acc="
            f"{100 * row['probe_validation_accuracy']:.2f}%"
        )

    probe_table = pd.DataFrame(rows)

    probe_values = probe_table[
        list(PROBE_FEATURE_NAMES)
    ].to_numpy(
        dtype=np.float64
    )

    expected_shape = (
        len(tasks),
        20,
    )

    if probe_values.shape != expected_shape:
        raise RuntimeError(
            "Invalid probe table shape: "
            f"{probe_values.shape}"
        )

    if not np.all(
        np.isfinite(probe_values)
    ):
        raise FloatingPointError(
            "Probe table contains "
            "non-finite values"
        )

    args.probe_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.full_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_table.to_csv(
        args.probe_output,
        index=False,
    )

    zero_table = pd.read_csv(
        args.zero_descriptors
    )

    selected_task_ids = {
        task.task_id
        for task in tasks
    }

    zero_table = zero_table[
        zero_table["task_id"].isin(
            selected_task_ids
        )
    ].copy()

    if set(zero_table["task_id"]) != (
        selected_task_ids
    ):
        missing = (
            selected_task_ids
            - set(zero_table["task_id"])
        )

        raise RuntimeError(
            "Zero descriptors are missing "
            f"tasks: {sorted(missing)}"
        )

    duplicate_task_ids = (
        zero_table["task_id"]
        .duplicated()
        .any()
    )

    if duplicate_task_ids:
        raise RuntimeError(
            "Zero descriptor table "
            "contains duplicate task IDs"
        )

    probe_feature_table = probe_table[
        [
            "task_id",
            *PROBE_FEATURE_NAMES,
            "probe_steps",
            "probe_validation_episodes",
            "probe_elapsed_seconds",
        ]
    ]

    full_table = zero_table.merge(
        probe_feature_table,
        on="task_id",
        how="inner",
        validate="one_to_one",
    )

    full_feature_names = (
        list(ZERO_FEATURE_NAMES)
        + list(PROBE_FEATURE_NAMES)
    )

    full_values = full_table[
        full_feature_names
    ].to_numpy(
        dtype=np.float64
    )

    if full_values.shape != (
        len(tasks),
        84,
    ):
        raise RuntimeError(
            f"Full descriptor table has "
            f"invalid shape: "
            f"{full_values.shape}"
        )

    if not np.all(
        np.isfinite(full_values)
    ):
        raise FloatingPointError(
            "Full 84-D descriptor contains "
            "non-finite values"
        )

    full_table.to_csv(
        args.full_output,
        index=False,
    )

    elapsed_seconds = (
        time.perf_counter()
        - run_start
    )

    metadata = {
        "schema_version": 1,
        "descriptor_name": (
            "probe_sml_84d"
        ),
        "dataset": dataset_name,
        "split": split_name,
        "task_count": len(tasks),
        "zero_feature_count": 64,
        "probe_feature_count": 20,
        "total_feature_count": 84,
        "zero_feature_names": list(
            ZERO_FEATURE_NAMES
        ),
        "probe_feature_names": list(
            PROBE_FEATURE_NAMES
        ),
        "manifest": str(
            args.manifest
        ),
        "manifest_sha256": file_sha256(
            args.manifest
        ),
        "zero_descriptors": str(
            args.zero_descriptors
        ),
        "zero_descriptors_sha256": (
            file_sha256(
                args.zero_descriptors
            )
        ),
        "probe_output": str(
            args.probe_output
        ),
        "probe_output_sha256": (
            file_sha256(
                args.probe_output
            )
        ),
        "full_output": str(
            args.full_output
        ),
        "full_output_sha256": (
            file_sha256(
                args.full_output
            )
        ),
        "probe_steps": (
            args.probe_steps
        ),
        "validation_episodes": (
            args.validation_episodes
        ),
        "probe_optimizer": "AdamW",
        "probe_learning_rate": (
            args.learning_rate
        ),
        "probe_weight_decay": (
            args.weight_decay
        ),
        "model_seed": (
            args.model_seed
        ),
        "curvature_relative_epsilon": (
            args.curvature_epsilon
        ),
        "device": str(device),
        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    metadata_path = (
        args.full_output
        .with_suffix(
            ".metadata.json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Probe table:",
        args.probe_output,
    )
    print(
        "Full 84-D table:",
        args.full_output,
    )
    print(
        "Metadata:",
        metadata_path,
    )
    print(
        "Probe shape:",
        probe_values.shape,
    )
    print(
        "Full feature shape:",
        full_values.shape,
    )
    print(
        f"Elapsed: "
        f"{elapsed_seconds:.2f} seconds"
    )
    print(
        "PROBE DESCRIPTOR "
        "EXTRACTION: PASS"
    )


if __name__ == "__main__":
    main()
