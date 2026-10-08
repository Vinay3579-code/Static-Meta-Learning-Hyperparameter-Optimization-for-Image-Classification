from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sml_hpo.data.registry import load_dataset_splits
from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.oracle.evaluator import (
    evaluate_anchor_config,
)
from sml_hpo.oracle.search_space import (
    load_anchor_manifest,
)
from sml_hpo.tasks.spec import load_task_manifest


DEFAULT_ROOTS = {
    "omniglot": Path("data/raw/omniglot"),
    "cifar100": Path("data/raw/cifar100"),
    "miniimagenet": Path("data/raw/miniimagenet"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task-id",
        type=str,
    )

    parser.add_argument(
        "--anchors",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--best-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[101],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit-configs",
        type=int,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
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


def atomic_write_csv(
    table: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    table.to_csv(
        temporary_path,
        index=False,
    )

    os.replace(
        temporary_path,
        path,
    )


def main() -> None:
    args = parse_args()

    tasks = load_task_manifest(
        args.task_manifest
    )

    if not tasks:
        raise RuntimeError(
            "Task manifest is empty"
        )

    if args.task_id is None:
        task = tasks[0]
    else:
        matching_tasks = [
            item
            for item in tasks
            if item.task_id == args.task_id
        ]

        if len(matching_tasks) != 1:
            raise ValueError(
                f"Could not uniquely resolve "
                f"task ID {args.task_id!r}"
            )

        task = matching_tasks[0]

    if task.dataset not in DEFAULT_ROOTS:
        raise ValueError(
            f"No default root for {task.dataset}"
        )

    data_root = (
        args.data_root
        or DEFAULT_ROOTS[task.dataset]
    )

    anchors = load_anchor_manifest(
        args.anchors
    )

    if args.limit_configs is not None:
        if args.limit_configs <= 0:
            raise ValueError(
                "limit-configs must be positive"
            )

        anchors = anchors[
            :args.limit_configs
        ]

    if not anchors:
        raise RuntimeError(
            "No anchor configurations selected"
        )

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
        dataset_name=task.dataset,
        root=data_root,
        image_size=84,
        split_seed=args.split_seed,
        download=False,
    )

    dataset = select_dataset(
        splits,
        task.split,
    )

    sampler = EpisodeSampler(
        dataset=dataset,
        allowed_classes=task.class_ids,
        n_way=task.n_way,
        n_shot=task.n_shot,
        n_query=task.n_query,
    )

    if args.overwrite:
        existing_table = pd.DataFrame()
    elif args.output.exists():
        existing_table = pd.read_csv(
            args.output
        )
    else:
        existing_table = pd.DataFrame()

    completed_config_ids = (
        set(existing_table["config_id"])
        if not existing_table.empty
        else set()
    )

    rows = (
        existing_table.to_dict(
            orient="records"
        )
        if not existing_table.empty
        else []
    )

    print("Task:", task.task_id)
    print("Dataset:", task.dataset)
    print("Split:", task.split)
    print("Class pool:", task.class_ids)
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Anchors selected:", len(anchors))
    print(
        "Already completed:",
        len(completed_config_ids),
    )
    print("Training episodes:", args.train_episodes)
    print(
        "Validation episodes:",
        args.validation_episodes,
    )
    print("Model seeds:", args.model_seeds)
    print()

    sweep_start = time.perf_counter()

    for index, config in enumerate(
        anchors
    ):
        if config.config_id in completed_config_ids:
            print(
                f"[{index + 1:3d}/{len(anchors):3d}] "
                f"{config.config_id}: SKIPPED"
            )
            continue

        print(
            f"[{index + 1:3d}/{len(anchors):3d}] "
            f"{config.config_id} | "
            f"{config.optimizer} | "
            f"lr={config.learning_rate:.7g} | "
            f"wd={config.weight_decay:.7g} | "
            f"dim={config.embedding_dim} | "
            f"dropout={config.dropout} | "
            f"scheduler={config.scheduler}"
        )

        result = evaluate_anchor_config(
            task=task,
            sampler=sampler,
            config=config,
            device=device,
            model_seeds=args.model_seeds,
            train_episodes=args.train_episodes,
            validation_episodes=(
                args.validation_episodes
            ),
        )

        rows.append(result)

        result_table = pd.DataFrame(
            rows
        )

        atomic_write_csv(
            result_table,
            args.output,
        )

        print(
            "    validation="
            f"{100 * result['validation_accuracy_mean']:.2f}% | "
            "time="
            f"{result['total_elapsed_seconds']:.2f}s"
        )

    result_table = pd.DataFrame(rows)

    if result_table.empty:
        raise RuntimeError(
            "Oracle sweep produced no results"
        )

    duplicate_mask = result_table[
        "config_id"
    ].duplicated()

    if duplicate_mask.any():
        duplicate_ids = result_table.loc[
            duplicate_mask,
            "config_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate result rows: {duplicate_ids}"
        )

    result_table = result_table.sort_values(
        by=[
            "validation_accuracy_mean",
            "total_elapsed_seconds",
            "config_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    result_table["validation_rank"] = (
        np.arange(len(result_table)) + 1
    )

    atomic_write_csv(
        result_table,
        args.output,
    )

    best_row = result_table.iloc[0]

    best_payload = {
        "schema_version": 1,
        "oracle_type": (
            "discrete_empirical_validation_oracle"
        ),
        "task_id": task.task_id,
        "dataset": task.dataset,
        "split": task.split,
        "best_config_id": str(
            best_row["config_id"]
        ),
        "validation_accuracy_mean": float(
            best_row[
                "validation_accuracy_mean"
            ]
        ),
        "validation_accuracy_seed_std": float(
            best_row[
                "validation_accuracy_seed_std"
            ]
        ),
        "configuration": {
            "learning_rate": float(
                best_row["learning_rate"]
            ),
            "weight_decay": float(
                best_row["weight_decay"]
            ),
            "optimizer": str(
                best_row["optimizer"]
            ),
            "embedding_dim": int(
                best_row["embedding_dim"]
            ),
            "dropout": float(
                best_row["dropout"]
            ),
            "scheduler": str(
                best_row["scheduler"]
            ),
        },
        "training_episodes": (
            args.train_episodes
        ),
        "validation_episodes": (
            args.validation_episodes
        ),
        "model_seeds": (
            args.model_seeds
        ),
        "evaluated_configuration_count": (
            len(result_table)
        ),
        "task_manifest": str(
            args.task_manifest
        ),
        "task_manifest_sha256": file_sha256(
            args.task_manifest
        ),
        "anchor_manifest": str(
            args.anchors
        ),
        "anchor_manifest_sha256": file_sha256(
            args.anchors
        ),
        "result_table": str(
            args.output
        ),
        "result_table_sha256": file_sha256(
            args.output
        ),
        "elapsed_seconds": float(
            time.perf_counter()
            - sweep_start
        ),
        "test_split_used": False,
    }

    args.best_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.best_output.write_text(
        json.dumps(
            best_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Top five configurations:")
    print(
        result_table[
            [
                "validation_rank",
                "config_id",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "embedding_dim",
                "dropout",
                "scheduler",
                "validation_accuracy_mean",
            ]
        ].head(5).to_string(
            index=False
        )
    )

    print()
    print("Result table:", args.output)
    print("Best configuration:", args.best_output)
    print(
        "Best validation accuracy:",
        f"{100 * best_payload['validation_accuracy_mean']:.2f}%",
    )
    print("Test split used: False")
    print("ORACLE SWEEP: PASS")


if __name__ == "__main__":
    main()
