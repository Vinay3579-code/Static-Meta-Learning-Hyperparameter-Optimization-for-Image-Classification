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

from sml_hpo.data.registry import (
    load_dataset_splits,
)
from sml_hpo.episodes.sampler import (
    EpisodeSampler,
)
from sml_hpo.oracle.evaluator_v2 import (
    evaluate_anchor_config_v2,
)
from sml_hpo.oracle.search_space_v2 import (
    load_anchor_manifest_v2,
)
from sml_hpo.tasks.spec import (
    load_task_manifest,
)


DEFAULT_ROOTS = {
    "omniglot": Path("data/raw/omniglot"),
    "cifar100": Path("data/raw/cifar100"),
    "miniimagenet": Path("data/raw/miniimagenet"),
    "dtd": Path("data/raw/dtd"),
    "flowers102": Path("data/raw/flowers102"),
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
        default=100,
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[101, 202],
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


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def select_dataset(
    splits,
    split_name: str,
):
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

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    table.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def main() -> None:
    args = parse_args()

    tasks = load_task_manifest(
        args.task_manifest
    )

    if args.task_id is None:
        task = tasks[0]
    else:
        matching = [
            task
            for task in tasks
            if task.task_id
            == args.task_id
        ]

        if len(matching) != 1:
            raise ValueError(
                "Could not uniquely resolve "
                f"{args.task_id!r}"
            )

        task = matching[0]

    anchors = load_anchor_manifest_v2(
        args.anchors
    )

    if args.limit_configs is not None:
        anchors = anchors[
            :args.limit_configs
        ]

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

    data_root = (
        args.data_root
        or DEFAULT_ROOTS[
            task.dataset
        ]
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
        allowed_classes=(
            task.class_ids
        ),
        n_way=task.n_way,
        n_shot=task.n_shot,
        n_query=task.n_query,
    )

    if (
        args.output.exists()
        and not args.overwrite
    ):
        existing = pd.read_csv(
            args.output
        )
    else:
        existing = pd.DataFrame()

    rows = (
        existing.to_dict(
            orient="records"
        )
        if not existing.empty
        else []
    )

    completed = (
        set(existing["config_id"])
        if not existing.empty
        else set()
    )

    print("Task:", task.task_id)
    print("Dataset:", task.dataset)
    print(
        "Regime:",
        f"{task.n_way}-way "
        f"{task.n_shot}-shot",
    )
    print("Device:", device)
    print("Anchors:", len(anchors))
    print("Completed:", len(completed))
    print("Model seeds:", args.model_seeds)
    print()

    run_start = time.perf_counter()

    for index, config in enumerate(
        anchors
    ):
        if config.config_id in completed:
            print(
                f"[{index + 1:3d}/"
                f"{len(anchors):3d}] "
                f"{config.config_id}: SKIPPED"
            )
            continue

        print(
            f"[{index + 1:3d}/"
            f"{len(anchors):3d}] "
            f"{config.config_id} | "
            f"{config.optimizer} | "
            f"width={config.hidden_channels} | "
            f"dim={config.embedding_dim} | "
            f"metric={config.distance_metric} | "
            f"temp={config.temperature:.4g} | "
            f"lr={config.learning_rate:.4g}"
        )

        result = (
            evaluate_anchor_config_v2(
                task=task,
                sampler=sampler,
                config=config,
                device=device,
                model_seeds=(
                    args.model_seeds
                ),
                train_episodes=(
                    args.train_episodes
                ),
                validation_episodes=(
                    args.validation_episodes
                ),
            )
        )

        rows.append(result)

        atomic_write_csv(
            pd.DataFrame(rows),
            args.output,
        )

        print(
            "    validation="
            f"{100 * result['validation_accuracy_mean']:.2f}% | "
            "time="
            f"{result['total_elapsed_seconds']:.2f}s"
        )

    table = pd.DataFrame(rows)

    if table["config_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate result rows"
        )

    table = table.sort_values(
        by=[
            "validation_accuracy_mean",
            "validation_accuracy_seed_std",
            "total_elapsed_seconds",
            "config_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    table["validation_rank"] = (
        np.arange(len(table)) + 1
    )

    atomic_write_csv(
        table,
        args.output,
    )

    best = table.iloc[0]

    configuration_fields = [
        "learning_rate",
        "weight_decay",
        "optimizer",
        "hidden_channels",
        "embedding_dim",
        "dropout",
        "scheduler",
        "distance_metric",
        "temperature",
        "label_smoothing",
    ]

    configuration = {
        field: (
            best[field].item()
            if hasattr(
                best[field],
                "item",
            )
            else best[field]
        )
        for field in configuration_fields
    }

    payload = {
        "schema_version": 2,
        "oracle_type": (
            "discrete_empirical_validation_oracle"
        ),
        "task_id": task.task_id,
        "dataset": task.dataset,
        "split": task.split,
        "n_way": task.n_way,
        "n_shot": task.n_shot,
        "n_query": task.n_query,
        "best_config_id": str(
            best["config_id"]
        ),
        "validation_accuracy_mean": float(
            best[
                "validation_accuracy_mean"
            ]
        ),
        "validation_accuracy_seed_std": float(
            best[
                "validation_accuracy_seed_std"
            ]
        ),
        "configuration": configuration,
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
            len(table)
        ),
        "task_manifest": str(
            args.task_manifest
        ),
        "task_manifest_sha256": (
            file_sha256(
                args.task_manifest
            )
        ),
        "anchor_manifest": str(
            args.anchors
        ),
        "anchor_manifest_sha256": (
            file_sha256(
                args.anchors
            )
        ),
        "result_table": str(
            args.output
        ),
        "result_table_sha256": (
            file_sha256(
                args.output
            )
        ),
        "elapsed_seconds": float(
            time.perf_counter()
            - run_start
        ),
        "test_split_used": False,
    }

    args.best_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.best_output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print()
    print("Top five configurations:")
    print(
        table[
            [
                "validation_rank",
                "config_id",
                "optimizer",
                "hidden_channels",
                "embedding_dim",
                "distance_metric",
                "temperature",
                "validation_accuracy_mean",
            ]
        ].head(5).to_string(
            index=False
        )
    )

    print()
    print("Result table:", args.output)
    print(
        "Best configuration:",
        args.best_output,
    )
    print("Test split used: False")
    print("ORACLE SWEEP V2: PASS")


if __name__ == "__main__":
    main()
