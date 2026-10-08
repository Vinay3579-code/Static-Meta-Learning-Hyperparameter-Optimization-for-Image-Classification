from __future__ import annotations

import argparse
import fcntl
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
from sml_hpo.models.protonet import (
    ProtoNet,
)
from sml_hpo.oracle import (
    evaluator_v2,
)
from sml_hpo.oracle.search_space_v2 import (
    load_anchor_manifest_v2,
)
from sml_hpo.tasks.spec import (
    load_task_manifest,
)


DEFAULT_ROOTS = {
    "omniglot":
        Path("data/raw/omniglot"),

    "cifar100":
        Path("data/raw/cifar100"),

    "miniimagenet":
        Path("data/raw/miniimagenet"),

    "dtd":
        Path("data/raw/dtd"),

    "flowers102":
        Path("data/raw/flowers102"),
}


METADATA_COLUMNS = {
    "task_id",
    "dataset",
    "regime",
    "n_way",
    "n_shot",
    "n_query",
}


MODEL_SEEDS = [
    101,
    202,
    303,
]

TRAIN_EPISODES = 200


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recommendations",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-labels",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--anchors",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--task-id",
        type=str,
    )

    parser.add_argument(
        "--limit-tasks",
        type=int,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def verify_sha_sidecar(
    target: Path,
) -> None:
    sidecar = target.with_suffix(
        ".sha256"
    )

    if not sidecar.exists():
        raise RuntimeError(
            f"Missing SHA sidecar: "
            f"{sidecar}"
        )

    text = (
        sidecar.read_text(
            encoding="utf-8"
        )
        .strip()
    )

    expected = (
        text.split()[0]
    )

    actual = sha256_file(
        target
    )

    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for "
            f"{target}"
        )

    print(
        f"Hash verified: {target.name}"
    )


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


def atomic_write_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def parse_json_list(
    value,
) -> list[str]:
    result = json.loads(
        str(value)
    )

    if not isinstance(
        result,
        list,
    ):
        raise RuntimeError(
            "Expected JSON list"
        )

    return [
        str(item)
        for item in result
    ]


def evaluate_config_on_test(
    *,
    task,
    sampler,
    config,
    device,
) -> dict:

    train_seed_bank = (
        task.training_seeds()[
            :TRAIN_EPISODES
        ]
    )

    test_seed_bank = (
        task.testing_seeds()
    )

    if (
        len(train_seed_bank)
        != TRAIN_EPISODES
    ):
        raise RuntimeError(
            f"{task.task_id}: "
            "insufficient training seeds"
        )

    if (
        len(test_seed_bank)
        != task.test_episodes
    ):
        raise RuntimeError(
            f"{task.task_id}: "
            "test seed-bank mismatch"
        )

    per_seed_results = []

    configuration_start = (
        time.perf_counter()
    )

    for model_seed in (
        MODEL_SEEDS
    ):

        evaluator_v2.seed_everything(
            int(model_seed),
            deterministic=True,
        )

        model = ProtoNet(
            input_channels=3,

            hidden_channels=(
                config.hidden_channels
            ),

            embedding_dim=(
                config.embedding_dim
            ),

            dropout=(
                config.dropout
            ),

            distance_metric=(
                config.distance_metric
            ),

            temperature=(
                config.temperature
            ),
        ).to(device)

        optimizer = (
            evaluator_v2
            .build_optimizer_v2(
                model,
                config,
            )
        )

        scheduler = (
            evaluator_v2
            .build_scheduler_v2(
                optimizer,
                config,
                TRAIN_EPISODES,
            )
        )

        train_losses = []
        train_accuracies = []

        seed_start = (
            time.perf_counter()
        )

        for episode_seed in (
            train_seed_bank
        ):
            (
                loss,
                accuracy,
            ) = (
                evaluator_v2
                .train_episode_v2(
                    model=model,

                    optimizer=optimizer,

                    sampler=sampler,

                    episode_seed=(
                        episode_seed
                    ),

                    task=task,

                    config=config,

                    device=device,
                )
            )

            train_losses.append(
                loss
            )

            train_accuracies.append(
                accuracy
            )

            if (
                scheduler
                is not None
            ):
                scheduler.step()

        test_result = (
            evaluator_v2
            .evaluate_episode_bank(
                model=model,

                sampler=sampler,

                episode_seeds=(
                    test_seed_bank
                ),

                n_way=(
                    task.n_way
                ),

                device=device,
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        per_seed_results.append(
            {
                "model_seed":
                    int(
                        model_seed
                    ),

                "test_accuracy":
                    float(
                        test_result[
                            "mean_accuracy"
                        ]
                    ),

                "test_episode_std":
                    float(
                        test_result[
                            "std_accuracy"
                        ]
                    ),

                "final_train_loss":
                    float(
                        train_losses[-1]
                    ),

                "mean_train_accuracy":
                    float(
                        np.mean(
                            train_accuracies
                        )
                    ),

                "elapsed_seconds":
                    float(
                        time.perf_counter()
                        - seed_start
                    ),
            }
        )

        del model
        del optimizer
        del scheduler

        if device.type == "cuda":
            torch.cuda.empty_cache()

    accuracies = np.asarray(
        [
            row["test_accuracy"]
            for row
            in per_seed_results
        ],
        dtype=np.float64,
    )

    episode_stds = np.asarray(
        [
            row[
                "test_episode_std"
            ]
            for row
            in per_seed_results
        ],
        dtype=np.float64,
    )

    elapsed = np.asarray(
        [
            row[
                "elapsed_seconds"
            ]
            for row
            in per_seed_results
        ],
        dtype=np.float64,
    )

    return {
        "task_id":
            task.task_id,

        "dataset":
            task.dataset,

        "regime":
            (
                f"{task.n_way}w"
                f"{task.n_shot}s"
            ),

        "config_id":
            config.config_id,

        "training_episodes":
            TRAIN_EPISODES,

        "test_episodes":
            len(
                test_seed_bank
            ),

        "model_seed_count":
            len(
                MODEL_SEEDS
            ),

        "model_seeds_json":
            json.dumps(
                MODEL_SEEDS
            ),

        "test_accuracies_by_seed_json":
            json.dumps(
                accuracies.tolist()
            ),

        "test_accuracy_mean":
            float(
                accuracies.mean()
            ),

        "test_accuracy_seed_std":
            float(
                accuracies.std(
                    ddof=1
                )
                if len(
                    accuracies
                ) > 1
                else 0.0
            ),

        "test_episode_std_mean":
            float(
                episode_stds.mean()
            ),

        "mean_seed_elapsed_seconds":
            float(
                elapsed.mean()
            ),

        "total_elapsed_seconds":
            float(
                time.perf_counter()
                - configuration_start
            ),
    }


def main():
    args = parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Prevent accidental duplicate runners.
    lock_path = (
        args.output_root
        / ".final_test_evaluation.lock"
    )

    lock_handle = lock_path.open(
        "w"
    )

    try:
        fcntl.flock(
            lock_handle,
            (
                fcntl.LOCK_EX
                | fcntl.LOCK_NB
            ),
        )
    except BlockingIOError:
        raise RuntimeError(
            "Another final test evaluator "
            "is already running."
        )

    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable"
            )

        torch.cuda.set_device(
            device.index or 0
        )

    verify_sha_sidecar(
        args.recommendations
    )

    lock_manifest = (
        args.recommendations.parent
        / "recommendation_lock_manifest.json"
    )

    verify_sha_sidecar(
        lock_manifest
    )

    recommendations = (
        pd.read_csv(
            args.recommendations
        )
    )

    labels = pd.read_csv(
        args.test_labels
    )

    if len(
        recommendations
    ) != 120:
        raise RuntimeError(
            "Expected 120 locked "
            "recommendations"
        )

    if len(labels) != 120:
        raise RuntimeError(
            "Expected 120 test labels"
        )

    if recommendations[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate recommendation "
            "task IDs"
        )

    if labels[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate test-label task IDs"
        )

    if set(
        recommendations[
            "task_id"
        ]
    ) != set(
        labels[
            "task_id"
        ]
    ):
        raise RuntimeError(
            "Recommendation/test-label "
            "task mismatch"
        )

    method_columns = [
        column
        for column
        in recommendations.columns
        if column
        not in METADATA_COLUMNS
    ]

    print(
        "Locked methods:",
        len(method_columns),
    )

    if len(
        method_columns
    ) != 20:
        raise RuntimeError(
            f"Expected 20 locked methods, "
            f"found {len(method_columns)}"
        )

    anchors = (
        load_anchor_manifest_v2(
            args.anchors
        )
    )

    config_map = {
        config.config_id:
            config
        for config in anchors
    }

    if len(config_map) != 64:
        raise RuntimeError(
            "Expected 64 anchors"
        )

    label_map = (
        labels.set_index(
            "task_id"
        )
    )

    recommendations = (
        recommendations
        .sort_values(
            "task_id"
        )
        .reset_index(
            drop=True
        )
    )

    if args.task_id:
        recommendations = (
            recommendations[
                recommendations[
                    "task_id"
                ]
                == args.task_id
            ]
            .reset_index(
                drop=True
            )
        )

        if len(
            recommendations
        ) != 1:
            raise RuntimeError(
                "Could not resolve "
                f"task {args.task_id}"
            )

    if (
        args.limit_tasks
        is not None
    ):
        recommendations = (
            recommendations.iloc[
                :args.limit_tasks
            ].copy()
        )

    dataset_cache = {}
    manifest_cache = {}

    completed_tasks = 0

    for task_position, row in (
        recommendations.iterrows()
    ):

        task_id = str(
            row["task_id"]
        )

        dataset_name = str(
            row["dataset"]
        )

        regime = str(
            row["regime"]
        )

        print()
        print("=" * 80)
        print(
            f"[{task_position + 1}/"
            f"{len(recommendations)}] "
            f"{task_id}"
        )
        print("=" * 80)

        manifest_path = (
            args.manifest_root
            / "test"
            / (
                f"{dataset_name}_"
                f"{regime}.json"
            )
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                manifest_path
            )

        if manifest_path not in (
            manifest_cache
        ):
            manifest_cache[
                manifest_path
            ] = (
                load_task_manifest(
                    manifest_path
                )
            )

        matching_tasks = [
            task
            for task in (
                manifest_cache[
                    manifest_path
                ]
            )
            if task.task_id
            == task_id
        ]

        if len(
            matching_tasks
        ) != 1:
            raise RuntimeError(
                f"Could not uniquely "
                f"resolve {task_id}"
            )

        task = matching_tasks[0]

        if task.split != "test":
            raise RuntimeError(
                "Final evaluator received "
                "non-test task"
            )

        if dataset_name not in (
            dataset_cache
        ):
            splits = (
                load_dataset_splits(
                    dataset_name=(
                        dataset_name
                    ),

                    root=(
                        DEFAULT_ROOTS[
                            dataset_name
                        ]
                    ),

                    image_size=84,

                    split_seed=(
                        args.split_seed
                    ),

                    download=False,
                )
            )

            dataset_cache[
                dataset_name
            ] = (
                splits.test_dataset
            )

        dataset = dataset_cache[
            dataset_name
        ]

        sampler = EpisodeSampler(
            dataset=dataset,

            allowed_classes=(
                task.class_ids
            ),

            n_way=(
                task.n_way
            ),

            n_shot=(
                task.n_shot
            ),

            n_query=(
                task.n_query
            ),
        )

        label_row = (
            label_map.loc[
                task_id
            ]
        )

        validation_oracle_config = (
            str(
                label_row[
                    "best_config_id"
                ]
            )
        )

        equivalent_ids = set(
            parse_json_list(
                label_row[
                    "oracle_equivalent_config_ids_json"
                ]
            )
        )

        method_to_config = {
            method:
                str(
                    row[method]
                )
            for method
            in method_columns
        }

        method_to_config[
            "validation_selected_oracle"
        ] = (
            validation_oracle_config
        )

        unknown = (
            set(
                method_to_config.values()
            )
            - set(
                config_map
            )
        )

        if unknown:
            raise RuntimeError(
                f"Unknown config IDs: "
                f"{sorted(unknown)}"
            )

        unique_configs = sorted(
            set(
                method_to_config.values()
            )
        )

        task_dir = (
            args.output_root
            / dataset_name
            / task_id
        )

        task_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_results_path = (
            task_dir
            / "config_test_results.csv"
        )

        if (
            config_results_path.exists()
            and not args.overwrite
        ):
            config_results = (
                pd.read_csv(
                    config_results_path
                )
            )
        else:
            config_results = (
                pd.DataFrame()
            )

        if (
            not config_results.empty
            and config_results[
                "config_id"
            ].duplicated().any()
        ):
            raise RuntimeError(
                f"{task_id}: duplicate "
                "config result rows"
            )

        completed_configs = (
            set(
                config_results[
                    "config_id"
                ].astype(str)
            )
            if not config_results.empty
            else set()
        )

        records = (
            config_results
            .to_dict(
                orient="records"
            )
            if not config_results.empty
            else []
        )

        print(
            "Unique required configs:",
            len(unique_configs),
        )

        print(
            "Already completed:",
            len(
                completed_configs
            ),
        )

        for config_index, config_id in enumerate(
            unique_configs,
            1,
        ):
            if (
                config_id
                in completed_configs
            ):
                print(
                    f"  [{config_index}/"
                    f"{len(unique_configs)}] "
                    f"{config_id}: SKIPPED"
                )

                continue

            print(
                f"  [{config_index}/"
                f"{len(unique_configs)}] "
                f"{config_id}: evaluating "
                f"on TEST bank"
            )

            result = (
                evaluate_config_on_test(
                    task=task,

                    sampler=sampler,

                    config=(
                        config_map[
                            config_id
                        ]
                    ),

                    device=device,
                )
            )

            records.append(
                result
            )

            config_results = (
                pd.DataFrame(
                    records
                )
                .sort_values(
                    "config_id"
                )
                .reset_index(
                    drop=True
                )
            )

            atomic_write_csv(
                config_results,
                config_results_path,
            )

            print(
                "      test="
                f"{100 * result['test_accuracy_mean']:.2f}% "
                "| time="
                f"{result['total_elapsed_seconds']:.1f}s"
            )

        config_results = (
            pd.read_csv(
                config_results_path
            )
        )

        available = set(
            config_results[
                "config_id"
            ].astype(str)
        )

        if not set(
            unique_configs
        ).issubset(
            available
        ):
            raise RuntimeError(
                f"{task_id}: incomplete "
                "config evaluation"
            )

        config_results = (
            config_results.set_index(
                "config_id"
            )
        )

        reference_accuracy = float(
            config_results.loc[
                validation_oracle_config,
                "test_accuracy_mean",
            ]
        )

        method_records = []

        for (
            method,
            config_id,
        ) in (
            method_to_config.items()
        ):

            config_result = (
                config_results.loc[
                    config_id
                ]
            )

            test_accuracy = float(
                config_result[
                    "test_accuracy_mean"
                ]
            )

            is_reference = (
                method
                == "validation_selected_oracle"
            )

            hard_agreement = (
                config_id
                == validation_oracle_config
            )

            equivalent = (
                config_id
                in equivalent_ids
            )

            method_records.append(
                {
                    "task_id":
                        task_id,

                    "dataset":
                        dataset_name,

                    "regime":
                        regime,

                    "method":
                        method,

                    "config_id":
                        config_id,

                    "validation_oracle_config_id":
                        validation_oracle_config,

                    "hard_oracle_agreement":
                        bool(
                            hard_agreement
                        ),

                    "validation_oracle_equivalent":
                        bool(
                            equivalent
                        ),

                    "test_accuracy_mean":
                        test_accuracy,

                    "test_accuracy_seed_std":
                        float(
                            config_result[
                                "test_accuracy_seed_std"
                            ]
                        ),

                    "validation_oracle_test_accuracy":
                        reference_accuracy,

                    # Signed gap.
                    # Negative means the method
                    # outperformed the validation-
                    # selected oracle on test.
                    "test_gap_to_validation_oracle_pp":
                        float(
                            100.0
                            * (
                                reference_accuracy
                                - test_accuracy
                            )
                        ),

                    "is_validation_oracle_reference":
                        bool(
                            is_reference
                        ),
                }
            )

        method_table = (
            pd.DataFrame(
                method_records
            )
        )

        method_path = (
            task_dir
            / "method_test_results.csv"
        )

        atomic_write_csv(
            method_table,
            method_path,
        )

        complete_payload = {
            "schema_version": 1,

            "task_id":
                task_id,

            "dataset":
                dataset_name,

            "regime":
                regime,

            "training_episodes":
                TRAIN_EPISODES,

            "test_episodes":
                int(
                    task.test_episodes
                ),

            "model_seeds":
                MODEL_SEEDS,

            "locked_method_count":
                len(
                    method_columns
                ),

            "unique_evaluated_config_count":
                len(
                    unique_configs
                ),

            "validation_oracle_config_id":
                validation_oracle_config,

            "recommendation_file_sha256":
                sha256_file(
                    args.recommendations
                ),

            "config_results_sha256":
                sha256_file(
                    config_results_path
                ),

            "method_results_sha256":
                sha256_file(
                    method_path
                ),
        }

        atomic_write_json(
            complete_payload,
            task_dir
            / "COMPLETE.json",
        )

        completed_tasks += 1

        print(
            f"Task complete: "
            f"{completed_tasks}/"
            f"{len(recommendations)}"
        )

    # Merge all completed task tables.
    method_files = sorted(
        args.output_root.glob(
            "*/*/method_test_results.csv"
        )
    )

    if method_files:
        merged = pd.concat(
            [
                pd.read_csv(
                    path
                )
                for path
                in method_files
            ],
            ignore_index=True,
        )

        atomic_write_csv(
            merged,
            args.output_root
            / "all_method_test_results.csv",
        )

    complete_markers = list(
        args.output_root.glob(
            "*/*/COMPLETE.json"
        )
    )

    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION STATUS")
    print("=" * 80)

    print(
        "Completed task markers:",
        len(
            complete_markers
        ),
        "/ 120",
    )

    if (
        args.task_id is None
        and args.limit_tasks is None
        and len(
            complete_markers
        ) == 120
    ):
        print()
        print(
            "FINAL TEST BANK EVALUATION: PASS"
        )


if __name__ == "__main__":
    main()
