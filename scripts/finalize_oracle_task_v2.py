from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.tasks.spec import (
    load_task_manifest,
)


EXPECTED_SCREENING_CONFIGS = 64
EXPECTED_ROBUST_SEEDS = (
    101,
    202,
    303,
)


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
        required=True,
    )
    parser.add_argument(
        "--stage1-results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.002,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def parse_vector(
    value: object,
    expected_seeds: tuple[int, ...],
    seed_value: object,
) -> np.ndarray:
    seeds = tuple(
        int(seed)
        for seed in json.loads(
            str(seed_value)
        )
    )

    if seeds != expected_seeds:
        raise RuntimeError(
            f"Expected seeds {expected_seeds}, "
            f"received {seeds}"
        )

    vector = np.asarray(
        json.loads(str(value)),
        dtype=np.float64,
    )

    if vector.shape != (
        len(expected_seeds),
    ):
        raise ValueError(
            f"Expected vector length "
            f"{len(expected_seeds)}, "
            f"received {vector.shape}"
        )

    if not np.isfinite(vector).all():
        raise FloatingPointError(
            "Non-finite validation scores"
        )

    return vector


def atomic_write_json(
    value: dict,
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
            value,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
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


def main() -> None:
    args = parse_args()

    if args.epsilon < 0:
        raise ValueError(
            "epsilon must be non-negative"
        )

    tasks = load_task_manifest(
        args.task_manifest
    )

    matching = [
        task
        for task in tasks
        if task.task_id == args.task_id
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve "
            f"{args.task_id!r}"
        )

    task = matching[0]

    stage1 = pd.read_csv(
        args.stage1_results
    )

    stage2 = pd.read_csv(
        args.stage2_results
    )

    if len(stage1) != (
        EXPECTED_SCREENING_CONFIGS
    ):
        raise RuntimeError(
            f"Expected 64 screening rows, "
            f"found {len(stage1)}"
        )

    if stage1[
        "config_id"
    ].nunique() != 64:
        raise RuntimeError(
            "Screening configuration IDs "
            "are not unique"
        )

    candidate_payload = json.loads(
        args.candidate_manifest.read_text(
            encoding="utf-8"
        )
    )

    candidate_configs = (
        candidate_payload[
            "configurations"
        ]
    )

    candidate_ids = [
        str(config["config_id"])
        for config in candidate_configs
    ]

    if set(
        stage2["config_id"]
    ) != set(candidate_ids):
        raise RuntimeError(
            "Stage-two results do not match "
            "the candidate manifest"
        )

    if stage2[
        "config_id"
    ].nunique() != len(
        candidate_ids
    ):
        raise RuntimeError(
            "Duplicate stage-two "
            "configuration IDs"
        )

    stage1 = (
        stage1.set_index(
            "config_id"
        )
        .sort_index()
    )

    stage2 = (
        stage2.set_index(
            "config_id"
        )
        .sort_index()
    )

    stage1_ranking = (
        stage1.reset_index()
        .sort_values(
            by=[
                "validation_accuracy_mean",
                "config_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    stage1_rank_map = {
        str(config_id): rank
        for rank, config_id
        in enumerate(
            stage1_ranking[
                "config_id"
            ],
            start=1,
        )
    }

    config_map = {
        str(config["config_id"]): config
        for config in candidate_configs
    }

    rows = []

    for config_id in candidate_ids:
        seed101 = parse_vector(
            stage1.loc[
                config_id,
                "validation_accuracies_"
                "by_seed_json",
            ],
            expected_seeds=(101,),
            seed_value=stage1.loc[
                config_id,
                "model_seeds_json",
            ],
        )

        seed202_303 = parse_vector(
            stage2.loc[
                config_id,
                "validation_accuracies_"
                "by_seed_json",
            ],
            expected_seeds=(
                202,
                303,
            ),
            seed_value=stage2.loc[
                config_id,
                "model_seeds_json",
            ],
        )

        robust_values = np.asarray(
            [
                float(seed101[0]),
                float(seed202_303[0]),
                float(seed202_303[1]),
            ],
            dtype=np.float64,
        )

        config = config_map[
            config_id
        ]

        row = {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "split": task.split,
            "n_way": task.n_way,
            "n_shot": task.n_shot,
            "n_query": task.n_query,
            "config_id": config_id,
            "screening_rank": (
                stage1_rank_map[
                    config_id
                ]
            ),
            "seed_101_accuracy": float(
                robust_values[0]
            ),
            "seed_202_accuracy": float(
                robust_values[1]
            ),
            "seed_303_accuracy": float(
                robust_values[2]
            ),
            "robust_validation_accuracy_mean": float(
                robust_values.mean()
            ),
            "robust_validation_accuracy_std": float(
                robust_values.std(
                    ddof=1
                )
            ),
            "robust_validation_accuracy_min": float(
                robust_values.min()
            ),
            "robust_validation_accuracy_max": float(
                robust_values.max()
            ),
        }

        for field in (
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
        ):
            row[field] = config[field]

        rows.append(row)

    scores = pd.DataFrame(rows)

    scores = scores.sort_values(
        by=[
            "robust_validation_accuracy_mean",
            "robust_validation_accuracy_std",
            "config_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    scores["robust_rank"] = (
        np.arange(len(scores)) + 1
    )

    best_score = float(
        scores.iloc[0][
            "robust_validation_accuracy_mean"
        ]
    )

    scores["robust_regret"] = (
        best_score
        - scores[
            "robust_validation_accuracy_mean"
        ]
    )

    scores[
        "oracle_equivalent"
    ] = (
        scores["robust_regret"]
        <= args.epsilon + 1e-12
    )

    best_row = scores.iloc[0]

    equivalent_ids = (
        scores.loc[
            scores[
                "oracle_equivalent"
            ],
            "config_id",
        ]
        .astype(str)
        .tolist()
    )

    atomic_write_csv(
        scores,
        args.scores_output,
    )

    label = {
        "schema_version": 1,
        "oracle_type": (
            "two_stage_three_seed_"
            "candidate_oracle_v2"
        ),
        "task_id": task.task_id,
        "dataset": task.dataset,
        "split": task.split,
        "class_ids": list(
            task.class_ids
        ),
        "task_seed": int(
            task.task_seed
        ),
        "n_way": int(
            task.n_way
        ),
        "n_shot": int(
            task.n_shot
        ),
        "n_query": int(
            task.n_query
        ),
        "training_episodes": 200,
        "validation_episodes": 100,
        "screening_configuration_count": (
            64
        ),
        "candidate_count": int(
            len(scores)
        ),
        "screening_seed": 101,
        "confirmation_seeds": [
            202,
            303,
        ],
        "robust_seed_set": list(
            EXPECTED_ROBUST_SEEDS
        ),
        "best_config_id": str(
            best_row["config_id"]
        ),
        "best_robust_validation_accuracy": float(
            best_row[
                "robust_validation_accuracy_mean"
            ]
        ),
        "best_robust_validation_seed_std": float(
            best_row[
                "robust_validation_accuracy_std"
            ]
        ),
        "oracle_equivalence_epsilon": float(
            args.epsilon
        ),
        "oracle_equivalent_config_ids": (
            equivalent_ids
        ),
        "oracle_equivalent_count": int(
            len(equivalent_ids)
        ),
        "candidate_ids": (
            scores["config_id"]
            .astype(str)
            .tolist()
        ),
        "screening_best_config_id": str(
            stage1_ranking.iloc[0][
                "config_id"
            ]
        ),
        "screening_best_validation_accuracy": float(
            stage1_ranking.iloc[0][
                "validation_accuracy_mean"
            ]
        ),
        "test_episode_bank_used": False,
        "task_manifest": str(
            args.task_manifest
        ),
        "task_manifest_sha256": (
            sha256(
                args.task_manifest
            )
        ),
        "stage1_results": str(
            args.stage1_results
        ),
        "stage1_results_sha256": (
            sha256(
                args.stage1_results
            )
        ),
        "candidate_manifest": str(
            args.candidate_manifest
        ),
        "candidate_manifest_sha256": (
            sha256(
                args.candidate_manifest
            )
        ),
        "stage2_results": str(
            args.stage2_results
        ),
        "stage2_results_sha256": (
            sha256(
                args.stage2_results
            )
        ),
        "candidate_scores": str(
            args.scores_output
        ),
        "candidate_scores_sha256": (
            sha256(
                args.scores_output
            )
        ),
    }

    atomic_write_json(
        label,
        args.label_output,
    )

    print("Task:", task.task_id)
    print(
        "Candidate count:",
        len(scores),
    )
    print(
        "Best robust configuration:",
        label["best_config_id"],
    )
    print(
        "Best robust accuracy:",
        label[
            "best_robust_validation_accuracy"
        ],
    )
    print(
        "Equivalent configurations:",
        equivalent_ids,
    )
    print(
        "Test episode bank used:",
        False,
    )
    print(
        "Scores:",
        args.scores_output,
    )
    print(
        "Label:",
        args.label_output,
    )
    print("ROBUST ORACLE FINALIZATION: PASS")


if __name__ == "__main__":
    main()
