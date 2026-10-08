from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.tasks.spec import (
    TaskSpec,
    load_task_manifest,
)


EXPECTED_TASKS = 20
EXPECTED_CONFIGS = 64
EXPECTED_SEEDS = (101, 202)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pilot-manifests",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--screening-selection",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
    )

    return parser.parse_args()


def task_fingerprint(
    task: TaskSpec,
) -> tuple[object, ...]:
    """
    Identify the actual task distribution rather than relying on task_id.

    Two tasks are considered identical only when their class pool,
    few-shot regime and complete episode seed banks are identical.
    """
    return (
        task.dataset,
        task.split,
        tuple(task.class_ids),
        int(task.task_seed),
        int(task.n_way),
        int(task.n_shot),
        int(task.n_query),
        int(task.train_episodes),
        int(task.validation_episodes),
        int(task.test_episodes),
        int(task.train_seed_base),
        int(task.validation_seed_base),
        int(task.test_seed_base),
    )


def load_named_task(
    manifest: Path,
    task_id: str,
) -> TaskSpec:
    matches = [
        task
        for task in load_task_manifest(
            manifest
        )
        if task.task_id == task_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve {task_id!r} "
            f"in {manifest}"
        )

    return matches[0]


def load_pilot_task_specs(
    manifests: list[Path],
) -> dict[str, TaskSpec]:
    task_specs: dict[str, TaskSpec] = {}

    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(
                manifest
            )

        for task in load_task_manifest(
            manifest
        ):
            if task.task_id in task_specs:
                raise RuntimeError(
                    "Duplicate pilot task ID across manifests: "
                    f"{task.task_id}"
                )

            task_specs[task.task_id] = task

    return task_specs


def parse_seed_vector(
    value: object,
    expected_length: int,
) -> np.ndarray:
    values = np.asarray(
        json.loads(str(value)),
        dtype=np.float64,
    )

    if values.shape != (
        expected_length,
    ):
        raise ValueError(
            f"Expected seed vector length "
            f"{expected_length}, received "
            f"{values.shape}"
        )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite seed accuracy"
        )

    return values


def rank_configs(
    scores: pd.Series,
) -> list[str]:
    ranking = pd.DataFrame(
        {
            "config_id": (
                scores.index.astype(str)
            ),
            "score": scores.to_numpy(
                dtype=np.float64
            ),
        }
    )

    ranking = ranking.sort_values(
        by=[
            "score",
            "config_id",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return ranking[
        "config_id"
    ].tolist()


def discover_primary_tables(
    root: Path,
) -> list[Path]:
    tables: list[Path] = []

    for path in sorted(
        root.glob("*.csv")
    ):
        if path.stem.endswith(
            "_ranked"
        ):
            continue

        table = pd.read_csv(path)

        required = {
            "task_id",
            "dataset",
            "config_id",
            "model_seeds_json",
            "validation_accuracies_by_seed_json",
        }

        if not required <= set(
            table.columns
        ):
            continue

        if (
            len(table) == EXPECTED_CONFIGS
            and table[
                "config_id"
            ].nunique()
            == EXPECTED_CONFIGS
            and table[
                "task_id"
            ].nunique()
            == 1
        ):
            tables.append(path)

    return tables


def main() -> None:
    args = parse_args()

    if args.epsilon < 0:
        raise ValueError(
            "epsilon must be non-negative"
        )

    k_values = sorted(
        set(args.k_values)
    )

    if not k_values or min(
        k_values
    ) <= 0:
        raise ValueError(
            "K values must be positive"
        )

    screening_payload = json.loads(
        args.screening_selection.read_text(
            encoding="utf-8"
        )
    )

    screening_specs = []

    for entry in screening_payload[
        "tasks"
    ]:
        screening_specs.append(
            load_named_task(
                manifest=Path(
                    entry["manifest"]
                ),
                task_id=str(
                    entry["task_id"]
                ),
            )
        )

    screening_ids = {
        task.task_id
        for task in screening_specs
    }

    screening_fingerprints = {
        task_fingerprint(task)
        for task in screening_specs
    }

    pilot_specs = (
        load_pilot_task_specs(
            args.pilot_manifests
        )
    )

    table_paths = (
        discover_primary_tables(
            args.pilot_root
        )
    )

    if len(table_paths) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} "
            f"primary pilot tables, found "
            f"{len(table_paths)}"
        )

    result_rows: list[
        dict[str, object]
    ] = []

    holdout_ids: set[str] = set()
    holdout_fingerprints: set[
        tuple[object, ...]
    ] = set()

    for path in table_paths:
        table = pd.read_csv(path)

        task_id = str(
            table.iloc[0][
                "task_id"
            ]
        )

        if task_id in holdout_ids:
            raise RuntimeError(
                f"Duplicate pilot result task: "
                f"{task_id}"
            )

        if task_id not in pilot_specs:
            raise RuntimeError(
                f"No pilot manifest task found "
                f"for result {task_id}"
            )

        pilot_task = pilot_specs[
            task_id
        ]

        holdout_ids.add(task_id)
        holdout_fingerprints.add(
            task_fingerprint(
                pilot_task
            )
        )

        seed_lists = {
            tuple(
                int(seed)
                for seed in json.loads(
                    str(value)
                )
            )
            for value in table[
                "model_seeds_json"
            ]
        }

        if seed_lists != {
            EXPECTED_SEEDS
        }:
            raise RuntimeError(
                f"{task_id} uses unexpected "
                f"model seeds: {seed_lists}"
            )

        table = (
            table.set_index(
                "config_id"
            )
            .sort_index()
        )

        records = []

        for config_id, row in (
            table.iterrows()
        ):
            values = parse_seed_vector(
                row[
                    "validation_accuracies_by_seed_json"
                ],
                expected_length=2,
            )

            records.append(
                {
                    "config_id": str(
                        config_id
                    ),
                    "seed101": float(
                        values[0]
                    ),
                    "seed202": float(
                        values[1]
                    ),
                    "robust_two_seed": float(
                        values.mean()
                    ),
                }
            )

        performance = (
            pd.DataFrame(records)
            .set_index("config_id")
            .sort_index()
        )

        seed101_ranking = rank_configs(
            performance["seed101"]
        )

        robust_ranking = rank_configs(
            performance[
                "robust_two_seed"
            ]
        )

        seed101_best_id = (
            seed101_ranking[0]
        )

        robust_best_id = (
            robust_ranking[0]
        )

        seed101_best_score = float(
            performance.loc[
                seed101_best_id,
                "seed101",
            ]
        )

        robust_best_score = float(
            performance.loc[
                robust_best_id,
                "robust_two_seed",
            ]
        )

        tied_ids = {
            str(config_id)
            for config_id
            in performance.index[
                (
                    seed101_best_score
                    - performance[
                        "seed101"
                    ]
                )
                <= args.epsilon
                + 1e-12
            ]
        }

        for candidate_k in (
            k_values
        ):
            candidate_ids = (
                set(
                    seed101_ranking[
                        :candidate_k
                    ]
                )
                | tied_ids
            )

            candidate_scores = (
                performance.loc[
                    sorted(
                        candidate_ids
                    ),
                    "robust_two_seed",
                ]
            )

            candidate_ranking = (
                rank_configs(
                    candidate_scores
                )
            )

            candidate_best_id = (
                candidate_ranking[0]
            )

            candidate_score = float(
                performance.loc[
                    candidate_best_id,
                    "robust_two_seed",
                ]
            )

            regret = (
                robust_best_score
                - candidate_score
            )

            result_rows.append(
                {
                    "task_id": task_id,
                    "dataset": (
                        pilot_task.dataset
                    ),
                    "n_way": (
                        pilot_task.n_way
                    ),
                    "n_shot": (
                        pilot_task.n_shot
                    ),
                    "candidate_k": (
                        candidate_k
                    ),
                    "candidate_count": len(
                        candidate_ids
                    ),
                    "seed101_tie_count": len(
                        tied_ids
                    ),
                    "seed101_best_config_id": (
                        seed101_best_id
                    ),
                    "robust_best_config_id": (
                        robust_best_id
                    ),
                    "candidate_best_config_id": (
                        candidate_best_id
                    ),
                    "exact_robust_best_recalled": bool(
                        robust_best_id
                        in candidate_ids
                    ),
                    "candidate_regret": float(
                        regret
                    ),
                    "candidate_regret_pp": float(
                        100 * regret
                    ),
                    "candidate_equivalent": bool(
                        regret
                        <= args.epsilon
                        + 1e-12
                    ),
                }
            )

    id_overlap = (
        screening_ids
        & holdout_ids
    )

    fingerprint_overlap = (
        screening_fingerprints
        & holdout_fingerprints
    )

    print(
        "Screening/holdout ID overlap:",
        len(id_overlap),
    )

    print(
        "Screening/holdout task-fingerprint overlap:",
        len(fingerprint_overlap),
    )

    if fingerprint_overlap:
        raise RuntimeError(
            "Pilot and screening sets contain "
            "identical underlying tasks"
        )

    result = pd.DataFrame(
        result_rows
    )

    summaries = []

    for candidate_k in k_values:
        subset = result[
            result["candidate_k"]
            == candidate_k
        ]

        summary = {
            "candidate_k": int(
                candidate_k
            ),
            "task_count": int(
                len(subset)
            ),
            "mean_candidate_count": float(
                subset[
                    "candidate_count"
                ].mean()
            ),
            "maximum_candidate_count": int(
                subset[
                    "candidate_count"
                ].max()
            ),
            "exact_robust_best_recall": float(
                subset[
                    "exact_robust_best_recalled"
                ].mean()
            ),
            "equivalent_coverage": float(
                subset[
                    "candidate_equivalent"
                ].mean()
            ),
            "mean_regret_pp": float(
                subset[
                    "candidate_regret_pp"
                ].mean()
            ),
            "maximum_regret_pp": float(
                subset[
                    "candidate_regret_pp"
                ].max()
            ),
        }

        summary["passes_gate"] = bool(
            summary[
                "exact_robust_best_recall"
            ] >= 0.90
            and summary[
                "equivalent_coverage"
            ] == 1.0
            and summary[
                "maximum_regret_pp"
            ]
            <= (
                100 * args.epsilon
                + 1e-10
            )
        )

        summaries.append(summary)

    passing_k = [
        summary["candidate_k"]
        for summary in summaries
        if summary["passes_gate"]
    ]

    selected_k = (
        min(passing_k)
        if passing_k
        else None
    )

    summary_payload = {
        "schema_version": 2,
        "validation_type": (
            "independent_task_fingerprint_"
            "two_seed_pilot_holdout"
        ),
        "holdout_task_count": (
            EXPECTED_TASKS
        ),
        "configuration_count": (
            EXPECTED_CONFIGS
        ),
        "screening_holdout_id_overlap_count": (
            len(id_overlap)
        ),
        "screening_holdout_task_fingerprint_overlap_count": (
            len(fingerprint_overlap)
        ),
        "model_seeds": list(
            EXPECTED_SEEDS
        ),
        "equivalence_epsilon": (
            args.epsilon
        ),
        "candidate_summaries": (
            summaries
        ),
        "selected_candidate_k": (
            selected_k
        ),
        "passes_holdout_validation": bool(
            selected_k is not None
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    args.summary_output.write_text(
        json.dumps(
            summary_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        pd.DataFrame(
            summaries
        ).to_string(index=False)
    )

    print()
    print(
        "Selected holdout-validated K:",
        selected_k,
    )

    print(
        "Candidate strategy holdout:",
        (
            "PASS"
            if selected_k is not None
            else "FAIL"
        ),
    )

    print()
    print("Worst K=4 cases:")

    print(
        result[
            result["candidate_k"] == 4
        ].sort_values(
            "candidate_regret_pp",
            ascending=False,
        )[
            [
                "dataset",
                "task_id",
                "candidate_count",
                "robust_best_config_id",
                "candidate_best_config_id",
                "exact_robust_best_recalled",
                "candidate_regret_pp",
                "candidate_equivalent",
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print()
    print("Output:", args.output)
    print(
        "Summary:",
        args.summary_output,
    )
    print(
        "CANDIDATE HOLDOUT VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
