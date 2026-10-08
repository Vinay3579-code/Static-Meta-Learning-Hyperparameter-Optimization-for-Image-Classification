from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
    )
    parser.add_argument(
        "--expected-config-count",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--task-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--long-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ranked_paths = sorted(
        args.input_dir.glob(
            "*_ranked.csv"
        )
    )

    if not ranked_paths:
        raise RuntimeError(
            "No ranked oracle tables found"
        )

    long_tables = []
    task_rows = []

    for path in ranked_paths:
        table = pd.read_csv(path)

        if len(table) != args.expected_config_count:
            raise RuntimeError(
                f"{path} contains {len(table)} rows; "
                f"expected {args.expected_config_count}"
            )

        if table["task_id"].nunique() != 1:
            raise RuntimeError(
                f"{path} mixes task IDs"
            )

        if (
            table["config_id"].nunique()
            != args.expected_config_count
        ):
            raise RuntimeError(
                f"{path} does not contain "
                f"{args.expected_config_count} configs"
            )

        task_id = str(
            table.iloc[0]["task_id"]
        )
        dataset = str(
            table.iloc[0]["dataset"]
        )

        table = table.copy()
        table["source_path"] = str(path)

        task_range = float(
            table[
                "validation_accuracy_mean"
            ].max()
            - table[
                "validation_accuracy_mean"
            ].min()
        )

        table["normalized_regret"] = (
            table["accuracy_regret"]
            / max(task_range, 1e-12)
        )

        best_row = table.sort_values(
            "robust_rank"
        ).iloc[0]

        paired_count = (
            int(
                table[
                    "paired_noninferior"
                ].astype(bool).sum()
            )
            if "paired_noninferior"
            in table.columns
            else 0
        )

        task_rows.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "best_config_id": str(
                    best_row["config_id"]
                ),
                "best_validation_accuracy": float(
                    best_row[
                        "validation_accuracy_mean"
                    ]
                ),
                "accuracy_range": task_range,
                "equivalent_configuration_count": int(
                    table[
                        "oracle_equivalent"
                    ].astype(bool).sum()
                ),
                "paired_noninferior_configuration_count": (
                    paired_count
                ),
                "saturation_fraction": float(
                    np.mean(
                        table[
                            "validation_accuracy_mean"
                        ]
                        >= 0.999
                    )
                ),
            }
        )

        long_tables.append(table)

    task_table = pd.DataFrame(
        task_rows
    )

    if (
        args.expected_task_count is not None
        and len(task_table)
        != args.expected_task_count
    ):
        raise RuntimeError(
            f"Expected {args.expected_task_count} tasks, "
            f"found {len(task_table)}"
        )

    long_table = pd.concat(
        long_tables,
        ignore_index=True,
    )

    expected_rows = (
        len(task_table)
        * args.expected_config_count
    )

    if len(long_table) != expected_rows:
        raise RuntimeError(
            "Unexpected long-table row count"
        )

    config_summary = (
        long_table
        .groupby("config_id")
        .agg(
            mean_validation_accuracy=(
                "validation_accuracy_mean",
                "mean",
            ),
            mean_regret=(
                "accuracy_regret",
                "mean",
            ),
            median_regret=(
                "accuracy_regret",
                "median",
            ),
            maximum_regret=(
                "accuracy_regret",
                "max",
            ),
            mean_normalized_regret=(
                "normalized_regret",
                "mean",
            ),
            mean_rank=(
                "robust_rank",
                "mean",
            ),
        )
        .reset_index()
    )

    hard_win_counts = (
        task_table[
            "best_config_id"
        ]
        .value_counts()
    )

    equivalent_counts = (
        long_table[
            long_table[
                "oracle_equivalent"
            ].astype(bool)
        ]
        ["config_id"]
        .value_counts()
    )

    if "paired_noninferior" in long_table.columns:
        paired_counts = (
            long_table[
                long_table[
                    "paired_noninferior"
                ].astype(bool)
            ]
            ["config_id"]
            .value_counts()
        )
    else:
        paired_counts = pd.Series(
            dtype=np.int64
        )

    config_summary["hard_wins"] = (
        config_summary["config_id"]
        .map(hard_win_counts)
        .fillna(0)
        .astype(int)
    )

    config_summary["equivalent_tasks"] = (
        config_summary["config_id"]
        .map(equivalent_counts)
        .fillna(0)
        .astype(int)
    )

    config_summary[
        "paired_noninferior_tasks"
    ] = (
        config_summary["config_id"]
        .map(paired_counts)
        .fillna(0)
        .astype(int)
    )

    config_summary = config_summary.sort_values(
        by=[
            "mean_regret",
            "mean_rank",
            "config_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    config_summary["global_rank"] = (
        np.arange(len(config_summary)) + 1
    )

    global_best_config_id = str(
        config_summary.iloc[0][
            "config_id"
        ]
    )

    global_rows = long_table[
        long_table["config_id"]
        == global_best_config_id
    ].copy()

    if len(global_rows) != len(task_table):
        raise RuntimeError(
            "Global configuration missing from tasks"
        )

    global_regrets = global_rows[
        "accuracy_regret"
    ].to_numpy(dtype=np.float64)

    winner_probabilities = (
        hard_win_counts.to_numpy(
            dtype=np.float64
        )
        / len(task_table)
    )

    winner_entropy = float(
        -np.sum(
            winner_probabilities
            * np.log(
                winner_probabilities
            )
        )
    )

    normalized_winner_entropy = (
        winner_entropy
        / math.log(args.expected_config_count)
        if winner_entropy > 0
        else 0.0
    )

    dataset_rows = []

    for dataset, dataset_tasks in (
        task_table.groupby("dataset")
    ):
        dataset_task_ids = set(
            dataset_tasks["task_id"]
        )

        dataset_global_rows = global_rows[
            global_rows["task_id"].isin(
                dataset_task_ids
            )
        ]

        dataset_rows.append(
            {
                "dataset": dataset,
                "task_count": len(
                    dataset_tasks
                ),
                "unique_hard_winners": int(
                    dataset_tasks[
                        "best_config_id"
                    ].nunique()
                ),
                "mean_best_accuracy": float(
                    dataset_tasks[
                        "best_validation_accuracy"
                    ].mean()
                ),
                "mean_accuracy_range": float(
                    dataset_tasks[
                        "accuracy_range"
                    ].mean()
                ),
                "mean_equivalent_count": float(
                    dataset_tasks[
                        "equivalent_configuration_count"
                    ].mean()
                ),
                "mean_paired_noninferior_count": float(
                    dataset_tasks[
                        "paired_noninferior_configuration_count"
                    ].mean()
                ),
                "global_config_mean_regret": float(
                    dataset_global_rows[
                        "accuracy_regret"
                    ].mean()
                ),
            }
        )

    dataset_summary = pd.DataFrame(
        dataset_rows
    )

    summary = {
        "schema_version": 1,
        "task_count": int(
            len(task_table)
        ),
        "configuration_count": int(
            config_summary["config_id"].nunique()
        ),
        "unique_hard_winner_count": int(
            task_table[
                "best_config_id"
            ].nunique()
        ),
        "hard_winner_entropy_nats": (
            winner_entropy
        ),
        "hard_winner_entropy_normalized_by_40": (
            normalized_winner_entropy
        ),
        "mean_equivalent_configuration_count": float(
            task_table[
                "equivalent_configuration_count"
            ].mean()
        ),
        "mean_paired_noninferior_configuration_count": float(
            task_table[
                "paired_noninferior_configuration_count"
            ].mean()
        ),
        "global_best_fixed_config_id": (
            global_best_config_id
        ),
        "global_fixed_config_mean_regret": float(
            global_regrets.mean()
        ),
        "global_fixed_config_median_regret": float(
            np.median(global_regrets)
        ),
        "global_fixed_config_maximum_regret": float(
            global_regrets.max()
        ),
        "dataset_summary": (
            dataset_summary.to_dict(
                orient="records"
            )
        ),
    }

    for path in [
        args.task_output,
        args.config_output,
        args.long_output,
        args.summary_output,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    task_table.to_csv(
        args.task_output,
        index=False,
    )

    config_summary.to_csv(
        args.config_output,
        index=False,
    )

    long_table.to_csv(
        args.long_output,
        index=False,
    )

    args.summary_output.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Tasks:", len(task_table))
    print(
        "Unique hard winners:",
        summary[
            "unique_hard_winner_count"
        ],
    )
    print(
        "Mean equivalent configurations:",
        f"{summary['mean_equivalent_configuration_count']:.2f}",
    )
    print(
        "Mean paired non-inferior configurations:",
        f"{summary['mean_paired_noninferior_configuration_count']:.2f}",
    )
    print(
        "Global fixed configuration:",
        global_best_config_id,
    )
    print(
        "Global fixed mean regret:",
        f"{100 * summary['global_fixed_config_mean_regret']:.3f} pp",
    )
    print(
        "Global fixed maximum regret:",
        f"{100 * summary['global_fixed_config_maximum_regret']:.3f} pp",
    )

    print()
    print("Dataset summary:")
    print(
        dataset_summary.to_string(
            index=False
        )
    )

    print()
    print("Top ten global configurations:")
    print(
        config_summary[
            [
                "global_rank",
                "config_id",
                "mean_regret",
                "mean_rank",
                "hard_wins",
                "equivalent_tasks",
                "paired_noninferior_tasks",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("ORACLE COLLECTION SUMMARY: PASS")


if __name__ == "__main__":
    main()
