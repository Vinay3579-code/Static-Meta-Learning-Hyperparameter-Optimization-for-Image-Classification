from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ranked-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--equivalence-epsilon",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )

    return parser.parse_args()


def parse_json_vector(
    value: object,
    column_name: str,
) -> np.ndarray:
    if pd.isna(value):
        raise ValueError(
            f"Missing value in {column_name}"
        )

    parsed = json.loads(str(value))

    array = np.asarray(
        parsed,
        dtype=np.float64,
    )

    if array.ndim != 1 or len(array) == 0:
        raise ValueError(
            f"Invalid vector in {column_name}"
        )

    if not np.isfinite(array).all():
        raise FloatingPointError(
            f"Non-finite values in {column_name}"
        )

    return array


def main() -> None:
    args = parse_args()

    if args.equivalence_epsilon < 0:
        raise ValueError(
            "equivalence-epsilon must be non-negative"
        )

    if args.softmax_temperature <= 0:
        raise ValueError(
            "softmax-temperature must be positive"
        )

    if not 0.5 < args.confidence_level < 1.0:
        raise ValueError(
            "confidence-level must be between 0.5 and 1"
        )

    table = pd.read_csv(args.input)

    required_columns = {
        "task_id",
        "dataset",
        "config_id",
        "validation_accuracy_mean",
        "validation_accuracy_seed_std",
        "total_elapsed_seconds",
    }

    missing = required_columns - set(table.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    if table.empty:
        raise RuntimeError("Oracle table is empty")

    if table["task_id"].nunique() != 1:
        raise RuntimeError(
            "Input must contain exactly one task"
        )

    if table["config_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate configuration IDs detected"
        )

    accuracies = table[
        "validation_accuracy_mean"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(accuracies).all():
        raise FloatingPointError(
            "Non-finite validation accuracies"
        )

    reference_ranking = table.sort_values(
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

    reference_best_id = str(
        reference_ranking.iloc[0]["config_id"]
    )

    best_accuracy = float(
        reference_ranking.iloc[0][
            "validation_accuracy_mean"
        ]
    )

    table["accuracy_regret"] = (
        best_accuracy
        - table["validation_accuracy_mean"]
    )

    table["oracle_equivalent"] = (
        table["accuracy_regret"]
        <= args.equivalence_epsilon + 1e-12
    )

    unnormalized = np.exp(
        -table["accuracy_regret"].to_numpy(
            dtype=np.float64
        )
        / args.softmax_temperature
    )

    table["soft_oracle_probability"] = (
        unnormalized / unnormalized.sum()
    )

    table["paired_regret_mean"] = np.nan
    table["paired_regret_std"] = np.nan
    table["paired_regret_upper_confidence"] = np.nan
    table["paired_noninferior"] = False

    paired_column = (
        "validation_accuracies_by_seed_json"
    )

    paired_analysis_available = (
        paired_column in table.columns
        and table[paired_column].notna().all()
    )

    paired_seed_count = 0

    if paired_analysis_available:
        best_row = table[
            table["config_id"]
            == reference_best_id
        ]

        if len(best_row) != 1:
            raise RuntimeError(
                "Could not resolve reference-best row"
            )

        best_seed_values = parse_json_vector(
            best_row.iloc[0][paired_column],
            paired_column,
        )

        paired_seed_count = len(best_seed_values)

        if paired_seed_count < 2:
            raise RuntimeError(
                "Paired analysis requires at least two seeds"
            )

        critical_value = float(
            t.ppf(
                args.confidence_level,
                df=paired_seed_count - 1,
            )
        )

        for row_index, row in table.iterrows():
            candidate_values = parse_json_vector(
                row[paired_column],
                paired_column,
            )

            if len(candidate_values) != paired_seed_count:
                raise RuntimeError(
                    "Configurations use inconsistent seed counts"
                )

            # Positive values mean the candidate performed worse than
            # the aggregate-best configuration on the same model seed.
            paired_regrets = (
                best_seed_values
                - candidate_values
            )

            regret_mean = float(
                paired_regrets.mean()
            )

            regret_std = float(
                paired_regrets.std(ddof=1)
            )

            standard_error = (
                regret_std
                / np.sqrt(paired_seed_count)
            )

            upper_confidence = (
                regret_mean
                + critical_value * standard_error
            )

            table.loc[
                row_index,
                "paired_regret_mean",
            ] = regret_mean

            table.loc[
                row_index,
                "paired_regret_std",
            ] = regret_std

            table.loc[
                row_index,
                "paired_regret_upper_confidence",
            ] = upper_confidence

            table.loc[
                row_index,
                "paired_noninferior",
            ] = bool(
                upper_confidence
                <= args.equivalence_epsilon
                + 1e-12
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

    table["robust_rank"] = (
        np.arange(len(table)) + 1
    )

    equivalent_table = table[
        table["oracle_equivalent"]
    ].copy()

    paired_table = table[
        table["paired_noninferior"]
    ].copy()

    second_best_accuracy = (
        float(
            table.iloc[1][
                "validation_accuracy_mean"
            ]
        )
        if len(table) > 1
        else best_accuracy
    )

    accuracy_range = float(
        accuracies.max() - accuracies.min()
    )

    summary = {
        "schema_version": 2,
        "task_id": str(table.iloc[0]["task_id"]),
        "dataset": str(table.iloc[0]["dataset"]),
        "configuration_count": int(len(table)),
        "best_config_id": reference_best_id,
        "best_validation_accuracy": best_accuracy,
        "second_best_validation_accuracy": (
            second_best_accuracy
        ),
        "best_to_second_gap": float(
            best_accuracy - second_best_accuracy
        ),
        "minimum_validation_accuracy": float(
            accuracies.min()
        ),
        "accuracy_range": accuracy_range,
        "accuracy_standard_deviation": float(
            accuracies.std(ddof=1)
            if len(accuracies) > 1
            else 0.0
        ),
        "unique_accuracy_count_4dp": int(
            np.unique(
                np.round(accuracies, decimals=4)
            ).size
        ),
        "saturation_fraction_at_99_9_percent": float(
            np.mean(accuracies >= 0.999)
        ),
        "equivalence_epsilon": float(
            args.equivalence_epsilon
        ),
        "equivalent_configuration_count": int(
            len(equivalent_table)
        ),
        "equivalent_config_ids": (
            equivalent_table["config_id"].tolist()
        ),
        "softmax_temperature": float(
            args.softmax_temperature
        ),
        "soft_probability_sum": float(
            table[
                "soft_oracle_probability"
            ].sum()
        ),
        "hard_oracle_is_unique": bool(
            len(equivalent_table) == 1
        ),
        "paired_seed_analysis_available": bool(
            paired_analysis_available
        ),
        "paired_seed_count": int(
            paired_seed_count
        ),
        "paired_confidence_level": float(
            args.confidence_level
        ),
        "paired_noninferior_configuration_count": int(
            len(paired_table)
        ),
        "paired_noninferior_config_ids": (
            paired_table["config_id"].tolist()
        ),
    }

    args.ranked_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        args.ranked_output,
        index=False,
    )

    args.summary_output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Task:", summary["task_id"])
    print("Dataset:", summary["dataset"])
    print(
        "Best accuracy:",
        f"{100 * best_accuracy:.3f}%",
    )
    print(
        "Accuracy range:",
        f"{100 * accuracy_range:.3f} percentage points",
    )
    print(
        "Fixed-epsilon equivalent configurations:",
        len(equivalent_table),
    )
    print(
        "Paired non-inferior configurations:",
        len(paired_table)
        if paired_analysis_available
        else "unavailable",
    )
    print(
        "Paired seeds:",
        paired_seed_count,
    )

    print()
    print("Fixed-epsilon oracle-equivalent set:")
    print(
        equivalent_table[
            [
                "robust_rank",
                "config_id",
                "validation_accuracy_mean",
                "accuracy_regret",
                "soft_oracle_probability",
            ]
        ].to_string(index=False)
    )

    if paired_analysis_available:
        print()
        print("Paired non-inferior set:")
        print(
            paired_table[
                [
                    "robust_rank",
                    "config_id",
                    "paired_regret_mean",
                    "paired_regret_upper_confidence",
                ]
            ].to_string(index=False)
        )

    print()
    print("Ranked output:", args.ranked_output)
    print("Summary output:", args.summary_output)
    print("ORACLE ANALYSIS: PASS")


if __name__ == "__main__":
    main()
