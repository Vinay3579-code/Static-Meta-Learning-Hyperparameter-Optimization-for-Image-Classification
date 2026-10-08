from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "results/final_test_eval_v2/"
    "all_method_test_results.csv"
)

OUTPUT = Path(
    "results/final_analysis_v2"
)

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260828

REFERENCE = (
    "validation_selected_oracle"
)

CORE_METHODS = [
    "zero_sml_ensemble",
    "probe_sml_ensemble",
    "global_majority",
    "regime_majority",
    "dataset_majority",
    "dataset_regime_majority",
    "probe_weighted_extra_trees",
    REFERENCE,
]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    indices = rng.integers(
        0,
        n,
        size=(
            n_boot,
            n,
        ),
    )

    boot = values[
        indices
    ].mean(
        axis=1
    )

    return (
        float(
            np.percentile(
                boot,
                2.5,
            )
        ),
        float(
            np.percentile(
                boot,
                97.5,
            )
        ),
    )


def method_summary(
    table: pd.DataFrame,
):
    rows = []

    for method, group in (
        table.groupby(
            "method",
            sort=False,
        )
    ):
        accuracy_pp = (
            100.0
            * group[
                "test_accuracy_mean"
            ].to_numpy(
                dtype=np.float64
            )
        )

        gaps = group[
            "test_gap_to_validation_oracle_pp"
        ].to_numpy(
            dtype=np.float64
        )

        (
            acc_ci_low,
            acc_ci_high,
        ) = bootstrap_mean_ci(
            accuracy_pp,
            seed=(
                BOOTSTRAP_SEED
                + sum(
                    ord(c)
                    for c in method
                )
            ),
        )

        (
            gap_ci_low,
            gap_ci_high,
        ) = bootstrap_mean_ci(
            gaps,
            seed=(
                BOOTSTRAP_SEED
                + 10000
                + sum(
                    ord(c)
                    for c in method
                )
            ),
        )

        rows.append({
            "method":
                method,

            "tasks":
                len(group),

            "mean_test_accuracy_percent":
                accuracy_pp.mean(),

            "test_accuracy_std_pp":
                accuracy_pp.std(
                    ddof=1
                ),

            "mean_test_accuracy_ci95_low":
                acc_ci_low,

            "mean_test_accuracy_ci95_high":
                acc_ci_high,

            "mean_signed_gap_pp":
                gaps.mean(),

            "mean_signed_gap_ci95_low":
                gap_ci_low,

            "mean_signed_gap_ci95_high":
                gap_ci_high,

            "median_signed_gap_pp":
                np.median(
                    gaps
                ),

            "p95_signed_gap_pp":
                np.percentile(
                    gaps,
                    95,
                ),

            "max_signed_gap_pp":
                gaps.max(),

            "min_signed_gap_pp":
                gaps.min(),

            "mean_absolute_gap_pp":
                np.abs(
                    gaps
                ).mean(),

            "hard_oracle_agreement_percent":
                100.0
                * group[
                    "hard_oracle_agreement"
                ].astype(
                    bool
                ).mean(),

            "validation_oracle_equivalent_percent":
                100.0
                * group[
                    "validation_oracle_equivalent"
                ].astype(
                    bool
                ).mean(),

            "beats_validation_oracle_percent":
                100.0
                * (
                    gaps < -1e-12
                ).mean(),

            "ties_validation_oracle_percent":
                100.0
                * (
                    np.abs(
                        gaps
                    )
                    <= 1e-12
                ).mean(),

            "loses_to_validation_oracle_percent":
                100.0
                * (
                    gaps > 1e-12
                ).mean(),
        })

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "mean_signed_gap_pp",
            "mean_test_accuracy_percent",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def breakdown(
    table: pd.DataFrame,
    column: str,
):
    rows = []

    for (
        group_value,
        method,
    ), group in table.groupby(
        [
            column,
            "method",
        ]
    ):
        accuracy = (
            100.0
            * group[
                "test_accuracy_mean"
            ].to_numpy(
                dtype=np.float64
            )
        )

        gap = group[
            "test_gap_to_validation_oracle_pp"
        ].to_numpy(
            dtype=np.float64
        )

        rows.append({
            column:
                group_value,

            "method":
                method,

            "tasks":
                len(group),

            "mean_test_accuracy_percent":
                accuracy.mean(),

            "mean_signed_gap_pp":
                gap.mean(),

            "median_signed_gap_pp":
                np.median(
                    gap
                ),

            "hard_oracle_agreement_percent":
                100.0
                * group[
                    "hard_oracle_agreement"
                ].astype(
                    bool
                ).mean(),

            "validation_oracle_equivalent_percent":
                100.0
                * group[
                    "validation_oracle_equivalent"
                ].astype(
                    bool
                ).mean(),
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            column,
            "mean_signed_gap_pp",
        ]
    ).reset_index(
        drop=True
    )


def paired_comparison(
    table: pd.DataFrame,
    method_a: str,
    method_b: str,
):
    a = (
        table[
            table["method"]
            == method_a
        ][
            [
                "task_id",
                "test_accuracy_mean",
            ]
        ]
        .rename(
            columns={
                "test_accuracy_mean":
                    "accuracy_a"
            }
        )
    )

    b = (
        table[
            table["method"]
            == method_b
        ][
            [
                "task_id",
                "test_accuracy_mean",
            ]
        ]
        .rename(
            columns={
                "test_accuracy_mean":
                    "accuracy_b"
            }
        )
    )

    merged = a.merge(
        b,
        on="task_id",
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged
    ) != 120:
        raise RuntimeError(
            f"{method_a} vs "
            f"{method_b}: "
            f"expected 120 paired tasks, "
            f"found {len(merged)}"
        )

    difference = (
        100.0
        * (
            merged[
                "accuracy_a"
            ].to_numpy(
                dtype=np.float64
            )
            -
            merged[
                "accuracy_b"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    (
        ci_low,
        ci_high,
    ) = bootstrap_mean_ci(
        difference,
        seed=(
            BOOTSTRAP_SEED
            + sum(
                ord(c)
                for c in (
                    method_a
                    + method_b
                )
            )
        ),
    )

    tolerance = 1e-12

    return {
        "method_a":
            method_a,

        "method_b":
            method_b,

        "tasks":
            len(merged),

        "mean_accuracy_difference_a_minus_b_pp":
            difference.mean(),

        "median_accuracy_difference_pp":
            np.median(
                difference
            ),

        "ci95_low_pp":
            ci_low,

        "ci95_high_pp":
            ci_high,

        "a_wins":
            int(
                (
                    difference
                    > tolerance
                ).sum()
            ),

        "ties":
            int(
                (
                    np.abs(
                        difference
                    )
                    <= tolerance
                ).sum()
            ),

        "a_losses":
            int(
                (
                    difference
                    < -tolerance
                ).sum()
            ),
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pd.read_csv(
        INPUT
    )

    print(
        "Rows:",
        len(table),
    )

    print(
        "Unique tasks:",
        table[
            "task_id"
        ].nunique(),
    )

    print(
        "Methods:",
        table[
            "method"
        ].nunique(),
    )

    if (
        table["task_id"]
        .nunique()
        != 120
    ):
        raise RuntimeError(
            "Expected 120 tasks"
        )

    counts = (
        table.groupby(
            "method"
        )[
            "task_id"
        ]
        .nunique()
    )

    incomplete = counts[
        counts != 120
    ]

    if not incomplete.empty:
        raise RuntimeError(
            "Methods missing tasks:\n"
            f"{incomplete}"
        )

    duplicate = table.duplicated(
        [
            "task_id",
            "method",
        ]
    )

    if duplicate.any():
        raise RuntimeError(
            "Duplicate task/method rows"
        )

    reference = table[
        table["method"]
        == REFERENCE
    ]

    if len(reference) != 120:
        raise RuntimeError(
            "Reference method incomplete"
        )

    reference_gap = (
        reference[
            "test_gap_to_validation_oracle_pp"
        ].to_numpy(
            dtype=np.float64
        )
    )

    if not np.allclose(
        reference_gap,
        0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Reference gap is not zero"
        )

    summary = method_summary(
        table
    )

    summary.to_csv(
        OUTPUT
        / "method_summary.csv",
        index=False,
    )

    dataset_table = breakdown(
        table,
        "dataset",
    )

    dataset_table.to_csv(
        OUTPUT
        / "breakdown_by_dataset.csv",
        index=False,
    )

    regime_table = breakdown(
        table,
        "regime",
    )

    regime_table.to_csv(
        OUTPUT
        / "breakdown_by_regime.csv",
        index=False,
    )

    present_core = [
        method
        for method
        in CORE_METHODS
        if method
        in set(
            table["method"]
        )
    ]

    missing_core = [
        method
        for method
        in CORE_METHODS
        if method
        not in set(
            table["method"]
        )
    ]

    core = (
        summary[
            summary["method"]
            .isin(
                present_core
            )
        ]
        .copy()
    )

    order = {
        method:
            index
        for index, method
        in enumerate(
            CORE_METHODS
        )
    }

    core[
        "_order"
    ] = (
        core["method"]
        .map(order)
    )

    core = (
        core.sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
    )

    core.to_csv(
        OUTPUT
        / "paper_core_table.csv",
        index=False,
    )

    pairwise_pairs = []

    for competitor in [
        "probe_sml_ensemble",
        "global_majority",
        "regime_majority",
        "dataset_majority",
        "dataset_regime_majority",
        "probe_weighted_extra_trees",
        REFERENCE,
    ]:
        if (
            "zero_sml_ensemble"
            in set(
                table["method"]
            )
            and competitor
            in set(
                table["method"]
            )
        ):
            pairwise_pairs.append(
                paired_comparison(
                    table,
                    "zero_sml_ensemble",
                    competitor,
                )
            )

    if (
        "probe_sml_ensemble"
        in set(
            table["method"]
        )
        and REFERENCE
        in set(
            table["method"]
        )
    ):
        pairwise_pairs.append(
            paired_comparison(
                table,
                "probe_sml_ensemble",
                REFERENCE,
            )
        )

    pairwise = pd.DataFrame(
        pairwise_pairs
    )

    pairwise.to_csv(
        OUTPUT
        / "paired_bootstrap_comparisons.csv",
        index=False,
    )

    zero_probe_configs = (
        table[
            table["method"].isin(
                [
                    "zero_sml_ensemble",
                    "probe_sml_ensemble",
                ]
            )
        ][
            [
                "task_id",
                "method",
                "config_id",
                "test_accuracy_mean",
            ]
        ]
        .pivot(
            index="task_id",
            columns="method",
            values=[
                "config_id",
                "test_accuracy_mean",
            ],
        )
    )

    zero_configs = (
        zero_probe_configs[
            "config_id"
        ][
            "zero_sml_ensemble"
        ]
    )

    probe_configs = (
        zero_probe_configs[
            "config_id"
        ][
            "probe_sml_ensemble"
        ]
    )

    zero_acc = (
        zero_probe_configs[
            "test_accuracy_mean"
        ][
            "zero_sml_ensemble"
        ].astype(float)
    )

    probe_acc = (
        zero_probe_configs[
            "test_accuracy_mean"
        ][
            "probe_sml_ensemble"
        ].astype(float)
    )

    config_same = (
        zero_configs
        == probe_configs
    )

    accuracy_difference = (
        100.0
        * (
            zero_acc
            - probe_acc
        )
    )

    zp = {
        "tasks":
            120,

        "same_recommendation_count":
            int(
                config_same.sum()
            ),

        "different_recommendation_count":
            int(
                (~config_same).sum()
            ),

        "recommendation_agreement_percent":
            float(
                100.0
                * config_same.mean()
            ),

        "zero_higher_test_accuracy_count":
            int(
                (
                    accuracy_difference
                    > 1e-12
                ).sum()
            ),

        "same_test_accuracy_count":
            int(
                (
                    np.abs(
                        accuracy_difference
                    )
                    <= 1e-12
                ).sum()
            ),

        "probe_higher_test_accuracy_count":
            int(
                (
                    accuracy_difference
                    < -1e-12
                ).sum()
            ),

        "mean_zero_minus_probe_accuracy_pp":
            float(
                accuracy_difference.mean()
            ),
    }

    (
        zp_ci_low,
        zp_ci_high,
    ) = bootstrap_mean_ci(
        accuracy_difference.to_numpy(),
        seed=BOOTSTRAP_SEED + 999,
    )

    zp[
        "zero_minus_probe_ci95_low_pp"
    ] = zp_ci_low

    zp[
        "zero_minus_probe_ci95_high_pp"
    ] = zp_ci_high

    (
        OUTPUT
        / "zero_probe_test_comparison.json"
    ).write_text(
        json.dumps(
            zp,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "schema_version":
            1,

        "input":
            str(INPUT),

        "tasks":
            int(
                table[
                    "task_id"
                ].nunique()
            ),

        "methods":
            int(
                table[
                    "method"
                ].nunique()
            ),

        "bootstrap_samples":
            BOOTSTRAP_SAMPLES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "reference_method":
            REFERENCE,

        "gap_definition":
            (
                "100 * "
                "(validation-selected "
                "oracle test accuracy "
                "- method test accuracy)"
            ),

        "gap_interpretation":
            (
                "positive = method worse "
                "than validation-selected "
                "oracle; negative = method "
                "better on held-out test bank"
            ),

        "missing_core_methods":
            missing_core,
    }

    (
        OUTPUT
        / "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "FINAL METHOD SUMMARY"
    )
    print("=" * 100)

    display_columns = [
        "method",
        "mean_test_accuracy_percent",
        "mean_signed_gap_pp",
        "median_signed_gap_pp",
        "p95_signed_gap_pp",
        "max_signed_gap_pp",
        "hard_oracle_agreement_percent",
        "validation_oracle_equivalent_percent",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print("=" * 100)
    print(
        "ZERO-SML PAIRED COMPARISONS"
    )
    print("=" * 100)

    if not pairwise.empty:
        print(
            pairwise.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )

    print()
    print("=" * 100)
    print(
        "ZERO-SML vs PROBE-SML"
    )
    print("=" * 100)

    for key, value in (
        zp.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "Outputs:",
        OUTPUT,
    )

    print()
    print(
        "FINAL TEST ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()
