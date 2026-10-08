from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OLD_RESULTS = Path(
    "results/final_test_eval_v2/"
    "all_method_test_results.csv"
)

NEW_ROOT = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation"
)

NEW_PER_TASK = (
    NEW_ROOT
    / "per_task_method_results.csv"
)

NEW_RUN_SUMMARY = (
    NEW_ROOT
    / "run_summary.csv"
)

OUT = Path(
    "results/reviewer_baseline_analysis_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260828


OLD_METHODS = [
    "zero_sml_ensemble",
    "probe_sml_ensemble",
    "probe_weighted_extra_trees",
    "global_majority",
    "regime_majority",
    "dataset_majority",
    "dataset_regime_majority",
    "validation_selected_oracle",
]


NEW_METHODS = [
    "random_search",
    "bayesian_optimization_gp_ei",
    "hyperband",
    "bohb_finite_portfolio",
    "probe_npp",
]


DISPLAY_NAMES = {
    "zero_sml_ensemble":
        "Zero-SML (ours)",

    "probe_sml_ensemble":
        "Probe-SML (ours)",

    "probe_weighted_extra_trees":
        "Probe ExtraTrees",

    "global_majority":
        "Global default",

    "regime_majority":
        "Regime default",

    "dataset_majority":
        "Dataset default*",

    "dataset_regime_majority":
        "Dataset + regime default*",

    "validation_selected_oracle":
        "Validation-selected reference",

    "random_search":
        "Random Search (40)",

    "bayesian_optimization_gp_ei":
        "Bayesian Optimization (20)",

    "hyperband":
        "Hyperband",

    "bohb_finite_portfolio":
        "BOHB (finite-portfolio)",

    "probe_npp":
        "Probe-NPP",
}


# Target-task HPO resource cost expressed in
# 200-training-episode full-budget equivalents.
FBE_COST = {
    "zero_sml_ensemble": 0.0,
    "probe_sml_ensemble": 0.0,
    "probe_weighted_extra_trees": 0.0,
    "probe_npp": 0.0,

    "random_search": 40.0,
    "bayesian_optimization_gp_ei": 20.0,
    "hyperband": 16.0,
    "bohb_finite_portfolio": 16.0,

    "global_majority": 0.0,
    "regime_majority": 0.0,
    "dataset_majority": 0.0,
    "dataset_regime_majority": 0.0,

    "validation_selected_oracle": np.nan,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def paired_bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    if n != 120:
        raise RuntimeError(
            f"Expected 120 paired tasks, got {n}."
        )

    rng = np.random.default_rng(
        seed
    )

    # 20k x 120 is small enough and gives
    # exact reproducibility.
    indices = rng.integers(
        0,
        n,
        size=(
            samples,
            n,
        ),
    )

    boot = (
        values[
            indices
        ]
        .mean(
            axis=1
        )
    )

    return (
        float(
            np.quantile(
                boot,
                0.025,
            )
        ),
        float(
            np.quantile(
                boot,
                0.975,
            )
        ),
    )


def load_combined_task_results():
    old = pd.read_csv(
        OLD_RESULTS
    )

    required_old = {
        "method",
        "task_id",
        "dataset",
        "regime",
        "test_accuracy_mean",
    }

    missing = (
        required_old
        - set(
            old.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Old final-test table missing "
            f"columns: {sorted(missing)}"
        )

    old = old[
        old[
            "method"
        ].isin(
            OLD_METHODS
        )
    ].copy()

    if (
        old.groupby(
            "method"
        )[
            "task_id"
        ].nunique()
        .min()
        != 120
    ):
        raise RuntimeError(
            "An existing method does not "
            "contain 120 test tasks."
        )

    # One row per method/task is required.
    if (
        old[
            [
                "method",
                "task_id",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate old method/task rows."
        )

    # Metadata lookup from the already frozen
    # common task set.
    metadata = (
        old[
            [
                "task_id",
                "dataset",
                "regime",
            ]
        ]
        .drop_duplicates()
    )

    if len(metadata) != 120:
        raise RuntimeError(
            "Expected metadata for 120 tasks."
        )

    new = pd.read_csv(
        NEW_PER_TASK
    )

    required_new = {
        "method",
        "task_id",
        "mean_test_accuracy",
        "selection_run_std",
    }

    missing = (
        required_new
        - set(
            new.columns
        )
    )

    if missing:
        raise RuntimeError(
            "New per-task table missing "
            f"columns: {sorted(missing)}"
        )

    if len(new) != 600:
        raise RuntimeError(
            f"Expected 600 new per-task rows, "
            f"got {len(new)}."
        )

    if set(
        new[
            "method"
        ].unique()
    ) != set(
        NEW_METHODS
    ):
        raise RuntimeError(
            "Unexpected reviewer-baseline "
            "method set."
        )

    if (
        new.groupby(
            "method"
        )[
            "task_id"
        ].nunique()
        != 120
    ).any():
        raise RuntimeError(
            "Every new method must have "
            "120 tasks."
        )

    new = new.merge(
        metadata,
        on="task_id",
        how="left",
        validate="many_to_one",
    )

    if (
        new[
            [
                "dataset",
                "regime",
            ]
        ].isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Unable to attach task metadata."
        )

    new = new.rename(
        columns={
            "mean_test_accuracy":
                "test_accuracy_mean",
        }
    )

    old = old[
        [
            "method",
            "task_id",
            "dataset",
            "regime",
            "test_accuracy_mean",
        ]
    ].copy()

    old[
        "selection_run_std"
    ] = np.nan

    new = new[
        [
            "method",
            "task_id",
            "dataset",
            "regime",
            "test_accuracy_mean",
            "selection_run_std",
        ]
    ].copy()

    combined = pd.concat(
        [
            old,
            new,
        ],
        ignore_index=True,
    )

    if (
        combined[
            "test_accuracy_mean"
        ].max()
        > 1.000001
    ):
        raise RuntimeError(
            "Expected test accuracies in [0,1]."
        )

    expected_methods = (
        OLD_METHODS
        + NEW_METHODS
    )

    if set(
        combined[
            "method"
        ].unique()
    ) != set(
        expected_methods
    ):
        raise RuntimeError(
            "Combined method set mismatch."
        )

    counts = (
        combined.groupby(
            "method"
        )[
            "task_id"
        ].nunique()
    )

    if not (
        counts == 120
    ).all():
        raise RuntimeError(
            "Every combined method must "
            "contain 120 tasks."
        )

    return combined


def add_reference_gap(
    combined: pd.DataFrame,
):
    reference = (
        combined[
            combined[
                "method"
            ]
            == "validation_selected_oracle"
        ][
            [
                "task_id",
                "test_accuracy_mean",
            ]
        ]
        .rename(
            columns={
                "test_accuracy_mean":
                    "reference_accuracy",
            }
        )
    )

    if len(reference) != 120:
        raise RuntimeError(
            "Expected 120 reference tasks."
        )

    result = combined.merge(
        reference,
        on="task_id",
        how="left",
        validate="many_to_one",
    )

    # Signed gap: positive means the method
    # is below the validation-selected reference.
    result[
        "signed_gap_to_reference_pp"
    ] = (
        100.0
        * (
            result[
                "reference_accuracy"
            ]
            -
            result[
                "test_accuracy_mean"
            ]
        )
    )

    return result


def method_summary_table(
    combined: pd.DataFrame,
):
    run_summary = pd.read_csv(
        NEW_RUN_SUMMARY
    )

    run_std_map = {}

    for method, group in (
        run_summary.groupby(
            "method"
        )
    ):
        if (
            group[
                "run_seed"
            ].nunique()
            != 5
        ):
            raise RuntimeError(
                f"{method}: expected five "
                "selection runs."
            )

        run_means = (
            group[
                "mean_test_accuracy"
            ]
            .to_numpy(
                dtype=float
            )
        )

        run_std_map[
            method
        ] = (
            100.0
            * float(
                np.std(
                    run_means,
                    ddof=1,
                )
            )
        )

    rows = []

    for method, group in (
        combined.groupby(
            "method",
            sort=False,
        )
    ):
        if len(group) != 120:
            raise RuntimeError(
                f"{method}: expected "
                "120 task rows."
            )

        accuracy_pp = (
            100.0
            * group[
                "test_accuracy_mean"
            ].to_numpy(
                dtype=float
            )
        )

        gap_pp = (
            group[
                "signed_gap_to_reference_pp"
            ]
            .to_numpy(
                dtype=float
            )
        )

        ci_low, ci_high = (
            paired_bootstrap_mean(
                accuracy_pp,

                seed=(
                    BOOTSTRAP_SEED
                    + sum(
                        ord(c)
                        for c
                        in method
                    )
                ),
            )
        )

        rows.append(
            {
                "method":
                    method,

                "display_name":
                    DISPLAY_NAMES[
                        method
                    ],

                "tasks":
                    120,

                "mean_test_accuracy_percent":
                    float(
                        accuracy_pp.mean()
                    ),

                "task_bootstrap_ci95_low_percent":
                    ci_low,

                "task_bootstrap_ci95_high_percent":
                    ci_high,

                "five_run_std_pp":
                    run_std_map.get(
                        method,
                        np.nan,
                    ),

                "mean_signed_gap_to_reference_pp":
                    float(
                        gap_pp.mean()
                    ),

                "median_signed_gap_to_reference_pp":
                    float(
                        np.median(
                            gap_pp
                        )
                    ),

                "p95_signed_gap_to_reference_pp":
                    float(
                        np.quantile(
                            gap_pp,
                            0.95,
                        )
                    ),

                "max_signed_gap_to_reference_pp":
                    float(
                        gap_pp.max()
                    ),

                "target_task_search_cost_fbe":
                    FBE_COST[
                        method
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def paired_comparison(
    combined,
    method_a,
    method_b,
):
    a = (
        combined[
            combined[
                "method"
            ] == method_a
        ][
            [
                "task_id",
                "test_accuracy_mean",
            ]
        ]
        .rename(
            columns={
                "test_accuracy_mean":
                    "a",
            }
        )
    )

    b = (
        combined[
            combined[
                "method"
            ] == method_b
        ][
            [
                "task_id",
                "test_accuracy_mean",
            ]
        ]
        .rename(
            columns={
                "test_accuracy_mean":
                    "b",
            }
        )
    )

    pair = (
        a.merge(
            b,
            on="task_id",
            validate="one_to_one",
        )
        .sort_values(
            "task_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if len(pair) != 120:
        raise RuntimeError(
            "Paired comparison does not "
            "contain 120 tasks."
        )

    difference_pp = (
        100.0
        * (
            pair["a"].to_numpy()
            -
            pair["b"].to_numpy()
        )
    )

    ci_low, ci_high = (
        paired_bootstrap_mean(
            difference_pp,
            seed=(
                BOOTSTRAP_SEED
                + sum(
                    ord(c)
                    for c
                    in (
                        method_a
                        + method_b
                    )
                )
            ),
        )
    )

    tolerance = 1e-12

    wins = int(
        (
            difference_pp
            > tolerance
        ).sum()
    )

    losses = int(
        (
            difference_pp
            < -tolerance
        ).sum()
    )

    ties = (
        len(
            difference_pp
        )
        - wins
        - losses
    )

    if ci_low > 0:
        interpretation = (
            "A higher; CI excludes 0"
        )

    elif ci_high < 0:
        interpretation = (
            "A lower; CI excludes 0"
        )

    else:
        interpretation = (
            "CI includes 0"
        )

    return {
        "method_a":
            method_a,

        "method_b":
            method_b,

        "tasks":
            120,

        "mean_difference_a_minus_b_pp":
            float(
                difference_pp.mean()
            ),

        "median_difference_a_minus_b_pp":
            float(
                np.median(
                    difference_pp
                )
            ),

        "ci95_low_pp":
            ci_low,

        "ci95_high_pp":
            ci_high,

        "a_wins":
            wins,

        "ties":
            ties,

        "a_losses":
            losses,

        "interpretation":
            interpretation,
    }


def breakdown(
    combined,
    grouping_column,
):
    rows = []

    for (
        group_name,
        method,
    ), frame in (
        combined.groupby(
            [
                grouping_column,
                "method",
            ]
        )
    ):
        rows.append(
            {
                grouping_column:
                    group_name,

                "method":
                    method,

                "display_name":
                    DISPLAY_NAMES[
                        method
                    ],

                "tasks":
                    int(
                        len(frame)
                    ),

                "mean_test_accuracy_percent":
                    float(
                        100.0
                        * frame[
                            "test_accuracy_mean"
                        ].mean()
                    ),

                "mean_signed_gap_to_reference_pp":
                    float(
                        frame[
                            "signed_gap_to_reference_pp"
                        ].mean()
                    ),

                "median_signed_gap_to_reference_pp":
                    float(
                        frame[
                            "signed_gap_to_reference_pp"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    combined = (
        load_combined_task_results()
    )

    combined = add_reference_gap(
        combined
    )

    combined_path = (
        OUT
        / "combined_per_task_results.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    summary = (
        method_summary_table(
            combined
        )
    )

    # Paper-oriented order, NOT a performance ranking.
    paper_order = [
        "validation_selected_oracle",

        "zero_sml_ensemble",
        "probe_sml_ensemble",
        "probe_weighted_extra_trees",

        "probe_npp",

        "random_search",
        "bayesian_optimization_gp_ei",
        "hyperband",
        "bohb_finite_portfolio",

        "global_majority",
        "regime_majority",
        "dataset_majority",
        "dataset_regime_majority",
    ]

    order_map = {
        method: index
        for index, method
        in enumerate(
            paper_order
        )
    }

    summary[
        "_order"
    ] = (
        summary[
            "method"
        ].map(
            order_map
        )
    )

    summary = (
        summary.sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary_path = (
        OUT
        / "combined_method_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------
    # Primary paired comparisons.
    # --------------------------------------------------

    comparisons = []

    reviewer_comparators = [
        "random_search",
        "bayesian_optimization_gp_ei",
        "hyperband",
        "bohb_finite_portfolio",
        "probe_npp",
        "validation_selected_oracle",
        "probe_weighted_extra_trees",
    ]

    for focal in [
        "zero_sml_ensemble",
        "probe_sml_ensemble",
    ]:
        for comparator in (
            reviewer_comparators
        ):
            if focal == comparator:
                continue

            comparisons.append(
                paired_comparison(
                    combined,
                    focal,
                    comparator,
                )
            )

    comparisons_df = pd.DataFrame(
        comparisons
    )

    comparisons_path = (
        OUT
        / "paired_bootstrap_comparisons.csv"
    )

    comparisons_df.to_csv(
        comparisons_path,
        index=False,
    )

    dataset_df = breakdown(
        combined,
        "dataset",
    )

    regime_df = breakdown(
        combined,
        "regime",
    )

    dataset_path = (
        OUT
        / "breakdown_by_dataset.csv"
    )

    regime_path = (
        OUT
        / "breakdown_by_regime.csv"
    )

    dataset_df.to_csv(
        dataset_path,
        index=False,
    )

    regime_df.to_csv(
        regime_path,
        index=False,
    )

    # --------------------------------------------------
    # Reviewer-facing compact table.
    # --------------------------------------------------

    core_methods = [
        "zero_sml_ensemble",
        "probe_sml_ensemble",
        "probe_weighted_extra_trees",
        "probe_npp",
        "random_search",
        "bayesian_optimization_gp_ei",
        "hyperband",
        "bohb_finite_portfolio",
        "validation_selected_oracle",
    ]

    core = summary[
        summary[
            "method"
        ].isin(
            core_methods
        )
    ].copy()

    core[
        "accuracy_ci"
    ] = core.apply(
        lambda row: (
            f"{row['mean_test_accuracy_percent']:.3f} "
            f"[{row['task_bootstrap_ci95_low_percent']:.3f}, "
            f"{row['task_bootstrap_ci95_high_percent']:.3f}]"
        ),
        axis=1,
    )

    core_path = (
        OUT
        / "reviewer_core_table.csv"
    )

    core.to_csv(
        core_path,
        index=False,
    )

    metadata = {
        "schema_version":
            1,

        "task_count":
            120,

        "bootstrap_samples":
            BOOTSTRAP_SAMPLES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "primary_inferential_unit":
            "held-out task",

        "new_baseline_task_accuracy_definition":
            (
                "For each task and method, "
                "test accuracy is averaged "
                "across the five frozen "
                "selection/recommendation runs; "
                "each selected configuration "
                "was evaluated using the common "
                "ProtoNet seeds 101/202/303 "
                "and the fixed 600-episode "
                "held-out test bank."
            ),

        "five_run_std_definition":
            (
                "Sample standard deviation of "
                "the five run-level means, each "
                "run-level mean being computed "
                "over all 120 held-out tasks."
            ),

        "signed_gap_definition":
            (
                "Validation-selected reference "
                "test accuracy minus method "
                "test accuracy in percentage "
                "points. Negative gaps are "
                "possible because the reference "
                "was selected on validation, "
                "not on the test bank."
            ),

        "target_task_search_cost":
            (
                "Full-budget equivalent (FBE) "
                "uses 200 training episodes as "
                "one full evaluation. Learned "
                "recommenders have zero target-"
                "task portfolio model evaluations."
            ),
    }

    metadata_path = (
        OUT
        / "analysis_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_paths = [
        combined_path,
        summary_path,
        comparisons_path,
        dataset_path,
        regime_path,
        core_path,
        metadata_path,
    ]

    hash_path = (
        OUT
        / "analysis.sha256"
    )

    hash_path.write_text(
        "".join(
            (
                f"{sha256_file(path)}  "
                f"{path}\n"
            )
            for path
            in output_paths
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Console output.
    # --------------------------------------------------

    print()
    print("=" * 120)
    print(
        "COMBINED HELD-OUT TEST SUMMARY"
    )
    print("=" * 120)

    print(
        summary[
            [
                "display_name",
                "mean_test_accuracy_percent",
                "task_bootstrap_ci95_low_percent",
                "task_bootstrap_ci95_high_percent",
                "five_run_std_pp",
                "mean_signed_gap_to_reference_pp",
                "target_task_search_cost_fbe",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print("=" * 120)
    print(
        "ZERO-SML / PROBE-SML "
        "PAIRED COMPARISONS"
    )
    print("=" * 120)

    printable = (
        comparisons_df.copy()
    )

    printable[
        "method_a"
    ] = printable[
        "method_a"
    ].map(
        DISPLAY_NAMES
    )

    printable[
        "method_b"
    ] = printable[
        "method_b"
    ].map(
        DISPLAY_NAMES
    )

    print(
        printable.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "Combined per-task rows:",
        len(
            combined
        ),
    )

    print(
        "Methods:",
        combined[
            "method"
        ].nunique(),
    )

    print(
        "Tasks/method:",
        combined.groupby(
            "method"
        )[
            "task_id"
        ].nunique().unique(),
    )

    print()
    print(
        "REVIEWER BASELINE ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()
