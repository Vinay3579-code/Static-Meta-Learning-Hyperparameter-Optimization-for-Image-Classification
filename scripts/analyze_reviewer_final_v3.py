from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Inputs
# ============================================================

LEGACY_RESULTS = Path(
    "results/final_test_eval_v2/"
    "all_method_test_results.csv"
)

HPO_ROOT = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation"
)

SML_ROOT = Path(
    "results/meta_learning/"
    "five_run_robustness_v2/"
    "final_test_evaluation"
)

OUT = Path(
    "results/reviewer_final_analysis_v3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260828


# ============================================================
# Canonical method names
# ============================================================

LEGACY_METHODS = [
    "validation_selected_oracle",
    "probe_weighted_extra_trees",
    "global_majority",
    "regime_majority",
    "dataset_majority",
    "dataset_regime_majority",
]

HPO_METHODS = [
    "random_search",
    "bayesian_optimization_gp_ei",
    "hyperband",
    "bohb_finite_portfolio",
    "probe_npp",
]

SML_RENAME = {
    "zero_sml":
        "zero_sml_five_run",

    "probe_sml":
        "probe_sml_five_run",
}


DISPLAY_NAMES = {
    "zero_sml_five_run":
        "Zero-SML (ours)",

    "probe_sml_five_run":
        "Probe-SML (ours)",

    "probe_npp":
        "Probe-NPP",

    "random_search":
        "Random Search (40)",

    "bayesian_optimization_gp_ei":
        "Bayesian Optimization (20)",

    "hyperband":
        "Hyperband",

    "bohb_finite_portfolio":
        "BOHB (finite-portfolio)",

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
}


# Full-budget-equivalent target-task HPO cost.
#
# IMPORTANT:
# For learned recommenders, 0 means ZERO target-task
# portfolio model evaluations before recommendation.
# It does NOT mean descriptor extraction or meta-training
# has zero computational cost.
SEARCH_COST_FBE = {
    "zero_sml_five_run":
        0.0,

    "probe_sml_five_run":
        0.0,

    "probe_npp":
        0.0,

    "probe_weighted_extra_trees":
        0.0,

    "random_search":
        40.0,

    "bayesian_optimization_gp_ei":
        20.0,

    "hyperband":
        16.0,

    "bohb_finite_portfolio":
        16.0,

    "global_majority":
        0.0,

    "regime_majority":
        0.0,

    "dataset_majority":
        0.0,

    "dataset_regime_majority":
        0.0,

    # 64 seed-101 screen jobs +
    # 2 confirmation seeds × mean shortlist 4.55.
    "validation_selected_oracle":
        73.1,
}


METHOD_TYPE = {
    "zero_sml_five_run":
        "Amortized direct recommendation",

    "probe_sml_five_run":
        "Amortized direct recommendation",

    "probe_npp":
        "Learned performance ranking",

    "random_search":
        "Iterative search",

    "bayesian_optimization_gp_ei":
        "Sequential model-based search",

    "hyperband":
        "Multi-fidelity search",

    "bohb_finite_portfolio":
        "Model-based multi-fidelity search",

    "probe_weighted_extra_trees":
        "Classical amortized recommender",

    "global_majority":
        "Static default",

    "regime_majority":
        "Static default",

    "dataset_majority":
        "Privileged static default",

    "dataset_regime_majority":
        "Privileged static default",

    "validation_selected_oracle":
        "Validation-selected reference",
}


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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


def stable_seed(
    text: str,
) -> int:

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return (
        BOOTSTRAP_SEED
        + int.from_bytes(
            digest[:4],
            "big",
        )
    ) % (2**32 - 1)


def bootstrap_mean_ci(
    values,
    *,
    name,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) != 120:
        raise RuntimeError(
            f"{name}: expected 120 tasks; "
            f"observed {len(values)}"
        )

    rng = np.random.default_rng(
        stable_seed(
            name
        )
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAP_SAMPLES,
            len(values),
        ),
    )

    samples = (
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
                samples,
                0.025,
            )
        ),
        float(
            np.quantile(
                samples,
                0.975,
            )
        ),
    )


# ============================================================
# Load task metadata/reference/static baselines
# ============================================================

def load_legacy():
    df = pd.read_csv(
        LEGACY_RESULTS
    )

    required = {
        "method",
        "task_id",
        "dataset",
        "regime",
        "test_accuracy_mean",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Legacy results missing: "
            f"{sorted(missing)}"
        )

    df = df[
        df[
            "method"
        ].isin(
            LEGACY_METHODS
        )
    ].copy()

    for method in LEGACY_METHODS:

        sub = df[
            df[
                "method"
            ] == method
        ]

        if len(sub) != 120:
            raise RuntimeError(
                f"{method}: expected "
                f"120 rows, found {len(sub)}"
            )

        if sub[
            "task_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{method}: duplicate tasks."
            )

    return df[
        [
            "method",
            "task_id",
            "dataset",
            "regime",
            "test_accuracy_mean",
        ]
    ].copy()


# ============================================================
# Reviewer HPO baselines
# ============================================================

def load_hpo():

    per_task = pd.read_csv(
        HPO_ROOT
        / "per_task_method_results.csv"
    )

    run_summary = pd.read_csv(
        HPO_ROOT
        / "run_summary.csv"
    )

    required = {
        "method",
        "task_id",
        "mean_test_accuracy",
        "selection_run_std",
    }

    missing = (
        required
        - set(
            per_task.columns
        )
    )

    if missing:
        raise RuntimeError(
            "HPO per-task results missing: "
            f"{sorted(missing)}"
        )

    if set(
        per_task[
            "method"
        ].unique()
    ) != set(
        HPO_METHODS
    ):
        raise RuntimeError(
            "Unexpected HPO method set."
        )

    if len(per_task) != (
        5 * 120
    ):
        raise RuntimeError(
            "Expected 600 HPO per-task rows."
        )

    return (
        per_task,
        run_summary,
    )


# ============================================================
# New five-run SML
# ============================================================

def load_sml():

    per_task = pd.read_csv(
        SML_ROOT
        / "per_task_method_results.csv"
    )

    run_summary = pd.read_csv(
        SML_ROOT
        / "run_summary.csv"
    )

    if len(per_task) != 240:
        raise RuntimeError(
            "Expected 240 five-run SML "
            "per-task rows."
        )

    if set(
        per_task[
            "method"
        ].unique()
    ) != {
        "zero_sml",
        "probe_sml",
    }:
        raise RuntimeError(
            "Unexpected SML method set."
        )

    per_task[
        "method"
    ] = per_task[
        "method"
    ].replace(
        SML_RENAME
    )

    run_summary[
        "method"
    ] = run_summary[
        "method"
    ].replace(
        SML_RENAME
    )

    return (
        per_task,
        run_summary,
    )


# ============================================================
# Build common 120-task table
# ============================================================

def build_combined():

    legacy = load_legacy()

    (
        hpo,
        hpo_runs,
    ) = load_hpo()

    (
        sml,
        sml_runs,
    ) = load_sml()

    metadata = (
        legacy[
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
            "Expected metadata for "
            "120 held-out tasks."
        )

    hpo = hpo.merge(
        metadata,
        on="task_id",
        how="left",
        validate="many_to_one",
    )

    sml = sml.merge(
        metadata,
        on="task_id",
        how="left",
        validate="many_to_one",
    )

    hpo = hpo.rename(
        columns={
            "mean_test_accuracy":
                "test_accuracy_mean",
        }
    )

    sml = sml.rename(
        columns={
            "mean_test_accuracy":
                "test_accuracy_mean",
        }
    )

    hpo = hpo[
        [
            "method",
            "task_id",
            "dataset",
            "regime",
            "test_accuracy_mean",
            "selection_run_std",
        ]
    ].copy()

    sml = sml[
        [
            "method",
            "task_id",
            "dataset",
            "regime",
            "test_accuracy_mean",
            "selection_run_std",
        ]
    ].copy()

    legacy[
        "selection_run_std"
    ] = np.nan

    combined = pd.concat(
        [
            legacy,
            sml,
            hpo,
        ],
        ignore_index=True,
    )

    expected_methods = (
        set(
            LEGACY_METHODS
        )
        | set(
            HPO_METHODS
        )
        | set(
            SML_RENAME.values()
        )
    )

    observed_methods = set(
        combined[
            "method"
        ].unique()
    )

    if (
        observed_methods
        != expected_methods
    ):
        raise RuntimeError(
            "Combined method-set mismatch.\n"
            f"Observed: "
            f"{sorted(observed_methods)}\n"
            f"Expected: "
            f"{sorted(expected_methods)}"
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
            "Every method must contain "
            "120 held-out tasks."
        )

    # Attach reference accuracy.
    ref = (
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

    combined = combined.merge(
        ref,
        on="task_id",
        how="left",
        validate="many_to_one",
    )

    combined[
        "signed_gap_to_reference_pp"
    ] = (
        100.0
        * (
            combined[
                "reference_accuracy"
            ]
            -
            combined[
                "test_accuracy_mean"
            ]
        )
    )

    runs = pd.concat(
        [
            sml_runs,
            hpo_runs,
        ],
        ignore_index=True,
    )

    return (
        combined,
        runs,
    )


# ============================================================
# Method summary
# ============================================================

def make_method_summary(
    combined,
    run_summary,
):

    run_sd = {}

    for method, frame in (
        run_summary.groupby(
            "method"
        )
    ):

        if frame[
            "run_seed"
        ].nunique() != 5:
            raise RuntimeError(
                f"{method}: expected "
                "five runs."
            )

        values = frame[
            "mean_test_accuracy"
        ].to_numpy(
            dtype=float
        )

        run_sd[
            method
        ] = (
            100.0
            * np.std(
                values,
                ddof=1,
            )
        )

    rows = []

    for method, frame in (
        combined.groupby(
            "method"
        )
    ):

        values_pp = (
            100.0
            * frame[
                "test_accuracy_mean"
            ].to_numpy(
                dtype=float
            )
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                values_pp,
                name=(
                    "accuracy::"
                    + method
                ),
            )
        )

        gaps = frame[
            "signed_gap_to_reference_pp"
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "method":
                    method,

                "display_name":
                    DISPLAY_NAMES[
                        method
                    ],

                "method_type":
                    METHOD_TYPE[
                        method
                    ],

                "tasks":
                    120,

                "mean_test_accuracy_percent":
                    float(
                        values_pp.mean()
                    ),

                "five_run_std_pp":
                    run_sd.get(
                        method,
                        np.nan,
                    ),

                "task_bootstrap_ci95_low_percent":
                    ci_low,

                "task_bootstrap_ci95_high_percent":
                    ci_high,

                "mean_signed_gap_to_reference_pp":
                    float(
                        gaps.mean()
                    ),

                "median_signed_gap_to_reference_pp":
                    float(
                        np.median(
                            gaps
                        )
                    ),

                "p95_signed_gap_to_reference_pp":
                    float(
                        np.quantile(
                            gaps,
                            0.95,
                        )
                    ),

                "max_signed_gap_to_reference_pp":
                    float(
                        gaps.max()
                    ),

                "target_task_search_cost_fbe":
                    SEARCH_COST_FBE[
                        method
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Paired task comparison
# ============================================================

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
                    "accuracy_a",
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
                    "accuracy_b",
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

    difference = (
        100.0
        * (
            pair[
                "accuracy_a"
            ].to_numpy()
            -
            pair[
                "accuracy_b"
            ].to_numpy()
        )
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            difference,
            name=(
                "paired::"
                + method_a
                + "::"
                + method_b
            ),
        )
    )

    eps = 1e-12

    wins = int(
        (
            difference > eps
        ).sum()
    )

    losses = int(
        (
            difference < -eps
        ).sum()
    )

    ties = (
        len(difference)
        - wins
        - losses
    )

    if ci_low > 0:
        conclusion = (
            "A higher; CI excludes 0"
        )

    elif ci_high < 0:
        conclusion = (
            "A lower; CI excludes 0"
        )

    else:
        conclusion = (
            "CI includes 0"
        )

    return {
        "method_a":
            method_a,

        "method_b":
            method_b,

        "display_a":
            DISPLAY_NAMES[
                method_a
            ],

        "display_b":
            DISPLAY_NAMES[
                method_b
            ],

        "tasks":
            120,

        "mean_difference_a_minus_b_pp":
            float(
                difference.mean()
            ),

        "median_difference_a_minus_b_pp":
            float(
                np.median(
                    difference
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
            conclusion,
    }


# ============================================================
# Dataset/regime breakdown
# ============================================================

def breakdown(
    combined,
    grouping,
):

    rows = []

    for (
        group_value,
        method,
    ), frame in (
        combined.groupby(
            [
                grouping,
                "method",
            ]
        )
    ):

        rows.append(
            {
                grouping:
                    group_value,

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
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    combined, run_summary = (
        build_combined()
    )

    combined_path = (
        OUT
        / "combined_per_task_results.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    run_path = (
        OUT
        / "five_run_method_results.csv"
    )

    run_summary.to_csv(
        run_path,
        index=False,
    )

    summary = make_method_summary(
        combined,
        run_summary,
    )

    # Logical manuscript order.
    # This is not an accuracy ranking.
    paper_order = [
        "zero_sml_five_run",
        "probe_sml_five_run",
        "probe_npp",
        "probe_weighted_extra_trees",
        "random_search",
        "bayesian_optimization_gp_ei",
        "hyperband",
        "bohb_finite_portfolio",
        "global_majority",
        "regime_majority",
        "dataset_majority",
        "dataset_regime_majority",
        "validation_selected_oracle",
    ]

    order = {
        method: i
        for i, method
        in enumerate(
            paper_order
        )
    }

    summary[
        "_order"
    ] = summary[
        "method"
    ].map(
        order
    )

    summary = (
        summary.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    summary_path = (
        OUT
        / "final_method_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Primary comparisons
    # --------------------------------------------------------

    comparators = [
        "probe_sml_five_run",
        "probe_npp",
        "probe_weighted_extra_trees",
        "random_search",
        "bayesian_optimization_gp_ei",
        "hyperband",
        "bohb_finite_portfolio",
        "global_majority",
        "regime_majority",
        "dataset_majority",
        "dataset_regime_majority",
        "validation_selected_oracle",
    ]

    comparisons = []

    for comparator in comparators:

        comparisons.append(
            paired_comparison(
                combined,
                "zero_sml_five_run",
                comparator,
            )
        )

    probe_comparators = [
        "probe_npp",
        "probe_weighted_extra_trees",
        "random_search",
        "bayesian_optimization_gp_ei",
        "hyperband",
        "bohb_finite_portfolio",
        "validation_selected_oracle",
    ]

    for comparator in (
        probe_comparators
    ):

        comparisons.append(
            paired_comparison(
                combined,
                "probe_sml_five_run",
                comparator,
            )
        )

    comparisons_df = (
        pd.DataFrame(
            comparisons
        )
    )

    comparison_path = (
        OUT
        / "paired_bootstrap_comparisons.csv"
    )

    comparisons_df.to_csv(
        comparison_path,
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

    # --------------------------------------------------------
    # Compact manuscript table
    # --------------------------------------------------------

    core_methods = [
        "zero_sml_five_run",
        "probe_sml_five_run",
        "probe_npp",
        "probe_weighted_extra_trees",
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

    def mean_sd_string(row):

        if pd.isna(
            row[
                "five_run_std_pp"
            ]
        ):
            return (
                f"{row['mean_test_accuracy_percent']:.3f}"
            )

        return (
            f"{row['mean_test_accuracy_percent']:.3f} "
            f"± "
            f"{row['five_run_std_pp']:.3f}"
        )

    core[
        "accuracy_mean_std_percent"
    ] = core.apply(
        mean_sd_string,
        axis=1,
    )

    core[
        "task_bootstrap_ci95_percent"
    ] = core.apply(
        lambda r: (
            "["
            f"{r['task_bootstrap_ci95_low_percent']:.3f}, "
            f"{r['task_bootstrap_ci95_high_percent']:.3f}"
            "]"
        ),
        axis=1,
    )

    core_path = (
        OUT
        / "manuscript_core_table.csv"
    )

    core.to_csv(
        core_path,
        index=False,
    )

    metadata = {
        "schema_version":
            3,

        "held_out_tasks":
            120,

        "selection_runs":
            5,

        "final_protonet_seeds":
            [
                101,
                202,
                303,
            ],

        "training_episodes":
            200,

        "test_episodes":
            600,

        "paired_bootstrap_samples":
            BOOTSTRAP_SAMPLES,

        "paired_bootstrap_seed":
            BOOTSTRAP_SEED,

        "primary_inferential_unit":
            "held-out task",

        "accuracy_for_five_run_methods":
            (
                "Per-task accuracy averaged "
                "across five independently "
                "seeded recommendation/search "
                "runs; each selected "
                "configuration is evaluated "
                "with the common three-seed "
                "ProtoNet test protocol."
            ),

        "five_run_std":
            (
                "Sample standard deviation "
                "of the five run-level mean "
                "accuracies over the same "
                "120 held-out tasks."
            ),

        "task_bootstrap_ci":
            (
                "20,000-sample paired/task "
                "bootstrap over the 120 "
                "held-out tasks."
            ),

        "signed_gap":
            (
                "Validation-selected reference "
                "accuracy minus method accuracy "
                "in percentage points."
            ),

        "search_cost_note":
            (
                "FBE is target-task HPO search "
                "cost only. A zero value for "
                "SML/NPP does not imply zero "
                "descriptor-extraction or "
                "offline meta-training cost."
            ),

        "reference_cost_note":
            (
                "Validation-selected reference "
                "cost is 64 one-seed screening "
                "fits plus two confirmation "
                "seeds times the observed mean "
                "shortlist size 4.55 = 73.1 "
                "full-budget model-seed fits "
                "per held-out task."
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

    output_files = [
        combined_path,
        run_path,
        summary_path,
        comparison_path,
        dataset_path,
        regime_path,
        core_path,
        metadata_path,
    ]

    hash_path = (
        OUT
        / "final_analysis.sha256"
    )

    hash_path.write_text(
        "".join(
            (
                f"{sha256_file(path)}  "
                f"{path}\n"
            )
            for path
            in output_files
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Console report
    # ========================================================

    print()
    print("=" * 125)
    print(
        "FINAL FIVE-RUN REVIEWER COMPARISON"
    )
    print("=" * 125)

    show = summary[
        [
            "display_name",
            "mean_test_accuracy_percent",
            "five_run_std_pp",
            "task_bootstrap_ci95_low_percent",
            "task_bootstrap_ci95_high_percent",
            "mean_signed_gap_to_reference_pp",
            "target_task_search_cost_fbe",
        ]
    ]

    print(
        show.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print("=" * 125)
    print(
        "FINAL PAIRED TASK-BOOTSTRAP COMPARISONS"
    )
    print("=" * 125)

    print(
        comparisons_df[
            [
                "display_a",
                "display_b",
                "mean_difference_a_minus_b_pp",
                "ci95_low_pp",
                "ci95_high_pp",
                "a_wins",
                "ties",
                "a_losses",
                "interpretation",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "Combined rows:",
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
        sorted(
            combined.groupby(
                "method"
            )[
                "task_id"
            ].nunique()
            .unique()
            .tolist()
        ),
    )

    print()
    print(
        "FINAL REVIEWER ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()
