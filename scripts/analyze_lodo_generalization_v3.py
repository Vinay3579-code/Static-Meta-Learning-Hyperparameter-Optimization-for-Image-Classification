from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Frozen analysis constants
# ============================================================

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260828

DATASETS = [
    "omniglot",
    "cifar100",
    "miniimagenet",
    "dtd",
    "flowers102",
]

DATASET_DISPLAY = {
    "omniglot":
        "Omniglot held out",

    "cifar100":
        "CIFAR-100 held out",

    "miniimagenet":
        "miniImageNet held out",

    "dtd":
        "DTD held out",

    "flowers102":
        "Flowers102 held out",
}

METHOD_ORDER = [
    "random_search",
    "bayesian_optimization_gp_ei",
    "hyperband",
    "bohb_finite_portfolio",
    "probe_npp_lodo",
    "zero_sml_lodo",
]

METHOD_DISPLAY = {
    "random_search":
        "Random Search (40)",

    "bayesian_optimization_gp_ei":
        "Bayesian Optimization (20)",

    "hyperband":
        "Hyperband",

    "bohb_finite_portfolio":
        "BOHB finite portfolio",

    "probe_npp_lodo":
        "LODO Probe-NPP",

    "zero_sml_lodo":
        "LODO Zero-SML (ours)",
}

SEARCH_COST_FBE = {
    "random_search": 40.0,
    "bayesian_optimization_gp_ei": 20.0,
    "hyperband": 16.0,
    "bohb_finite_portfolio": 16.0,
    "probe_npp_lodo": 0.0,
    "zero_sml_lodo": 0.0,
}


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "results/lodo_generalization_v3"
)

EVAL_ROOT = (
    ROOT
    / "final_test_evaluation"
)

ZERO_REC = (
    ROOT
    / "zero_sml"
    / "locked_recommendations.csv"
)

NPP_REC = (
    ROOT
    / "probe_npp"
    / "locked_recommendations.csv"
)

OLD_CACHE = (
    EVAL_ROOT
    / "existing_pair_cache_index.csv"
)

FRESH_CACHE_ROOT = (
    EVAL_ROOT
    / "fresh_cache"
)

HPO_RESULTS = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation/"
    "recommendation_test_results.csv"
)

TEST_DESCRIPTORS = Path(
    "results/descriptors/final800/"
    "merged/test_full84.csv"
)

OUTPUT = (
    ROOT
    / "final_analysis"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# General helpers
# ============================================================

def normalize_dataset(value) -> str:
    value = (
        str(value)
        .strip()
        .lower()
    )

    aliases = {
        "cifar-100":
            "cifar100",

        "cifar_100":
            "cifar100",

        "mini-imagenet":
            "miniimagenet",

        "mini_imagenet":
            "miniimagenet",

        "flowers-102":
            "flowers102",

        "flowers_102":
            "flowers102",
    }

    return aliases.get(
        value,
        value,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def first_existing_column(
    df: pd.DataFrame,
    candidates,
):
    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError(
        "None of these columns exist: "
        f"{candidates}\n"
        f"Available columns: "
        f"{df.columns.tolist()}"
    )


def to_percent(
    series: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="raise",
    ).astype(float)

    if values.max() <= 1.5:
        return values * 100.0

    return values


def canonical_hpo_method(
    value,
):
    name = (
        str(value)
        .strip()
        .lower()
    )

    if (
        "bohb"
        in name
    ):
        return (
            "bohb_finite_portfolio"
        )

    if (
        "hyperband"
        in name
    ):
        return "hyperband"

    if (
        "random"
        in name
    ):
        return "random_search"

    if (
        "bayesian"
        in name
        or "gp_ei"
        in name
        or name
        in {
            "bo",
            "bo_gp_ei",
        }
    ):
        return (
            "bayesian_optimization_gp_ei"
        )

    # Deliberately exclude old NPP:
    # we are replacing it with strict LODO-NPP.
    return None


# ============================================================
# 1. Verify all source artifacts
# ============================================================

required_files = [
    ZERO_REC,
    NPP_REC,
    OLD_CACHE,
    HPO_RESULTS,
    TEST_DESCRIPTORS,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

if not FRESH_CACHE_ROOT.exists():
    raise FileNotFoundError(
        FRESH_CACHE_ROOT
    )


# ============================================================
# 2. Task -> dataset mapping
# ============================================================

descriptor_df = pd.read_csv(
    TEST_DESCRIPTORS
)

descriptor_df[
    "task_id"
] = (
    descriptor_df[
        "task_id"
    ].astype(str)
)

descriptor_df[
    "dataset"
] = (
    descriptor_df[
        "dataset"
    ].map(
        normalize_dataset
    )
)

if len(
    descriptor_df
) != 120:
    raise RuntimeError(
        "Expected exactly 120 held-out tasks."
    )

if descriptor_df[
    "task_id"
].duplicated().any():
    raise RuntimeError(
        "Duplicate held-out task IDs."
    )

task_dataset = dict(
    zip(
        descriptor_df[
            "task_id"
        ],
        descriptor_df[
            "dataset"
        ],
    )
)

if set(
    descriptor_df[
        "dataset"
    ].unique()
) != set(DATASETS):
    raise RuntimeError(
        "Unexpected test datasets."
    )


# ============================================================
# 3. Load old compatible final-test cache
# ============================================================

old_cache = pd.read_csv(
    OLD_CACHE
)

required_old = {
    "task_id",
    "config_id",
    "test_accuracy_seed101",
    "test_accuracy_seed202",
    "test_accuracy_seed303",
    "test_accuracy_mean",
}

missing = (
    required_old
    - set(
        old_cache.columns
    )
)

if missing:
    raise RuntimeError(
        "Old cache missing columns: "
        f"{sorted(missing)}"
    )

old_cache[
    "task_id"
] = (
    old_cache[
        "task_id"
    ].astype(str)
)

old_cache[
    "config_id"
] = (
    old_cache[
        "config_id"
    ].astype(str)
)

old_cache[
    "pair_key"
] = (
    old_cache[
        "task_id"
    ]
    + "||"
    + old_cache[
        "config_id"
    ]
)

old_cache[
    "evaluation_origin"
] = "existing_cache"


# ============================================================
# 4. Load the 88 fresh final-test evaluations
# ============================================================

fresh_rows = []

fresh_files = sorted(
    FRESH_CACHE_ROOT.glob(
        "*/*.json"
    )
)

for path in fresh_files:

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    task_id = str(
        payload[
            "task_id"
        ]
    )

    config_id = str(
        payload[
            "config_id"
        ]
    )

    if int(
        payload[
            "training_episodes"
        ]
    ) != 200:
        raise RuntimeError(
            f"{path}: wrong train budget."
        )

    if int(
        payload[
            "test_episodes"
        ]
    ) != 600:
        raise RuntimeError(
            f"{path}: wrong test budget."
        )

    seeds = [
        int(x)
        for x in payload[
            "model_seeds"
        ]
    ]

    if seeds != [
        101,
        202,
        303,
    ]:
        raise RuntimeError(
            f"{path}: wrong model seeds."
        )

    accuracies = [
        float(x)
        for x in payload[
            "test_accuracies_by_seed"
        ]
    ]

    if len(
        accuracies
    ) != 3:
        raise RuntimeError(
            f"{path}: wrong accuracy count."
        )

    mean_accuracy = float(
        payload[
            "test_accuracy_mean"
        ]
    )

    if not np.isclose(
        np.mean(
            accuracies
        ),
        mean_accuracy,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{path}: test mean mismatch."
        )

    fresh_rows.append(
        {
            "task_id":
                task_id,

            "config_id":
                config_id,

            "pair_key":
                (
                    task_id
                    + "||"
                    + config_id
                ),

            "test_accuracy_seed101":
                accuracies[0],

            "test_accuracy_seed202":
                accuracies[1],

            "test_accuracy_seed303":
                accuracies[2],

            "test_accuracy_mean":
                mean_accuracy,

            "evaluation_origin":
                "fresh_lodo_v3",

            "cache_file":
                str(path),
        }
    )

fresh_cache = pd.DataFrame(
    fresh_rows
)

if len(
    fresh_cache
) != 88:
    raise RuntimeError(
        "Expected exactly 88 fresh "
        "LODO evaluations, observed "
        f"{len(fresh_cache)}."
    )

if fresh_cache[
    "pair_key"
].duplicated().any():
    raise RuntimeError(
        "Duplicate fresh evaluation."
    )


# ============================================================
# 5. Merge reusable + fresh pair cache
# ============================================================

cache_columns = [
    "task_id",
    "config_id",
    "pair_key",
    "test_accuracy_seed101",
    "test_accuracy_seed202",
    "test_accuracy_seed303",
    "test_accuracy_mean",
    "evaluation_origin",
]

pair_cache = pd.concat(
    [
        old_cache[
            cache_columns
        ],
        fresh_cache[
            cache_columns
        ],
    ],
    ignore_index=True,
)

# Verify duplicates, if any, numerically agree.
duplicate_groups = (
    pair_cache[
        pair_cache[
            "pair_key"
        ].duplicated(
            keep=False
        )
    ]
    .groupby(
        "pair_key"
    )
)

for pair_key, group in duplicate_groups:

    for column in [
        "test_accuracy_seed101",
        "test_accuracy_seed202",
        "test_accuracy_seed303",
        "test_accuracy_mean",
    ]:

        values = (
            group[
                column
            ]
            .astype(float)
            .to_numpy()
        )

        if not np.allclose(
            values,
            values[0],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Inconsistent duplicate cache "
                f"entry: {pair_key}, "
                f"{column}"
            )

pair_cache = (
    pair_cache
    .sort_values(
        [
            "pair_key",
            "evaluation_origin",
        ],
        kind="mergesort",
    )
    .drop_duplicates(
        subset=[
            "pair_key",
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 6. Load locked LODO recommendations
# ============================================================

def load_locked_recommendations(
    path: Path,
    method: str,
):
    df = pd.read_csv(
        path
    )

    required = {
        "task_id",
        "heldout_dataset",
        "run_seed",
        "selected_config_id",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing "
            f"{sorted(missing)}"
        )

    result = df[
        [
            "task_id",
            "heldout_dataset",
            "run_seed",
            "selected_config_id",
        ]
    ].copy()

    result[
        "task_id"
    ] = (
        result[
            "task_id"
        ].astype(str)
    )

    result[
        "selected_config_id"
    ] = (
        result[
            "selected_config_id"
        ].astype(str)
    )

    result[
        "heldout_dataset"
    ] = (
        result[
            "heldout_dataset"
        ].map(
            normalize_dataset
        )
    )

    result[
        "run_seed"
    ] = (
        result[
            "run_seed"
        ].astype(int)
    )

    result[
        "method"
    ] = method

    result[
        "pair_key"
    ] = (
        result[
            "task_id"
        ]
        + "||"
        + result[
            "selected_config_id"
        ]
    )

    if len(
        result
    ) != 600:
        raise RuntimeError(
            f"{method}: expected 600 rows, "
            f"observed {len(result)}."
        )

    return result


zero_rec = (
    load_locked_recommendations(
        ZERO_REC,
        "zero_sml_lodo",
    )
)

npp_rec = (
    load_locked_recommendations(
        NPP_REC,
        "probe_npp_lodo",
    )
)

learned_rec = pd.concat(
    [
        zero_rec,
        npp_rec,
    ],
    ignore_index=True,
)


# ============================================================
# 7. Domain-exclusion / identity audits
# ============================================================

for row in (
    learned_rec.itertuples(
        index=False
    )
):
    actual_dataset = (
        task_dataset[
            str(
                row.task_id
            )
        ]
    )

    if (
        actual_dataset
        != row.heldout_dataset
    ):
        raise RuntimeError(
            "Held-out dataset mismatch: "
            f"{row.task_id}"
        )


# ============================================================
# 8. Determine exact required unique learned pairs
# ============================================================

required_pairs = (
    learned_rec[
        [
            "pair_key",
            "task_id",
            "selected_config_id",
        ]
    ]
    .drop_duplicates(
        subset=[
            "pair_key",
        ]
    )
)

if len(
    required_pairs
) != 405:
    raise RuntimeError(
        "Expected exactly 405 unique "
        "LODO task/config pairs, observed "
        f"{len(required_pairs)}."
    )

cache_coverage = (
    required_pairs.merge(
        pair_cache[
            [
                "pair_key",
                "test_accuracy_mean",
            ]
        ],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
)

missing_after_eval = int(
    cache_coverage[
        "test_accuracy_mean"
    ].isna().sum()
)

if missing_after_eval != 0:
    raise RuntimeError(
        "Incomplete LODO final cache: "
        f"{missing_after_eval} pairs missing."
    )


# ============================================================
# 9. Attach final-test results to 1,200 learned recs
# ============================================================

learned_results = (
    learned_rec.merge(
        pair_cache[
            [
                "pair_key",
                "test_accuracy_seed101",
                "test_accuracy_seed202",
                "test_accuracy_seed303",
                "test_accuracy_mean",
                "evaluation_origin",
            ]
        ],
        on="pair_key",
        how="left",
        validate="many_to_one",
    )
)

if learned_results[
    "test_accuracy_mean"
].isna().any():
    raise RuntimeError(
        "Missing learned-method test result."
    )

learned_results[
    "test_accuracy_percent"
] = to_percent(
    learned_results[
        "test_accuracy_mean"
    ]
)

learned_results[
    "dataset"
] = (
    learned_results[
        "heldout_dataset"
    ]
)

learned_results.to_csv(
    OUTPUT
    / "lodo_learned_recommendation_test_results.csv",
    index=False,
)


# ============================================================
# 10. Load target-local HPO methods
# ============================================================

hpo = pd.read_csv(
    HPO_RESULTS
)

method_col = (
    first_existing_column(
        hpo,
        [
            "method",
            "method_id",
            "baseline",
        ],
    )
)

run_col = (
    first_existing_column(
        hpo,
        [
            "run_seed",
            "search_seed",
            "seed",
        ],
    )
)

task_col = (
    first_existing_column(
        hpo,
        [
            "task_id",
        ],
    )
)

accuracy_col = (
    first_existing_column(
        hpo,
        [
            "test_accuracy_mean",
            "mean_test_accuracy",
        ],
    )
)

hpo[
    "_canonical_method"
] = (
    hpo[
        method_col
    ].map(
        canonical_hpo_method
    )
)

hpo = (
    hpo[
        hpo[
            "_canonical_method"
        ].notna()
    ]
    .copy()
)

hpo[
    "method"
] = (
    hpo[
        "_canonical_method"
    ]
)

hpo[
    "run_seed"
] = (
    hpo[
        run_col
    ].astype(int)
)

hpo[
    "task_id"
] = (
    hpo[
        task_col
    ].astype(str)
)

hpo[
    "dataset"
] = (
    hpo[
        "task_id"
    ].map(
        task_dataset
    )
)

if hpo[
    "dataset"
].isna().any():
    unknown = (
        hpo.loc[
            hpo[
                "dataset"
            ].isna(),
            "task_id",
        ]
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "HPO rows have unknown task IDs: "
        f"{unknown[:10]}"
    )

hpo[
    "test_accuracy_percent"
] = to_percent(
    hpo[
        accuracy_col
    ]
)

hpo_results = hpo[
    [
        "method",
        "run_seed",
        "task_id",
        "dataset",
        "test_accuracy_percent",
    ]
].copy()


# ============================================================
# 11. Standardize learned rows
# ============================================================

learned_standard = (
    learned_results[
        [
            "method",
            "run_seed",
            "task_id",
            "dataset",
            "test_accuracy_percent",
        ]
    ]
    .copy()
)


# ============================================================
# 12. Six-method master table
# ============================================================

all_results = pd.concat(
    [
        hpo_results,
        learned_standard,
    ],
    ignore_index=True,
)

all_results = (
    all_results[
        all_results[
            "method"
        ].isin(
            METHOD_ORDER
        )
    ]
    .copy()
)

# Strict row count:
# six methods × five runs × 120 tasks.
expected_rows = (
    6
    * 5
    * 120
)

if len(
    all_results
) != expected_rows:
    print()
    print(
        "Observed rows by method:"
    )

    print(
        all_results[
            "method"
        ]
        .value_counts()
        .to_string()
    )

    raise RuntimeError(
        "Expected exactly "
        f"{expected_rows} rows, observed "
        f"{len(all_results)}."
    )

if all_results[
    [
        "method",
        "run_seed",
        "task_id",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate method/run/task rows."
    )

for method in METHOD_ORDER:

    subset = (
        all_results[
            all_results[
                "method"
            ]
            == method
        ]
    )

    if len(subset) != 600:
        raise RuntimeError(
            f"{method}: expected 600 rows."
        )

    if subset[
        "run_seed"
    ].nunique() != 5:
        raise RuntimeError(
            f"{method}: expected 5 runs."
        )

    if subset[
        "task_id"
    ].nunique() != 120:
        raise RuntimeError(
            f"{method}: expected 120 tasks."
        )

    dataset_counts = (
        subset[
            [
                "task_id",
                "dataset",
            ]
        ]
        .drop_duplicates()
        [
            "dataset"
        ]
        .value_counts()
    )

    for dataset in DATASETS:
        if int(
            dataset_counts.get(
                dataset,
                0,
            )
        ) != 24:
            raise RuntimeError(
                f"{method}/{dataset}: "
                "expected 24 tasks."
            )


all_results.to_csv(
    OUTPUT
    / "six_method_per_task_run_results.csv",
    index=False,
)


# ============================================================
# 13. Run-level dataset means
# ============================================================

dataset_run = (
    all_results.groupby(
        [
            "method",
            "run_seed",
            "dataset",
        ],
        as_index=False,
    )[
        "test_accuracy_percent"
    ]
    .mean()
    .rename(
        columns={
            "test_accuracy_percent":
                "run_dataset_mean_accuracy_percent",
        }
    )
)

dataset_run.to_csv(
    OUTPUT
    / "dataset_run_means.csv",
    index=False,
)


# ============================================================
# 14. Run-level overall means
# ============================================================

overall_run = (
    all_results.groupby(
        [
            "method",
            "run_seed",
        ],
        as_index=False,
    )[
        "test_accuracy_percent"
    ]
    .mean()
    .rename(
        columns={
            "test_accuracy_percent":
                "run_overall_mean_accuracy_percent",
        }
    )
)

overall_run.to_csv(
    OUTPUT
    / "overall_run_means.csv",
    index=False,
)


# ============================================================
# 15. Mean ± sample SD across five runs
# ============================================================

summary_rows = []

for method in METHOD_ORDER:

    for dataset in DATASETS:

        values = (
            dataset_run[
                (
                    dataset_run[
                        "method"
                    ]
                    == method
                )
                &
                (
                    dataset_run[
                        "dataset"
                    ]
                    == dataset
                )
            ][
                "run_dataset_mean_accuracy_percent"
            ]
            .sort_index()
            .to_numpy(
                dtype=float
            )
        )

        if len(values) != 5:
            raise RuntimeError(
                f"{method}/{dataset}: "
                "expected five run-level means."
            )

        summary_rows.append(
            {
                "method":
                    method,

                "dataset":
                    dataset,

                "runs":
                    5,

                "mean_accuracy_percent":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "sample_sd_pp":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),
            }
        )

    overall_values = (
        overall_run[
            overall_run[
                "method"
            ]
            == method
        ][
            "run_overall_mean_accuracy_percent"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if len(
        overall_values
    ) != 5:
        raise RuntimeError(
            f"{method}: expected five "
            "overall run means."
        )

    summary_rows.append(
        {
            "method":
                method,

            "dataset":
                "__overall__",

            "runs":
                5,

            "mean_accuracy_percent":
                float(
                    np.mean(
                        overall_values
                    )
                ),

            "sample_sd_pp":
                float(
                    np.std(
                        overall_values,
                        ddof=1,
                    )
                ),
        }
    )


numeric_summary = pd.DataFrame(
    summary_rows
)

numeric_summary.to_csv(
    OUTPUT
    / "lodo_numeric_summary.csv",
    index=False,
)


# ============================================================
# 16. Reviewer-facing paper table
# ============================================================

paper_rows = []

for method in METHOD_ORDER:

    row = {
        "Method":
            METHOD_DISPLAY[
                method
            ],
    }

    for dataset in DATASETS:

        record = (
            numeric_summary[
                (
                    numeric_summary[
                        "method"
                    ]
                    == method
                )
                &
                (
                    numeric_summary[
                        "dataset"
                    ]
                    == dataset
                )
            ]
            .iloc[0]
        )

        row[
            DATASET_DISPLAY[
                dataset
            ]
        ] = (
            f"{record['mean_accuracy_percent']:.3f} "
            f"± "
            f"{record['sample_sd_pp']:.3f}"
        )

    overall = (
        numeric_summary[
            (
                numeric_summary[
                    "method"
                ]
                == method
            )
            &
            (
                numeric_summary[
                    "dataset"
                ]
                == "__overall__"
            )
        ]
        .iloc[0]
    )

    row[
        "LODO Overall"
    ] = (
        f"{overall['mean_accuracy_percent']:.3f} "
        f"± "
        f"{overall['sample_sd_pp']:.3f}"
    )

    row[
        "Target-task search cost (FBE)"
    ] = (
        SEARCH_COST_FBE[
            method
        ]
    )

    paper_rows.append(
        row
    )


paper_table = pd.DataFrame(
    paper_rows
)

paper_table.to_csv(
    OUTPUT
    / "lodo_five_dataset_paper_table.csv",
    index=False,
)


# ============================================================
# 17. Paired task bootstrap
#
# IMPORTANT:
# Inferential unit = held-out task.
#
# Each task is first averaged over the five
# recommendation/search runs.
#
# We then bootstrap the 120 task-level paired
# differences, NOT the 600 run-task rows.
# ============================================================

task_method = (
    all_results.groupby(
        [
            "method",
            "task_id",
            "dataset",
        ],
        as_index=False,
    )[
        "test_accuracy_percent"
    ]
    .mean()
    .rename(
        columns={
            "test_accuracy_percent":
                "five_run_task_mean_accuracy_percent",
        }
    )
)

task_method.to_csv(
    OUTPUT
    / "five_run_task_mean_results.csv",
    index=False,
)


def paired_bootstrap(
    *,
    zero_values,
    comparator_values,
    reps,
    seed,
):
    zero_values = np.asarray(
        zero_values,
        dtype=np.float64,
    )

    comparator_values = np.asarray(
        comparator_values,
        dtype=np.float64,
    )

    if (
        len(zero_values)
        != len(
            comparator_values
        )
    ):
        raise RuntimeError(
            "Paired vectors differ in length."
        )

    difference = (
        zero_values
        - comparator_values
    )

    n = len(
        difference
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        reps,
        dtype=np.float64,
    )

    # Memory-safe bootstrap.
    chunk = 1000

    completed = 0

    while completed < reps:

        current = min(
            chunk,
            reps - completed,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                current,
                n,
            ),
        )

        bootstrap_means[
            completed:
            completed + current
        ] = (
            difference[
                indices
            ]
            .mean(
                axis=1
            )
        )

        completed += current

    tolerance = 1e-12

    return {
        "tasks":
            int(
                n
            ),

        "mean_difference_pp":
            float(
                np.mean(
                    difference
                )
            ),

        "ci95_low_pp":
            float(
                np.percentile(
                    bootstrap_means,
                    2.5,
                )
            ),

        "ci95_high_pp":
            float(
                np.percentile(
                    bootstrap_means,
                    97.5,
                )
            ),

        "wins":
            int(
                np.sum(
                    difference
                    > tolerance
                )
            ),

        "ties":
            int(
                np.sum(
                    np.abs(
                        difference
                    )
                    <= tolerance
                )
            ),

        "losses":
            int(
                np.sum(
                    difference
                    < -tolerance
                )
            ),
    }


# ============================================================
# 18. Overall paired comparisons
# ============================================================

zero_task = (
    task_method[
        task_method[
            "method"
        ]
        == "zero_sml_lodo"
    ][
        [
            "task_id",
            "five_run_task_mean_accuracy_percent",
        ]
    ]
    .rename(
        columns={
            "five_run_task_mean_accuracy_percent":
                "zero_accuracy",
        }
    )
)


overall_bootstrap_rows = []

comparators = [
    method
    for method in METHOD_ORDER
    if method
    != "zero_sml_lodo"
]


for index, comparator in enumerate(
    comparators
):

    comp = (
        task_method[
            task_method[
                "method"
            ]
            == comparator
        ][
            [
                "task_id",
                "five_run_task_mean_accuracy_percent",
            ]
        ]
        .rename(
            columns={
                "five_run_task_mean_accuracy_percent":
                    "comparator_accuracy",
            }
        )
    )

    paired = (
        zero_task.merge(
            comp,
            on="task_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(
        paired
    ) != 120:
        raise RuntimeError(
            f"{comparator}: expected "
            "120 paired tasks."
        )

    stats = paired_bootstrap(
        zero_values=(
            paired[
                "zero_accuracy"
            ]
        ),

        comparator_values=(
            paired[
                "comparator_accuracy"
            ]
        ),

        reps=(
            BOOTSTRAP_REPS
        ),

        seed=(
            BOOTSTRAP_SEED
            + index
        ),
    )

    stats[
        "method"
    ] = "zero_sml_lodo"

    stats[
        "comparator"
    ] = comparator

    stats[
        "method_display"
    ] = METHOD_DISPLAY[
        "zero_sml_lodo"
    ]

    stats[
        "comparator_display"
    ] = METHOD_DISPLAY[
        comparator
    ]

    overall_bootstrap_rows.append(
        stats
    )


overall_bootstrap = pd.DataFrame(
    overall_bootstrap_rows
)

overall_bootstrap.to_csv(
    OUTPUT
    / "paired_bootstrap_overall.csv",
    index=False,
)


# ============================================================
# 19. Dataset-specific paired bootstrap
# ============================================================

dataset_bootstrap_rows = []

counter = 100

for dataset in DATASETS:

    zero_dataset = (
        task_method[
            (
                task_method[
                    "method"
                ]
                == "zero_sml_lodo"
            )
            &
            (
                task_method[
                    "dataset"
                ]
                == dataset
            )
        ][
            [
                "task_id",
                "five_run_task_mean_accuracy_percent",
            ]
        ]
        .rename(
            columns={
                "five_run_task_mean_accuracy_percent":
                    "zero_accuracy",
            }
        )
    )

    if len(
        zero_dataset
    ) != 24:
        raise RuntimeError(
            f"{dataset}: Zero-SML "
            "must contain 24 tasks."
        )

    for comparator in comparators:

        comp = (
            task_method[
                (
                    task_method[
                        "method"
                    ]
                    == comparator
                )
                &
                (
                    task_method[
                        "dataset"
                    ]
                    == dataset
                )
            ][
                [
                    "task_id",
                    "five_run_task_mean_accuracy_percent",
                ]
            ]
            .rename(
                columns={
                    "five_run_task_mean_accuracy_percent":
                        "comparator_accuracy",
                }
            )
        )

        paired = (
            zero_dataset.merge(
                comp,
                on="task_id",
                how="inner",
                validate="one_to_one",
            )
        )

        if len(
            paired
        ) != 24:
            raise RuntimeError(
                f"{dataset}/{comparator}: "
                "expected 24 paired tasks."
            )

        stats = paired_bootstrap(
            zero_values=(
                paired[
                    "zero_accuracy"
                ]
            ),

            comparator_values=(
                paired[
                    "comparator_accuracy"
                ]
            ),

            reps=(
                BOOTSTRAP_REPS
            ),

            seed=(
                BOOTSTRAP_SEED
                + counter
            ),
        )

        stats[
            "dataset"
        ] = dataset

        stats[
            "comparator"
        ] = comparator

        stats[
            "comparator_display"
        ] = METHOD_DISPLAY[
            comparator
        ]

        dataset_bootstrap_rows.append(
            stats
        )

        counter += 1


dataset_bootstrap = pd.DataFrame(
    dataset_bootstrap_rows
)

dataset_bootstrap.to_csv(
    OUTPUT
    / "paired_bootstrap_by_dataset.csv",
    index=False,
)


# ============================================================
# 20. Search-cost table
# ============================================================

cost_rows = []

for method in METHOD_ORDER:

    cost_rows.append(
        {
            "method":
                method,

            "display":
                METHOD_DISPLAY[
                    method
                ],

            "target_task_search_cost_fbe":
                SEARCH_COST_FBE[
                    method
                ],

            "note":
                (
                    "Zero target-task portfolio "
                    "model evaluations for "
                    "configuration selection; "
                    "offline training/descriptor "
                    "cost is non-zero."
                    if (
                        SEARCH_COST_FBE[
                            method
                        ]
                        == 0
                    )
                    else (
                        "Target-local search cost "
                        "expressed as 200-episode "
                        "full-budget equivalents."
                    )
                ),
        }
    )

pd.DataFrame(
    cost_rows
).to_csv(
    OUTPUT
    / "lodo_search_costs.csv",
    index=False,
)


# ============================================================
# 21. Metadata
# ============================================================

metadata = {
    "schema_version":
        3,

    "experiment":
        "strict_leave_one_dataset_out_generalization",

    "datasets":
        DATASETS,

    "folds":
        5,

    "source_meta_train_tasks_per_fold":
        448,

    "source_meta_validation_tasks_per_fold":
        96,

    "source_refit_tasks_per_fold":
        544,

    "heldout_target_tasks_per_fold":
        24,

    "recommendation_search_runs":
        5,

    "sml_run_seeds":
        [0, 1, 2, 3, 4],

    "npp_run_seeds":
        [0, 1, 2, 3, 4],

    "downstream_protonet_model_seeds":
        [101, 202, 303],

    "training_episodes":
        200,

    "test_episodes":
        600,

    "unique_lodo_required_pairs":
        405,

    "fresh_lodo_pair_evaluations":
        88,

    "reused_pair_evaluations":
        317,

    "bootstrap_repetitions":
        BOOTSTRAP_REPS,

    "bootstrap_seed_base":
        BOOTSTRAP_SEED,

    "bootstrap_inferential_unit":
        "held-out task",

    "bootstrap_run_handling":
        (
            "Average the five recommendation/search "
            "runs for each task first; bootstrap "
            "paired held-out tasks afterward."
        ),

    "reported_standard_deviation":
        (
            "sample standard deviation (ddof=1) "
            "across five run-level dataset means"
        ),

    "lodo_zero_sml_target_dataset_in_training":
        False,

    "lodo_npp_target_dataset_in_training":
        False,

    "iterative_hpo_note":
        (
            "Random Search, Bayesian Optimization, "
            "Hyperband, and BOHB are target-local "
            "search procedures and therefore do not "
            "use source-dataset meta-training. Their "
            "previously frozen target-task results "
            "are reused unchanged in each LODO fold."
        ),

    "learned_methods_search_cost_note":
        (
            "0 FBE denotes zero target-task portfolio "
            "model evaluations before configuration "
            "selection; it does not denote zero "
            "offline compute."
        ),
}

metadata_path = (
    OUTPUT
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


# ============================================================
# 22. Hash final artifacts
# ============================================================

artifacts = [
    OUTPUT
    / "lodo_learned_recommendation_test_results.csv",

    OUTPUT
    / "six_method_per_task_run_results.csv",

    OUTPUT
    / "dataset_run_means.csv",

    OUTPUT
    / "overall_run_means.csv",

    OUTPUT
    / "lodo_numeric_summary.csv",

    OUTPUT
    / "lodo_five_dataset_paper_table.csv",

    OUTPUT
    / "five_run_task_mean_results.csv",

    OUTPUT
    / "paired_bootstrap_overall.csv",

    OUTPUT
    / "paired_bootstrap_by_dataset.csv",

    OUTPUT
    / "lodo_search_costs.csv",

    OUTPUT
    / "analysis_metadata.json",
]

hash_path = (
    OUTPUT
    / "final_analysis.sha256"
)

with hash_path.open(
    "w",
    encoding="utf-8",
) as handle:

    for path in artifacts:
        handle.write(
            f"{sha256_file(path)}  "
            f"{path}\n"
        )


# ============================================================
# 23. Console report
# ============================================================

print()
print("=" * 120)
print(
    "STRICT FIVE-DATASET LODO GENERALIZATION RESULTS"
)
print("=" * 120)

print()
print(
    paper_table.to_string(
        index=False
    )
)

print()
print("=" * 120)
print(
    "ZERO-SML LODO PAIRED TASK BOOTSTRAP"
)
print("=" * 120)

display_bootstrap = (
    overall_bootstrap[
        [
            "comparator_display",
            "tasks",
            "mean_difference_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "wins",
            "ties",
            "losses",
        ]
    ]
    .copy()
)

print(
    display_bootstrap.to_string(
        index=False,
        formatters={
            "mean_difference_pp":
                lambda x:
                    f"{x:.3f}",

            "ci95_low_pp":
                lambda x:
                    f"{x:.3f}",

            "ci95_high_pp":
                lambda x:
                    f"{x:.3f}",
        },
    )
)

print()
print("=" * 120)
print(
    "LODO FINAL AUDIT"
)
print("=" * 120)

print(
    "Required unique LODO pairs:",
    len(
        required_pairs
    ),
)

print(
    "Fresh final-test evaluations:",
    len(
        fresh_cache
    ),
)

print(
    "Required pairs still missing:",
    missing_after_eval,
)

print(
    "Learned recommendation rows:",
    len(
        learned_results
    ),
)

print(
    "Six-method task/run rows:",
    len(
        all_results
    ),
)

print(
    "Methods:",
    all_results[
        "method"
    ].nunique(),
)

print(
    "Runs/method:",
    all_results.groupby(
        "method"
    )[
        "run_seed"
    ]
    .nunique()
    .to_dict(),
)

print(
    "Unique tasks/method:",
    all_results.groupby(
        "method"
    )[
        "task_id"
    ]
    .nunique()
    .to_dict(),
)

print(
    "Bootstrap repetitions:",
    BOOTSTRAP_REPS,
)

print()
print(
    "Final artifacts:",
    OUTPUT
)

print()
print(
    "STRICT FIVE-DATASET "
    "LODO GENERALIZATION: PASS"
)


if __name__ == "__main__":
    pass
