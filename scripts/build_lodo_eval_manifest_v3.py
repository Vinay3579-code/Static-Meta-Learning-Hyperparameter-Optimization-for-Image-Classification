from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ZERO_REC = Path(
    "results/lodo_generalization_v3/"
    "zero_sml/locked_recommendations.csv"
)

NPP_REC = Path(
    "results/lodo_generalization_v3/"
    "probe_npp/locked_recommendations.csv"
)

OUTPUT = Path(
    "results/lodo_generalization_v3/"
    "final_test_evaluation"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# Existing final held-out evaluations that used the frozen
# 200-train / 600-test / seeds 101,202,303 protocol.
CACHE_ROOTS = [
    Path(
        "results/hpo_baselines_v2/"
        "final_test_evaluation"
    ),

    Path(
        "results/meta_learning/"
        "five_run_robustness_v2/"
        "final_test_evaluation"
    ),

    Path(
        "results/final_test_eval_v2"
    ),
]


TEST_COLUMNS = [
    "test_accuracy_seed101",
    "test_accuracy_seed202",
    "test_accuracy_seed303",
    "test_accuracy_mean",
]


# ============================================================
# Helpers
# ============================================================

def normalize_dataset(x):
    x = str(x).strip().lower()

    aliases = {
        "cifar-100": "cifar100",
        "cifar_100": "cifar100",
        "mini-imagenet": "miniimagenet",
        "mini_imagenet": "miniimagenet",
        "flowers-102": "flowers102",
        "flowers_102": "flowers102",
    }

    return aliases.get(x, x)


def standardize_locked(
    path: Path,
    method_name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "task_id",
        "heldout_dataset",
        "run_seed",
        "selected_config_id",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing columns "
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

    result["method"] = (
        method_name
    )

    result["task_id"] = (
        result["task_id"]
        .astype(str)
    )

    result[
        "selected_config_id"
    ] = (
        result[
            "selected_config_id"
        ]
        .astype(str)
    )

    result[
        "heldout_dataset"
    ] = (
        result[
            "heldout_dataset"
        ]
        .map(
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

    return result


def extract_cache_rows(
    path: Path,
):
    """
    Accept only CSVs containing:
      task_id
      config identifier
      all three fixed evaluation-seed accuracies
      test_accuracy_mean

    This intentionally refuses looser / incompatible files.
    """

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "task_id" not in df.columns:
        return None

    if (
        "selected_config_id"
        in df.columns
    ):
        config_col = (
            "selected_config_id"
        )

    elif (
        "config_id"
        in df.columns
    ):
        config_col = "config_id"

    else:
        return None

    if not set(
        TEST_COLUMNS
    ).issubset(
        df.columns
    ):
        return None

    keep = df[
        [
            "task_id",
            config_col,
            *TEST_COLUMNS,
        ]
    ].copy()

    keep = keep.rename(
        columns={
            config_col:
                "config_id",
        }
    )

    keep["task_id"] = (
        keep["task_id"]
        .astype(str)
    )

    keep["config_id"] = (
        keep["config_id"]
        .astype(str)
    )

    for column in TEST_COLUMNS:
        keep[column] = (
            pd.to_numeric(
                keep[column],
                errors="coerce",
            )
        )

    keep = keep.dropna(
        subset=TEST_COLUMNS
    )

    if keep.empty:
        return None

    keep["pair_key"] = (
        keep["task_id"]
        + "||"
        + keep["config_id"]
    )

    keep[
        "cache_source"
    ] = str(path)

    return keep


# ============================================================
# Load locked recommendations
# ============================================================

zero = standardize_locked(
    ZERO_REC,
    "zero_sml_lodo",
)

npp = standardize_locked(
    NPP_REC,
    "probe_npp_lodo",
)


if len(zero) != 600:
    raise RuntimeError(
        f"Expected 600 Zero-SML rows, "
        f"found {len(zero)}"
    )

if len(npp) != 600:
    raise RuntimeError(
        f"Expected 600 NPP rows, "
        f"found {len(npp)}"
    )


recommendations = (
    pd.concat(
        [
            zero,
            npp,
        ],
        ignore_index=True,
    )
)


# ============================================================
# Locked recommendation audits
# ============================================================

expected_datasets = {
    "omniglot",
    "cifar100",
    "miniimagenet",
    "dtd",
    "flowers102",
}


if set(
    recommendations[
        "heldout_dataset"
    ].unique()
) != expected_datasets:
    raise RuntimeError(
        "Unexpected held-out datasets."
    )


if set(
    recommendations[
        "run_seed"
    ].unique()
) != {
    0, 1, 2, 3, 4
}:
    raise RuntimeError(
        "Unexpected run seeds."
    )


counts = (
    recommendations.groupby(
        [
            "method",
            "heldout_dataset",
            "run_seed",
        ]
    )[
        "task_id"
    ]
    .nunique()
)


if not (
    counts == 24
).all():
    raise RuntimeError(
        "Every method/dataset/run "
        "must contain exactly 24 tasks."
    )


if recommendations[
    [
        "method",
        "heldout_dataset",
        "run_seed",
        "task_id",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate locked recommendations."
    )


print()
print("=" * 100)
print(
    "LODO LOCKED RECOMMENDATION AUDIT"
)
print("=" * 100)

print(
    "Total recommendation rows:",
    len(recommendations),
)

print(
    "Unique methods:",
    recommendations[
        "method"
    ].nunique(),
)

print(
    "Held-out datasets:",
    recommendations[
        "heldout_dataset"
    ].nunique(),
)

print(
    "Runs:",
    sorted(
        recommendations[
            "run_seed"
        ]
        .unique()
        .tolist()
    ),
)


# ============================================================
# Discover compatible existing final-evaluation CSVs
# ============================================================

candidate_files = []

for root in CACHE_ROOTS:

    if not root.exists():
        print(
            "WARNING: cache root missing:",
            root,
        )
        continue

    candidate_files.extend(
        sorted(
            root.rglob("*.csv")
        )
    )


cache_frames = []
accepted_files = []


for path in candidate_files:

    # Avoid recursively consuming anything we are
    # writing in the new LODO output location.
    try:
        path.resolve().relative_to(
            OUTPUT.resolve()
        )
        continue
    except ValueError:
        pass

    extracted = (
        extract_cache_rows(
            path
        )
    )

    if extracted is None:
        continue

    cache_frames.append(
        extracted
    )

    accepted_files.append(
        (
            str(path),
            len(extracted),
            extracted[
                "pair_key"
            ].nunique(),
        )
    )


if not cache_frames:
    raise RuntimeError(
        "No compatible existing final-test "
        "evaluation CSVs were found."
    )


cache_raw = pd.concat(
    cache_frames,
    ignore_index=True,
)


print()
print("=" * 100)
print(
    "COMPATIBLE EXISTING TEST CACHE SOURCES"
)
print("=" * 100)

for path, rows, pairs in (
    accepted_files
):
    print(
        f"{path}"
        f" | rows={rows}"
        f" | pairs={pairs}"
    )


# ============================================================
# Cross-source consistency audit
# ============================================================

numeric = cache_raw[
    [
        "pair_key",
        *TEST_COLUMNS,
    ]
].copy()


consistency = (
    numeric.groupby(
        "pair_key"
    )[
        TEST_COLUMNS
    ]
    .agg(
        [
            "min",
            "max",
        ]
    )
)


bad_pairs = []


for pair_key, row in (
    consistency.iterrows()
):

    for column in TEST_COLUMNS:

        low = float(
            row[
                (
                    column,
                    "min",
                )
            ]
        )

        high = float(
            row[
                (
                    column,
                    "max",
                )
            ]
        )

        if not np.isclose(
            low,
            high,
            rtol=0.0,
            atol=1e-12,
        ):
            bad_pairs.append(
                (
                    pair_key,
                    column,
                    low,
                    high,
                )
            )


if bad_pairs:
    print()
    print(
        "INCONSISTENT EXISTING "
        "CACHE ENTRIES:"
    )

    for item in bad_pairs[:20]:
        print(item)

    raise RuntimeError(
        "Existing final-test cache sources "
        "disagree for the same task/config pair."
    )


print()
print(
    "Existing cache consistency: PASS"
)


# ============================================================
# Collapse to one row per evaluated task/config pair
# ============================================================

cache = (
    cache_raw.sort_values(
        [
            "pair_key",
            "cache_source",
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


cache = cache[
    [
        "pair_key",
        "task_id",
        "config_id",
        *TEST_COLUMNS,
        "cache_source",
    ]
]


cache.to_csv(
    OUTPUT
    / "existing_pair_cache_index.csv",
    index=False,
)


print(
    "Unique existing evaluated pairs:",
    len(cache),
)


# ============================================================
# Required unique pairs for the new LODO recommendations
# ============================================================

required_pairs = (
    recommendations[
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
    .rename(
        columns={
            "selected_config_id":
                "config_id",
        }
    )
    .sort_values(
        [
            "task_id",
            "config_id",
        ],
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)


required_pairs = (
    required_pairs.merge(
        cache[
            [
                "pair_key",
                *TEST_COLUMNS,
                "cache_source",
            ]
        ],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
)


required_pairs[
    "evaluation_status"
] = np.where(
    required_pairs[
        "test_accuracy_mean"
    ].notna(),
    "cache_hit",
    "missing",
)


required_pairs.to_csv(
    OUTPUT
    / "required_unique_pairs.csv",
    index=False,
)


missing = (
    required_pairs[
        required_pairs[
            "evaluation_status"
        ]
        == "missing"
    ][
        [
            "task_id",
            "config_id",
            "pair_key",
        ]
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


missing.to_csv(
    OUTPUT
    / "missing_unique_pairs.csv",
    index=False,
)


# ============================================================
# Map coverage back to every recommendation
# ============================================================

manifest = (
    recommendations.merge(
        required_pairs[
            [
                "pair_key",
                *TEST_COLUMNS,
                "cache_source",
                "evaluation_status",
            ]
        ],
        on="pair_key",
        how="left",
        validate="many_to_one",
    )
)


manifest.to_csv(
    OUTPUT
    / "lodo_recommendation_eval_manifest.csv",
    index=False,
)


coverage = (
    manifest.groupby(
        [
            "method",
            "heldout_dataset",
        ],
        as_index=False,
    )
    .agg(
        recommendations=(
            "task_id",
            "size",
        ),

        unique_pairs=(
            "pair_key",
            "nunique",
        ),

        cache_hits=(
            "evaluation_status",
            lambda s:
                int(
                    (
                        s
                        == "cache_hit"
                    ).sum()
                ),
        ),

        missing_rows=(
            "evaluation_status",
            lambda s:
                int(
                    (
                        s
                        == "missing"
                    ).sum()
                ),
        ),
    )
)


coverage.to_csv(
    OUTPUT
    / "cache_coverage_by_method_dataset.csv",
    index=False,
)


# ============================================================
# Print audit
# ============================================================

print()
print("=" * 100)
print(
    "LODO FINAL TEST CACHE COVERAGE"
)
print("=" * 100)

print(
    "Required unique task/config pairs:",
    len(
        required_pairs
    ),
)

print(
    "Already evaluated unique pairs:",
    int(
        (
            required_pairs[
                "evaluation_status"
            ]
            == "cache_hit"
        ).sum()
    ),
)

print(
    "Missing unique pairs:",
    len(missing),
)


print()
print(
    coverage.to_string(
        index=False
    )
)


print()

if len(missing) == 0:

    print(
        "All required LODO pairs are "
        "already present in the frozen "
        "held-out evaluation cache."
    )

    print(
        "NO NEW PROTONET EVALUATIONS REQUIRED."
    )

else:

    print(
        "Fresh ProtoNet evaluations required:",
        len(missing),
    )

    print(
        "Saved missing-pair manifest:"
    )

    print(
        OUTPUT
        / "missing_unique_pairs.csv"
    )


print()
print(
    "LODO EVALUATION MANIFEST AUDIT: PASS"
)
