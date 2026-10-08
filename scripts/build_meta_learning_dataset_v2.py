from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.descriptors.probe import (
    PROBE_FEATURE_NAMES,
)
from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)


DESCRIPTOR_ROOT = Path(
    "results/descriptors/final800/merged"
)

LABEL_ROOT = Path(
    "results/oracles/final800_v2_merged"
)

OUTPUT_ROOT = Path(
    "results/meta_learning/datasets_v2"
)


def build_split(
    split_name: str,
    expected_rows: int,
) -> pd.DataFrame:
    descriptor_path = (
        DESCRIPTOR_ROOT
        / f"{split_name}_full84.csv"
    )

    label_path = (
        LABEL_ROOT
        / f"{split_name}_labels.csv"
    )

    descriptors = pd.read_csv(
        descriptor_path
    )

    labels = pd.read_csv(
        label_path
    )

    if len(descriptors) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} "
            f"descriptor rows for {split_name}, "
            f"found {len(descriptors)}"
        )

    if len(labels) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} "
            f"label rows for {split_name}, "
            f"found {len(labels)}"
        )

    if descriptors[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate descriptor task IDs"
        )

    if labels[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate oracle task IDs"
        )

    descriptor_ids = set(
        descriptors["task_id"]
    )

    label_ids = set(
        labels["task_id"]
    )

    if descriptor_ids != label_ids:
        raise RuntimeError(
            f"Descriptor/oracle task mismatch "
            f"for {split_name}"
        )

    # Keep descriptor-side dataset/regime fields.
    label_columns = [
        "task_id",
        "best_config_id",
        "best_robust_validation_accuracy",
        "best_robust_validation_seed_std",
        "candidate_count",
        "oracle_equivalent_count",
        "oracle_equivalent_config_ids_json",
        "candidate_ids_json",
        "final_label_path",
    ]

    merged = descriptors.merge(
        labels[label_columns],
        on="task_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != expected_rows:
        raise RuntimeError(
            f"Merge produced {len(merged)} rows "
            f"for {split_name}"
        )

    feature_names = [
        *ZERO_FEATURE_NAMES,
        *PROBE_FEATURE_NAMES,
    ]

    if len(feature_names) != 84:
        raise RuntimeError(
            f"Expected 84 features, "
            f"found {len(feature_names)}"
        )

    features = merged[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    if features.shape != (
        expected_rows,
        84,
    ):
        raise RuntimeError(
            f"Invalid feature shape "
            f"{features.shape}"
        )

    if not np.isfinite(
        features
    ).all():
        raise FloatingPointError(
            f"Non-finite {split_name} features"
        )

    merged[
        "meta_split"
    ] = split_name

    return merged


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = build_split(
        "train",
        560,
    )

    validation = build_split(
        "validation",
        120,
    )

    train_ids = set(
        train["task_id"]
    )

    validation_ids = set(
        validation["task_id"]
    )

    if not train_ids.isdisjoint(
        validation_ids
    ):
        raise RuntimeError(
            "Meta-train/meta-validation overlap"
        )

    train.to_csv(
        OUTPUT_ROOT
        / "train.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_ROOT
        / "validation.csv",
        index=False,
    )

    all680 = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    all680.to_csv(
        OUTPUT_ROOT
        / "all680.csv",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "descriptor_dimension": 84,
        "train_tasks": 560,
        "validation_tasks": 120,
        "combined_tasks": 680,
        "train_unique_targets": int(
            train[
                "best_config_id"
            ].nunique()
        ),
        "validation_unique_targets": int(
            validation[
                "best_config_id"
            ].nunique()
        ),
    }

    (
        OUTPUT_ROOT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Train shape:",
        train.shape,
    )
    print(
        "Validation shape:",
        validation.shape,
    )
    print(
        "Descriptor dimension:",
        84,
    )
    print(
        "Train unique oracle configs:",
        summary[
            "train_unique_targets"
        ],
    )
    print(
        "Validation unique oracle configs:",
        summary[
            "validation_unique_targets"
        ],
    )
    print(
        "META-LEARNING DATASET BUILD: PASS"
    )


if __name__ == "__main__":
    main()
