from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.descriptors.probe import (
    PROBE_FEATURE_NAMES,
)
from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)


EXPECTED_COUNTS = {
    "train": 560,
    "validation": 120,
    "test": 120,
}


def main() -> None:
    root = Path(
        "results/descriptors/final800/full84"
    )

    output_root = Path(
        "results/descriptors/final800/merged"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_names = [
        *ZERO_FEATURE_NAMES,
        *PROBE_FEATURE_NAMES,
    ]

    all_tables = []

    for split_name, expected_count in (
        EXPECTED_COUNTS.items()
    ):
        paths = sorted(
            (
                root
                / split_name
            ).glob("*.csv")
        )

        if len(paths) != 20:
            raise RuntimeError(
                f"Expected 20 {split_name} descriptor files, "
                f"found {len(paths)}"
            )

        tables = [
            pd.read_csv(path)
            for path in paths
        ]

        table = pd.concat(
            tables,
            ignore_index=True,
        )

        if len(table) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} {split_name} tasks, "
                f"found {len(table)}"
            )

        if table["task_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate task IDs in {split_name}"
            )

        values = table[
            feature_names
        ].to_numpy(
            dtype=np.float64
        )

        if values.shape != (
            expected_count,
            84,
        ):
            raise RuntimeError(
                f"Invalid {split_name} feature shape: "
                f"{values.shape}"
            )

        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"Non-finite values in {split_name}"
            )

        table["meta_split"] = (
            split_name
        )

        output_path = (
            output_root
            / f"{split_name}_full84.csv"
        )

        table.to_csv(
            output_path,
            index=False,
        )

        all_tables.append(table)

        print()
        print(split_name)
        print("Rows:", len(table))
        print("Feature shape:", values.shape)
        print(
            "Datasets:",
            Counter(table["dataset"]),
        )
        print(
            "Regimes:",
            Counter(
                zip(
                    table["n_way"],
                    table["n_shot"],
                )
            ),
        )

    combined = pd.concat(
        all_tables,
        ignore_index=True,
    )

    if len(combined) != 800:
        raise RuntimeError(
            f"Expected 800 tasks, found "
            f"{len(combined)}"
        )

    if combined["task_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate task IDs across meta splits"
        )

    combined_values = combined[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    feature_stds = combined_values.std(
        axis=0
    )

    nonconstant_count = int(
        np.sum(feature_stds > 1e-12)
    )

    combined_path = (
        output_root
        / "all800_full84.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    summary = {
        "schema_version": 1,
        "task_count": 800,
        "feature_count": 84,
        "split_counts": EXPECTED_COUNTS,
        "dataset_counts": dict(
            Counter(
                combined["dataset"]
            )
        ),
        "nonconstant_feature_count": (
            nonconstant_count
        ),
        "feature_names": feature_names,
    }

    summary_path = (
        output_root
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    if nonconstant_count < 80:
        raise RuntimeError(
            "Too many constant descriptor features"
        )

    print()
    print("Combined rows:", len(combined))
    print(
        "Nonconstant features:",
        nonconstant_count,
        "/ 84",
    )
    print("Combined output:", combined_path)
    print("Summary:", summary_path)
    print("FINAL 800 DESCRIPTOR MERGE: PASS")


if __name__ == "__main__":
    main()
