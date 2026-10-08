from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation"
)

CACHE = ROOT / "cache"


def cache_path(
    task_id,
    config_id,
):
    return (
        CACHE
        / str(task_id)
        / f"{config_id}.json"
    )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main():
    rec = pd.read_csv(
        ROOT
        / "frozen_recommendations.csv"
    )

    pairs = pd.read_csv(
        ROOT
        / "unique_task_config_pairs.csv"
    )

    pair_results = []

    for row in pairs.itertuples(
        index=False
    ):
        path = cache_path(
            row.task_id,
            row.selected_config_id,
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing test result: "
                f"{row.task_id} / "
                f"{row.selected_config_id}"
            )

        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        values = np.asarray(
            value[
                "test_accuracies_by_seed"
            ],
            dtype=float,
        )

        if len(values) != 3:
            raise RuntimeError(
                "Expected 3 model seeds."
            )

        if not np.isclose(
            values.mean(),
            float(
                value[
                    "test_accuracy_mean"
                ]
            ),
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Cached mean mismatch."
            )

        pair_results.append(
            {
                "pair_id":
                    row.pair_id,

                "task_id":
                    row.task_id,

                "selected_config_id":
                    row.selected_config_id,

                "test_accuracy_seed101":
                    values[0],

                "test_accuracy_seed202":
                    values[1],

                "test_accuracy_seed303":
                    values[2],

                "test_accuracy_mean":
                    float(
                        values.mean()
                    ),

                "test_accuracy_seed_std":
                    float(
                        values.std(
                            ddof=1
                        )
                    ),

                "evaluation_source":
                    value[
                        "source"
                    ],
            }
        )

    pair_df = pd.DataFrame(
        pair_results
    )

    pair_path = (
        ROOT
        / "unique_pair_test_results.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    merged = rec.merge(
        pair_df,
        on=[
            "pair_id",
            "task_id",
            "selected_config_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(merged) != 3000:
        raise RuntimeError(
            "Expected 3000 recommendation "
            "test-result rows."
        )

    if (
        merged[
            "test_accuracy_mean"
        ].isna().any()
    ):
        raise RuntimeError(
            "Missing joined test result."
        )

    merged_path = (
        ROOT
        / "recommendation_test_results.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    # --------------------------------------------------
    # Each run = mean across the same 120 held-out tasks.
    # Each task/config value itself already averages the
    # common model seeds 101/202/303.
    # --------------------------------------------------

    run_summary = (
        merged.groupby(
            [
                "method",
                "run_seed",
            ]
        )
        .agg(
            tasks=(
                "task_id",
                "nunique",
            ),

            mean_test_accuracy=(
                "test_accuracy_mean",
                "mean",
            ),

            std_across_tasks=(
                "test_accuracy_mean",
                "std",
            ),
        )
        .reset_index()
    )

    if not (
        run_summary[
            "tasks"
        ] == 120
    ).all():
        raise RuntimeError(
            "Every method/run must have "
            "120 held-out tasks."
        )

    run_path = (
        ROOT
        / "run_summary.csv"
    )

    run_summary.to_csv(
        run_path,
        index=False,
    )

    method_summary = (
        run_summary.groupby(
            "method"
        )
        .agg(
            runs=(
                "run_seed",
                "nunique",
            ),

            mean_test_accuracy=(
                "mean_test_accuracy",
                "mean",
            ),

            run_std=(
                "mean_test_accuracy",
                "std",
            ),

            min_run_accuracy=(
                "mean_test_accuracy",
                "min",
            ),

            max_run_accuracy=(
                "mean_test_accuracy",
                "max",
            ),
        )
        .reset_index()
    )

    if not (
        method_summary[
            "runs"
        ] == 5
    ).all():
        raise RuntimeError(
            "Expected five selection "
            "runs per method."
        )

    method_path = (
        ROOT
        / "method_summary.csv"
    )

    method_summary.to_csv(
        method_path,
        index=False,
    )

    # Per-task mean across the five selection/model runs.
    # This becomes the paired-analysis input in Step 8.
    per_task = (
        merged.groupby(
            [
                "method",
                "task_id",
            ]
        )
        .agg(
            mean_test_accuracy=(
                "test_accuracy_mean",
                "mean",
            ),

            selection_run_std=(
                "test_accuracy_mean",
                "std",
            ),
        )
        .reset_index()
    )

    per_task_path = (
        ROOT
        / "per_task_method_results.csv"
    )

    per_task.to_csv(
        per_task_path,
        index=False,
    )

    hashes = {
        path.name:
            sha256_file(path)

        for path in [
            pair_path,
            merged_path,
            run_path,
            method_path,
            per_task_path,
        ]
    }

    hash_path = (
        ROOT
        / "final_test_results.sha256"
    )

    hash_path.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest
            in hashes.items()
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "REVIEWER BASELINE HELD-OUT TEST RESULTS"
    )
    print("=" * 100)

    display = (
        method_summary.copy()
    )

    display[
        "mean_pct"
    ] = (
        100
        * display[
            "mean_test_accuracy"
        ]
    )

    display[
        "std_pct"
    ] = (
        100
        * display[
            "run_std"
        ]
    )

    for row in display.itertuples(
        index=False
    ):
        print(
            f"{row.method:34s} "
            f"{row.mean_pct:8.3f}% "
            f"± {row.std_pct:.3f} pp"
        )

    print()
    print(
        "Unique evaluated task/config pairs:",
        len(pair_df),
    )

    print(
        "Recommendation result rows:",
        len(merged),
    )

    print(
        "Per-task method rows:",
        len(per_task),
    )

    print()
    print(
        "HPO FINAL TEST MERGE: PASS"
    )


if __name__ == "__main__":
    main()
