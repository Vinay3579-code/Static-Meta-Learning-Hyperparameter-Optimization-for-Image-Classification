from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "results/meta_learning/"
    "five_run_robustness_v2/"
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

    with open(
        path,
        "rb",
    ) as f:
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

    pair_rows = []

    for row in pairs.itertuples(
        index=False
    ):
        path = cache_path(
            row.task_id,
            row.selected_config_id,
        )

        if not path.exists():
            raise RuntimeError(
                "Missing cache for "
                f"{row.task_id} / "
                f"{row.selected_config_id}"
            )

        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        acc = np.asarray(
            value[
                "test_accuracies_by_seed"
            ],
            dtype=float,
        )

        if len(acc) != 3:
            raise RuntimeError(
                "Expected 3 ProtoNet seeds."
            )

        if not np.isclose(
            acc.mean(),
            float(
                value[
                    "test_accuracy_mean"
                ]
            ),
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Cached mean mismatch."
            )

        pair_rows.append(
            {
                "pair_id":
                    row.pair_id,

                "task_id":
                    row.task_id,

                "selected_config_id":
                    row.selected_config_id,

                "test_accuracy_seed101":
                    float(
                        acc[0]
                    ),

                "test_accuracy_seed202":
                    float(
                        acc[1]
                    ),

                "test_accuracy_seed303":
                    float(
                        acc[2]
                    ),

                "test_accuracy_mean":
                    float(
                        acc.mean()
                    ),

                "test_accuracy_seed_std":
                    float(
                        acc.std(
                            ddof=1
                        )
                    ),

                "evaluation_source":
                    str(
                        value.get(
                            "source",
                            "unknown",
                        )
                    ),
            }
        )

    pair_df = pd.DataFrame(
        pair_rows
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

    if len(merged) != 1200:
        raise RuntimeError(
            "Expected 1200 joined rows."
        )

    if merged[
        "test_accuracy_mean"
    ].isna().any():
        raise RuntimeError(
            "Missing held-out test score."
        )

    merged_path = (
        ROOT
        / "recommendation_test_results.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    # One run-level value = average over 120 tasks.
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

    if len(
        run_summary
    ) != 10:
        raise RuntimeError(
            "Expected 10 method/run rows."
        )

    if not (
        run_summary[
            "tasks"
        ] == 120
    ).all():
        raise RuntimeError(
            "Every SML run must contain "
            "120 tasks."
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
            "Expected five SML runs."
        )

    method_path = (
        ROOT
        / "method_summary.csv"
    )

    method_summary.to_csv(
        method_path,
        index=False,
    )

    # For paired bootstrap in the final analysis:
    # average each task over its five SML
    # recommendation runs.
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

    if len(per_task) != 240:
        raise RuntimeError(
            "Expected 240 per-task "
            "method rows."
        )

    per_task_path = (
        ROOT
        / "per_task_method_results.csv"
    )

    per_task.to_csv(
        per_task_path,
        index=False,
    )

    output_paths = [
        pair_path,
        merged_path,
        run_path,
        method_path,
        per_task_path,
    ]

    hash_path = (
        ROOT
        / "final_test_results.sha256"
    )

    hash_path.write_text(
        "".join(
            (
                f"{sha256_file(p)}  "
                f"{p}\n"
            )
            for p
            in output_paths
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "FIVE-SEED SML HELD-OUT TEST RESULTS"
    )
    print("=" * 100)

    for row in (
        method_summary.itertuples(
            index=False
        )
    ):
        print(
            f"{row.method:20s} "
            f"{100 * row.mean_test_accuracy:8.3f}% "
            f"± "
            f"{100 * row.run_std:.3f} pp"
        )

    print()
    print(
        "Unique evaluated pairs:",
        len(pair_df),
    )

    print(
        "Recommendation results:",
        len(merged),
    )

    print(
        "Run summary rows:",
        len(run_summary),
    )

    print(
        "Per-task method rows:",
        len(per_task),
    )

    print()
    print(
        "FIVE-SEED SML FINAL "
        "TEST MERGE: PASS"
    )


if __name__ == "__main__":
    main()
