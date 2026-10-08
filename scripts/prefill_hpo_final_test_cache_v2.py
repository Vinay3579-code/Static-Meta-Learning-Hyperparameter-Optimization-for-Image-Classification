from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_ROOT = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation"
)

PAIR_PATH = (
    MANIFEST_ROOT
    / "unique_task_config_pairs.csv"
)

EXISTING_RESULTS = Path(
    "results/final_test_eval_v2/"
    "all_method_test_results.csv"
)

CACHE_ROOT = (
    MANIFEST_ROOT
    / "cache"
)


def cache_path(
    task_id,
    config_id,
):
    return (
        CACHE_ROOT
        / task_id
        / f"{config_id}.json"
    )


def atomic_json(
    path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = (
        path.parent
        / (
            path.name
            + f".tmp.{os.getpid()}"
        )
    )

    temp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def parse_json_list(
    value,
):
    if isinstance(
        value,
        list,
    ):
        return value

    return json.loads(
        str(value)
    )


def main():
    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not EXISTING_RESULTS.exists():
        print(
            "Existing final-test file "
            "not found; nothing prefetched."
        )
        return

    pairs = pd.read_csv(
        PAIR_PATH
    )

    existing = pd.read_csv(
        EXISTING_RESULTS
    )

    required = {
        "task_id",
        "config_id",
        "test_accuracy_mean",
    }

    missing = (
        required
        - set(
            existing.columns
        )
    )

    if missing:
        print(
            "Existing results do not "
            "contain required columns:",
            sorted(missing),
        )
        return

    accuracy_column = None

    for candidate in [
        "test_accuracies_by_seed_json",
        "test_accuracies_by_seed",
    ]:
        if candidate in existing.columns:
            accuracy_column = candidate
            break

    if accuracy_column is None:
        print(
            "Existing file lacks "
            "per-seed test accuracies. "
            "Skipping prefill safely."
        )
        return

    wanted = set(
        zip(
            pairs[
                "task_id"
            ].astype(str),
            pairs[
                "selected_config_id"
            ].astype(str),
        )
    )

    existing[
        "task_id"
    ] = (
        existing[
            "task_id"
        ].astype(str)
    )

    existing[
        "config_id"
    ] = (
        existing[
            "config_id"
        ].astype(str)
    )

    imported = 0
    already = 0

    for (
        task_id,
        config_id,
    ), group in existing.groupby(
        [
            "task_id",
            "config_id",
        ]
    ):
        if (
            task_id,
            config_id,
        ) not in wanted:
            continue

        path = cache_path(
            task_id,
            config_id,
        )

        if path.exists():
            already += 1
            continue

        row = group.iloc[0]

        if (
            "training_episodes"
            in group.columns
            and
            int(
                row[
                    "training_episodes"
                ]
            )
            != 200
        ):
            continue

        if (
            "test_episodes"
            in group.columns
            and
            int(
                row[
                    "test_episodes"
                ]
            )
            != 600
        ):
            continue

        if (
            "model_seeds_json"
            in group.columns
        ):
            seeds = parse_json_list(
                row[
                    "model_seeds_json"
                ]
            )

            if [
                int(x)
                for x in seeds
            ] != [
                101,
                202,
                303,
            ]:
                continue

        values = [
            float(x)
            for x in parse_json_list(
                row[
                    accuracy_column
                ]
            )
        ]

        if len(values) != 3:
            continue

        mean = float(
            np.mean(
                values
            )
        )

        stored_mean = float(
            row[
                "test_accuracy_mean"
            ]
        )

        if not np.isclose(
            mean,
            stored_mean,
            atol=1e-10,
            rtol=0.0,
        ):
            continue

        payload = {
            "task_id":
                task_id,

            "config_id":
                config_id,

            "training_episodes":
                200,

            "test_episodes":
                600,

            "model_seeds": [
                101,
                202,
                303,
            ],

            "test_accuracies_by_seed":
                values,

            "test_accuracy_mean":
                mean,

            "test_accuracy_seed_std":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),

            "source":
                "existing_final_test_eval_v2",
        }

        atomic_json(
            path,
            payload,
        )

        imported += 1

    total_pairs = len(
        pairs
    )

    cached = sum(
        1
        for row in pairs.itertuples()
        if cache_path(
            str(row.task_id),
            str(
                row.selected_config_id
            ),
        ).exists()
    )

    print(
        "Unique required pairs:",
        total_pairs,
    )

    print(
        "Imported existing pairs:",
        imported,
    )

    print(
        "Already cached:",
        already,
    )

    print(
        "Total cache coverage:",
        cached,
    )

    print(
        "Remaining GPU evaluations:",
        total_pairs - cached,
    )

    print()
    print(
        "FINAL TEST CACHE PREFILL: PASS"
    )


if __name__ == "__main__":
    main()
