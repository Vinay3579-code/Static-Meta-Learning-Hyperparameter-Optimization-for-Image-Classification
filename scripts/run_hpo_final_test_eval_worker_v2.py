from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.baselines.adapter_v2 import (
    BaselineAdapterV2,
    DEFAULT_ANCHORS,
    DEFAULT_MANIFEST_ROOT,
)


ROOT = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation"
)

PAIR_PATH = (
    ROOT
    / "unique_task_config_pairs.csv"
)

CACHE_ROOT = (
    ROOT
    / "cache"
)

MODEL_SEEDS = (
    101,
    202,
    303,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
    )

    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
    )

    return p.parse_args()


def stable_shard(
    task_id,
    config_id,
    shard_count,
):
    payload = (
        f"{task_id}|{config_id}"
    ).encode(
        "utf-8"
    )

    value = int.from_bytes(
        hashlib.sha256(
            payload
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )

    return value % shard_count


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


def validate_cache(
    path,
    task_id,
    config_id,
):
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert str(
        value[
            "task_id"
        ]
    ) == task_id

    assert str(
        value[
            "config_id"
        ]
    ) == config_id

    assert int(
        value[
            "training_episodes"
        ]
    ) == 200

    assert int(
        value[
            "test_episodes"
        ]
    ) == 600

    assert [
        int(x)
        for x in value[
            "model_seeds"
        ]
    ] == [
        101,
        202,
        303,
    ]

    values = [
        float(x)
        for x in value[
            "test_accuracies_by_seed"
        ]
    ]

    assert len(values) == 3

    assert np.isclose(
        np.mean(values),
        float(
            value[
                "test_accuracy_mean"
            ]
        ),
        atol=1e-10,
        rtol=0.0,
    )

    return value


def main():
    args = parse_args()

    if not (
        0
        <= args.shard_index
        < args.shard_count
    ):
        raise ValueError(
            "Invalid shard index/count."
        )

    pairs = pd.read_csv(
        PAIR_PATH
    )

    pairs = pairs[
        [
            stable_shard(
                str(row.task_id),
                str(
                    row.selected_config_id
                ),
                args.shard_count,
            )
            == args.shard_index

            for row
            in pairs.itertuples()
        ]
    ].reset_index(
        drop=True
    )

    print(
        "Device:",
        args.device,
    )

    print(
        "Shard:",
        args.shard_index,
        "/",
        args.shard_count,
    )

    print(
        "Pairs in shard:",
        len(
            pairs
        ),
    )

    adapter = BaselineAdapterV2(
        anchor_path=(
            DEFAULT_ANCHORS
        ),

        manifest_root=(
            DEFAULT_MANIFEST_ROOT
        ),

        device=args.device,

        image_size=84,

        split_seed=42,
    )

    start_all = time.perf_counter()

    fresh = 0
    cached = 0

    for index, row in enumerate(
        pairs.itertuples(
            index=False
        ),
        start=1,
    ):
        task_id = str(
            row.task_id
        )

        config_id = str(
            row.selected_config_id
        )

        path = cache_path(
            task_id,
            config_id,
        )

        if path.exists():
            validate_cache(
                path,
                task_id,
                config_id,
            )

            cached += 1

            print(
                f"[{index:4d}/"
                f"{len(pairs):4d}] "
                f"CACHE "
                f"{task_id} "
                f"{config_id}"
            )

            continue

        print(
            f"[{index:4d}/"
            f"{len(pairs):4d}] "
            f"EVAL  "
            f"{task_id} "
            f"{config_id}"
        )

        result = (
            adapter
            .evaluate_selected_anchor_on_test(
                task_id=task_id,

                config_id=config_id,

                train_episodes=200,

                model_seeds=(
                    MODEL_SEEDS
                ),
            )
        )

        payload = {
            "task_id":
                task_id,

            "config_id":
                config_id,

            "training_episodes":
                200,

            "test_episodes":
                600,

            "model_seeds":
                list(
                    MODEL_SEEDS
                ),

            "test_accuracies_by_seed":
                [
                    float(x)
                    for x in result[
                        "test_accuracies_by_seed"
                    ]
                ],

            "test_accuracy_mean":
                float(
                    result[
                        "test_accuracy_mean"
                    ]
                ),

            "test_accuracy_seed_std":
                float(
                    result[
                        "test_accuracy_seed_std"
                    ]
                ),

            "mean_seed_elapsed_seconds":
                float(
                    result[
                        "mean_seed_elapsed_seconds"
                    ]
                ),

            "total_elapsed_seconds":
                float(
                    result[
                        "total_elapsed_seconds"
                    ]
                ),

            "source":
                "fresh_baseline_final_test_eval",
        }

        atomic_json(
            path,
            payload,
        )

        validate_cache(
            path,
            task_id,
            config_id,
        )

        fresh += 1

    print()
    print("=" * 90)
    print(
        "FINAL TEST WORKER SUMMARY"
    )
    print("=" * 90)

    print(
        "Shard:",
        args.shard_index,
    )

    print(
        "Pairs:",
        len(pairs),
    )

    print(
        "Fresh evaluations:",
        fresh,
    )

    print(
        "Cache hits:",
        cached,
    )

    print(
        "Wall time:",
        f"{time.perf_counter() - start_all:.2f} s"
    )

    print()
    print(
        "FINAL TEST WORKER: PASS"
    )


if __name__ == "__main__":
    main()
