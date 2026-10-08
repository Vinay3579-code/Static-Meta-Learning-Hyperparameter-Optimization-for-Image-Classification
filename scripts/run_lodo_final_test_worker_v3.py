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
    "results/lodo_generalization_v3/"
    "final_test_evaluation"
)

PAIR_PATH = (
    ROOT
    / "missing_unique_pairs.csv"
)

CACHE_ROOT = (
    ROOT
    / "fresh_cache"
)

MODEL_SEEDS = (
    101,
    202,
    303,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
    )

    return parser.parse_args()


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
        value["task_id"]
    ) == task_id

    assert str(
        value["config_id"]
    ) == config_id

    assert int(
        value["training_episodes"]
    ) == 200

    assert int(
        value["test_episodes"]
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

    accuracies = [
        float(x)
        for x in value[
            "test_accuracies_by_seed"
        ]
    ]

    assert len(
        accuracies
    ) == 3

    assert np.isclose(
        np.mean(
            accuracies
        ),
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
        raise RuntimeError(
            "Invalid shard."
        )

    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            PAIR_PATH
        )

    pairs = pd.read_csv(
        PAIR_PATH
    )

    required = {
        "task_id",
        "config_id",
    }

    missing = (
        required
        - set(
            pairs.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing pair-manifest columns: "
            f"{sorted(missing)}"
        )

    pairs[
        "task_id"
    ] = (
        pairs[
            "task_id"
        ].astype(str)
    )

    pairs[
        "config_id"
    ] = (
        pairs[
            "config_id"
        ].astype(str)
    )

    if pairs[
        [
            "task_id",
            "config_id",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate task/config pairs "
            "in missing manifest."
        )

    if len(pairs) != 88:
        raise RuntimeError(
            "Expected exactly 88 missing "
            f"LODO pairs, found {len(pairs)}."
        )

    selected = []

    for row in pairs.itertuples(
        index=False
    ):
        task_id = str(
            row.task_id
        )

        config_id = str(
            row.config_id
        )

        if (
            stable_shard(
                task_id,
                config_id,
                args.shard_count,
            )
            == args.shard_index
        ):
            selected.append(
                {
                    "task_id":
                        task_id,

                    "config_id":
                        config_id,
                }
            )

    print()
    print("=" * 90)
    print(
        "LODO FINAL TEST WORKER"
    )
    print("=" * 90)

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
        "Total missing pairs:",
        len(pairs),
    )

    print(
        "Pairs in shard:",
        len(selected),
    )

    print(
        "Train episodes:",
        200,
    )

    print(
        "Test episodes:",
        600,
    )

    print(
        "Model seeds:",
        MODEL_SEEDS,
    )

    print()

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

    fresh = 0
    cached = 0

    start = time.perf_counter()

    for index, item in enumerate(
        selected,
        start=1,
    ):
        task_id = (
            item[
                "task_id"
            ]
        )

        config_id = (
            item[
                "config_id"
            ]
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
                f"{len(selected):4d}] "
                f"CACHE "
                f"{task_id} "
                f"{config_id}"
            )

            continue

        print(
            f"[{index:4d}/"
            f"{len(selected):4d}] "
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

        accuracies = [
            float(x)
            for x in result[
                "test_accuracies_by_seed"
            ]
        ]

        if len(
            accuracies
        ) != 3:
            raise RuntimeError(
                f"{task_id}/{config_id}: "
                "expected exactly three "
                "model-seed accuracies."
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
                accuracies,

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
                "fresh_lodo_final_test_v3",
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

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 90)
    print(
        "LODO FINAL TEST WORKER SUMMARY"
    )
    print("=" * 90)

    print(
        "Fresh evaluations:",
        fresh,
    )

    print(
        "Cache hits:",
        cached,
    )

    print(
        "Pairs processed:",
        fresh + cached,
    )

    print(
        "Wall time:",
        f"{elapsed:.2f} s",
    )

    print()
    print(
        "LODO FINAL TEST WORKER: PASS"
    )


if __name__ == "__main__":
    main()
