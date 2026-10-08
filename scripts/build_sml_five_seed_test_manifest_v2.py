from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/meta_learning/"
    "five_run_robustness_v2"
)

REC_PATH = (
    ROOT
    / "test120_recommendations.csv"
)

LOCK_PATH = (
    ROOT
    / "recommendation_lock_manifest.json"
)

OUT = (
    ROOT
    / "final_test_evaluation"
)

LOCAL_CACHE = OUT / "cache"

SHARED_CACHE = Path(
    "results/hpo_baselines_v2/"
    "final_test_evaluation/cache"
)

SEEDS = [0, 1, 2, 3, 4]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def pair_id(
    task_id: str,
    config_id: str,
) -> str:

    return hashlib.sha256(
        f"{task_id}|{config_id}".encode(
            "utf-8"
        )
    ).hexdigest()


def cache_path(
    root: Path,
    task_id: str,
    config_id: str,
) -> Path:

    return (
        root
        / task_id
        / f"{config_id}.json"
    )


def validate_cache(
    path: Path,
    task_id: str,
    config_id: str,
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

    values = value[
        "test_accuracies_by_seed"
    ]

    assert len(values) == 3

    return value


def atomic_json(
    path: Path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = (
        path.parent
        / f"{path.name}.tmp.{os.getpid()}"
    )

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCAL_CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = json.loads(
        LOCK_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        lock[
            "selection_frozen"
        ]
        is True
    )

    assert (
        lock[
            "test_oracle_labels_read"
        ]
        is False
    )

    assert (
        lock[
            "test_bank_used"
        ]
        is False
    )

    assert (
        lock[
            "test_evaluation_performed"
        ]
        is False
    )

    rec = pd.read_csv(
        REC_PATH
    )

    if len(rec) != 120:
        raise RuntimeError(
            "Expected 120 test tasks."
        )

    rows = []

    for variant in [
        "zero_sml",
        "probe_sml",
    ]:
        for seed in SEEDS:

            column = (
                f"{variant}_seed{seed}"
            )

            if column not in rec.columns:
                raise RuntimeError(
                    f"Missing {column}"
                )

            for row in rec.itertuples(
                index=False
            ):
                task_id = str(
                    row.task_id
                )

                config_id = str(
                    getattr(
                        row,
                        column,
                    )
                )

                rows.append(
                    {
                        "method":
                            variant,

                        "task_id":
                            task_id,

                        "dataset":
                            str(
                                row.dataset
                            ),

                        "regime":
                            str(
                                row.regime
                            ),

                        "run_seed":
                            int(seed),

                        "selected_config_id":
                            config_id,

                        "pair_id":
                            pair_id(
                                task_id,
                                config_id,
                            ),
                    }
                )

    long_df = pd.DataFrame(
        rows
    )

    if len(long_df) != 1200:
        raise RuntimeError(
            "Expected 1200 frozen "
            "recommendation rows."
        )

    if (
        long_df[
            [
                "method",
                "task_id",
                "run_seed",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate method/task/run."
        )

    for method in [
        "zero_sml",
        "probe_sml",
    ]:
        sub = long_df[
            long_df[
                "method"
            ] == method
        ]

        if len(sub) != 600:
            raise RuntimeError(
                f"{method}: expected "
                "600 rows."
            )

        counts = (
            sub.groupby(
                "run_seed"
            )[
                "task_id"
            ].nunique()
        )

        if not (
            counts == 120
        ).all():
            raise RuntimeError(
                f"{method}: each seed "
                "must have 120 tasks."
            )

    unique_pairs = (
        long_df[
            [
                "pair_id",
                "task_id",
                "selected_config_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "task_id",
                "selected_config_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # Reuse common HPO final-test cache wherever possible.
    # --------------------------------------------------

    imported = 0
    local_already = 0
    shared_missing = 0

    for row in unique_pairs.itertuples(
        index=False
    ):
        task_id = str(
            row.task_id
        )

        config_id = str(
            row.selected_config_id
        )

        local_path = cache_path(
            LOCAL_CACHE,
            task_id,
            config_id,
        )

        if local_path.exists():
            validate_cache(
                local_path,
                task_id,
                config_id,
            )

            local_already += 1
            continue

        shared_path = cache_path(
            SHARED_CACHE,
            task_id,
            config_id,
        )

        if not shared_path.exists():
            shared_missing += 1
            continue

        value = validate_cache(
            shared_path,
            task_id,
            config_id,
        )

        imported_value = dict(
            value
        )

        imported_value[
            "source"
        ] = (
            "shared_hpo_final_test_cache"
        )

        imported_value[
            "shared_cache_path"
        ] = str(
            shared_path
        )

        atomic_json(
            local_path,
            imported_value,
        )

        validate_cache(
            local_path,
            task_id,
            config_id,
        )

        imported += 1

    long_path = (
        OUT
        / "frozen_recommendations.csv"
    )

    pair_path = (
        OUT
        / "unique_task_config_pairs.csv"
    )

    long_df.to_csv(
        long_path,
        index=False,
    )

    unique_pairs.to_csv(
        pair_path,
        index=False,
    )

    cached_total = 0

    for row in unique_pairs.itertuples(
        index=False
    ):
        if cache_path(
            LOCAL_CACHE,
            str(row.task_id),
            str(
                row.selected_config_id
            ),
        ).exists():
            cached_total += 1

    manifest = {
        "schema_version":
            1,

        "experiment":
            "five_seed_sml_final_test",

        "methods": [
            "zero_sml",
            "probe_sml",
        ],

        "selection_seeds":
            SEEDS,

        "recommendation_rows":
            int(
                len(long_df)
            ),

        "unique_task_config_pairs":
            int(
                len(unique_pairs)
            ),

        "training_episodes":
            200,

        "test_episodes":
            600,

        "protonet_model_seeds": [
            101,
            202,
            303,
        ],

        "recommendations_locked_before_evaluation":
            True,

        "recommendation_file":
            str(
                REC_PATH
            ),

        "recommendation_file_sha256":
            sha256_file(
                REC_PATH
            ),

        "recommendation_lock_sha256":
            sha256_file(
                LOCK_PATH
            ),

        "shared_cache_imports":
            int(
                imported
            ),

        "local_existing_cache":
            int(
                local_already
            ),

        "remaining_gpu_evaluations":
            int(
                len(unique_pairs)
                - cached_total
            ),
    }

    manifest_path = (
        OUT
        / "evaluation_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "FIVE-SEED SML FINAL-TEST MANIFEST"
    )
    print("=" * 100)

    print(
        "Frozen recommendation rows:",
        len(long_df),
    )

    print(
        "Unique task/config pairs:",
        len(unique_pairs),
    )

    print(
        "Imported from common cache:",
        imported,
    )

    print(
        "Already locally cached:",
        local_already,
    )

    print(
        "Total cache coverage:",
        cached_total,
    )

    print(
        "Remaining GPU evaluations:",
        len(unique_pairs)
        - cached_total,
    )

    print(
        "Training episodes:",
        200,
    )

    print(
        "Test episodes:",
        600,
    )

    print(
        "ProtoNet seeds:",
        [101, 202, 303],
    )

    print()
    print(
        "FIVE-SEED SML TEST "
        "MANIFEST: PASS"
    )


if __name__ == "__main__":
    main()
