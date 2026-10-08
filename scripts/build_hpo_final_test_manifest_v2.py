from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/hpo_baselines_v2"
)

OUT = (
    ROOT
    / "final_test_evaluation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SOURCES = [
    {
        "method":
            "random_search",

        "root":
            ROOT
            / "random_search",

        "selection_file":
            "search_selections.csv",

        "seed_column":
            "search_seed",
    },

    {
        "method":
            "bayesian_optimization_gp_ei",

        "root":
            ROOT
            / "bayesian_optimization",

        "selection_file":
            "search_selections.csv",

        "seed_column":
            "search_seed",
    },

    {
        "method":
            "hyperband",

        "root":
            ROOT
            / "hyperband",

        "selection_file":
            "search_selections.csv",

        "seed_column":
            "search_seed",
    },

    {
        "method":
            "bohb_finite_portfolio",

        "root":
            ROOT
            / "bohb",

        "selection_file":
            "search_selections.csv",

        "seed_column":
            "search_seed",
    },

    {
        "method":
            "probe_npp",

        "root":
            ROOT
            / "npp",

        "selection_file":
            "test_recommendations.csv",

        "seed_column":
            "npp_seed",
    },
]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def pair_id(
    task_id: str,
    config_id: str,
) -> str:
    payload = (
        f"{task_id}|{config_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def main():
    all_rows = []
    source_metadata = []

    for source in SOURCES:
        root = source[
            "root"
        ]

        selection_path = (
            root
            / source[
                "selection_file"
            ]
        )

        lock_path = (
            root
            / "selection_lock_manifest.json"
        )

        if not selection_path.exists():
            raise FileNotFoundError(
                selection_path
            )

        if not lock_path.exists():
            raise FileNotFoundError(
                lock_path
            )

        lock = json.loads(
            lock_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            lock.get(
                "selection_frozen"
            )
            is not True
        ):
            raise RuntimeError(
                f"{source['method']}: "
                "selection is not frozen."
            )

        if (
            lock.get(
                "test_evaluation_performed",
                False,
            )
            is not False
        ):
            raise RuntimeError(
                f"{source['method']}: "
                "lock does not indicate "
                "pre-test state."
            )

        expected_hash = (
            lock[
                "files"
            ].get(
                source[
                    "selection_file"
                ]
            )
        )

        actual_hash = sha256_file(
            selection_path
        )

        if (
            expected_hash
            is not None
            and
            expected_hash
            != actual_hash
        ):
            raise RuntimeError(
                f"{source['method']}: "
                "selection-file hash "
                "does not match lock."
            )

        df = pd.read_csv(
            selection_path
        )

        if len(df) != 600:
            raise RuntimeError(
                f"{source['method']}: "
                f"expected 600 rows, "
                f"observed {len(df)}."
            )

        required = {
            "task_id",
            "selected_config_id",
            source[
                "seed_column"
            ],
        }

        missing = (
            required
            - set(
                df.columns
            )
        )

        if missing:
            raise RuntimeError(
                f"{source['method']}: "
                f"missing {sorted(missing)}"
            )

        if (
            df[
                "task_id"
            ].nunique()
            != 120
        ):
            raise RuntimeError(
                f"{source['method']}: "
                "expected 120 tasks."
            )

        seeds = sorted(
            df[
                source[
                    "seed_column"
                ]
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        if seeds != [
            0, 1, 2, 3, 4
        ]:
            raise RuntimeError(
                f"{source['method']}: "
                f"unexpected seeds {seeds}"
            )

        counts = (
            df.groupby(
                "task_id"
            )
            .size()
        )

        if not (
            counts == 5
        ).all():
            raise RuntimeError(
                f"{source['method']}: "
                "not exactly 5 selections "
                "per task."
            )

        for row in (
            df.itertuples(
                index=False
            )
        ):
            task_id = str(
                getattr(
                    row,
                    "task_id"
                )
            )

            config_id = str(
                getattr(
                    row,
                    "selected_config_id"
                )
            )

            run_seed = int(
                getattr(
                    row,
                    source[
                        "seed_column"
                    ]
                )
            )

            all_rows.append(
                {
                    "method":
                        source[
                            "method"
                        ],

                    "task_id":
                        task_id,

                    "run_seed":
                        run_seed,

                    "selected_config_id":
                        config_id,

                    "pair_id":
                        pair_id(
                            task_id,
                            config_id,
                        ),
                }
            )

        source_metadata.append(
            {
                "method":
                    source[
                        "method"
                    ],

                "selection_path":
                    str(
                        selection_path
                    ),

                "selection_sha256":
                    actual_hash,

                "lock_path":
                    str(
                        lock_path
                    ),

                "lock_sha256":
                    sha256_file(
                        lock_path
                    ),
            }
        )

    rec = pd.DataFrame(
        all_rows
    )

    if len(rec) != 3000:
        raise RuntimeError(
            "Expected exactly 3000 "
            "frozen recommendation rows."
        )

    if (
        rec[
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

    unique_pairs = (
        rec[
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

    recommendations_path = (
        OUT
        / "frozen_recommendations.csv"
    )

    pairs_path = (
        OUT
        / "unique_task_config_pairs.csv"
    )

    rec.to_csv(
        recommendations_path,
        index=False,
    )

    unique_pairs.to_csv(
        pairs_path,
        index=False,
    )

    manifest = {
        "schema_version":
            1,

        "recommendation_rows":
            int(
                len(rec)
            ),

        "methods":
            5,

        "tasks_per_method":
            120,

        "runs_per_method":
            5,

        "unique_task_config_pairs":
            int(
                len(
                    unique_pairs
                )
            ),

        "final_training_episodes":
            200,

        "final_test_episodes":
            600,

        "final_model_seeds": [
            101,
            202,
            303,
        ],

        "selection_frozen_before_test":
            True,

        "sources":
            source_metadata,

        "files": {
            "frozen_recommendations.csv":
                sha256_file(
                    recommendations_path
                ),

            "unique_task_config_pairs.csv":
                sha256_file(
                    pairs_path
                ),
        },
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
    print("=" * 90)
    print(
        "FINAL TEST EVALUATION MANIFEST"
    )
    print("=" * 90)

    print(
        "Frozen recommendations:",
        len(rec),
    )

    print(
        "Methods:",
        rec[
            "method"
        ].nunique(),
    )

    print(
        "Tasks:",
        rec[
            "task_id"
        ].nunique(),
    )

    print(
        "Unique task/config pairs:",
        len(
            unique_pairs
        ),
    )

    print(
        "Model seeds:",
        [101, 202, 303],
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
        "Selection frozen before test: True"
    )

    print()
    print(
        "FINAL TEST MANIFEST: PASS"
    )


if __name__ == "__main__":
    main()
