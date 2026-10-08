from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.baselines.adapter_v2 import (
    BaselineAdapterV2,
    DEFAULT_ANCHORS,
    DEFAULT_MANIFEST_ROOT,
    FULL_TRAIN_EPISODES,
    VALIDATION_EPISODES,
    DEFAULT_SEARCH_MODEL_SEED,
    FINAL_MODEL_SEEDS,
)


DEFAULT_STAGE1_ROOT = Path(
    "results/oracles/final800_v2/test"
)

DEFAULT_OUTPUT_ROOT = Path(
    "results/hpo_baselines_v2/random_search"
)

# These are HPO-search randomness seeds.
# They are intentionally distinct in role from
# model-evaluation seeds 101/202/303.
DEFAULT_SEARCH_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_N_TRIALS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=DEFAULT_STAGE1_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
    )

    parser.add_argument(
        "--search-seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SEARCH_SEEDS
        ),
    )

    return parser.parse_args()


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


def stable_rng_seed(
    *,
    task_id: str,
    search_seed: int,
) -> int:
    """
    Derive a deterministic task-specific RNG seed.

    This means search seed 0 represents one
    independent Random Search run, but different
    tasks do not receive an artificially identical
    40-anchor subset.
    """
    payload = (
        "random_search_v2|"
        f"{search_seed}|"
        f"{task_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def canonical_json(
    value,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def validate_stage1_table(
    *,
    table: pd.DataFrame,
    task_id: str,
    expected_anchor_ids: list[str],
):
    required = {
        "config_id",
        "validation_accuracy_mean",
        "total_elapsed_seconds",
    }

    missing = (
        required
        - set(
            table.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{task_id}: missing "
            f"stage1 columns: "
            f"{sorted(missing)}"
        )

    if len(table) != 64:
        raise RuntimeError(
            f"{task_id}: expected "
            f"64 stage1 rows, "
            f"found {len(table)}"
        )

    config_ids = (
        table["config_id"]
        .astype(str)
        .tolist()
    )

    if (
        len(config_ids)
        != len(
            set(config_ids)
        )
    ):
        raise RuntimeError(
            f"{task_id}: duplicate "
            "config_id in stage1 table."
        )

    if (
        set(config_ids)
        != set(
            expected_anchor_ids
        )
    ):
        missing_anchors = sorted(
            set(
                expected_anchor_ids
            )
            - set(
                config_ids
            )
        )

        extra_anchors = sorted(
            set(
                config_ids
            )
            - set(
                expected_anchor_ids
            )
        )

        raise RuntimeError(
            f"{task_id}: stage1 anchor "
            "set differs from frozen "
            "64-anchor portfolio.\n"
            f"Missing: "
            f"{missing_anchors}\n"
            f"Extra: "
            f"{extra_anchors}"
        )

    accuracy = pd.to_numeric(
        table[
            "validation_accuracy_mean"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        accuracy
    ).all():
        raise RuntimeError(
            f"{task_id}: non-finite "
            "validation accuracy."
        )

    if (
        (accuracy < 0.0).any()
        or
        (accuracy > 1.0).any()
    ):
        raise RuntimeError(
            f"{task_id}: validation "
            "accuracy outside [0, 1]."
        )

    elapsed = pd.to_numeric(
        table[
            "total_elapsed_seconds"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        elapsed
    ).all():
        raise RuntimeError(
            f"{task_id}: non-finite "
            "elapsed time."
        )

    if (
        elapsed <= 0.0
    ).any():
        raise RuntimeError(
            f"{task_id}: non-positive "
            "elapsed time."
        )

    # Strong audit of the frozen stage-1
    # evaluation protocol when these
    # columns are present.
    if (
        "train_episodes"
        in table.columns
    ):
        values = set(
            pd.to_numeric(
                table[
                    "train_episodes"
                ],
                errors="raise",
            ).astype(int)
        )

        if values != {
            FULL_TRAIN_EPISODES
        }:
            raise RuntimeError(
                f"{task_id}: stage1 "
                "training budget is "
                f"{sorted(values)}, "
                "expected "
                f"{FULL_TRAIN_EPISODES}."
            )

    if (
        "validation_episodes"
        in table.columns
    ):
        values = set(
            pd.to_numeric(
                table[
                    "validation_episodes"
                ],
                errors="raise",
            ).astype(int)
        )

        if values != {
            VALIDATION_EPISODES
        }:
            raise RuntimeError(
                f"{task_id}: stage1 "
                "validation budget is "
                f"{sorted(values)}, "
                "expected "
                f"{VALIDATION_EPISODES}."
            )

    if (
        "model_seed_count"
        in table.columns
    ):
        values = set(
            pd.to_numeric(
                table[
                    "model_seed_count"
                ],
                errors="raise",
            ).astype(int)
        )

        if values != {1}:
            raise RuntimeError(
                f"{task_id}: stage1 "
                "must contain exactly "
                "one screening model seed."
            )

    if (
        "model_seeds_json"
        in table.columns
    ):
        for raw in (
            table[
                "model_seeds_json"
            ]
            .astype(str)
            .tolist()
        ):
            seeds = json.loads(
                raw
            )

            if seeds != [
                DEFAULT_SEARCH_MODEL_SEED
            ]:
                raise RuntimeError(
                    f"{task_id}: expected "
                    "stage1 model seed "
                    f"{DEFAULT_SEARCH_MODEL_SEED}, "
                    f"found {seeds}."
                )


def stage1_global_ranking(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministic ranking used only for
    diagnostics.

    Primary criterion:
        validation accuracy descending

    Tie-break:
        config_id ascending

    We deliberately do NOT use test accuracy
    or runtime as a quality criterion.
    """
    ranked = (
        table.copy()
        .assign(
            config_id=lambda df:
                df[
                    "config_id"
                ].astype(str)
        )
        .sort_values(
            by=[
                "validation_accuracy_mean",
                "config_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    ranked[
        "stage1_global_rank"
    ] = (
        np.arange(
            len(ranked)
        )
        + 1
    )

    return ranked


def select_random_search(
    *,
    stage1: pd.DataFrame,
    anchor_ids: list[str],
    task_id: str,
    search_seed: int,
    n_trials: int,
):
    rng_seed = stable_rng_seed(
        task_id=task_id,
        search_seed=search_seed,
    )

    rng = np.random.default_rng(
        rng_seed
    )

    evaluation_order = (
        rng.choice(
            np.asarray(
                anchor_ids,
                dtype=object,
            ),
            size=n_trials,
            replace=False,
        )
        .tolist()
    )

    indexed = (
        stage1
        .copy()
        .assign(
            config_id=lambda df:
                df[
                    "config_id"
                ].astype(str)
        )
        .set_index(
            "config_id",
            drop=False,
        )
    )

    candidate_rows = (
        indexed.loc[
            evaluation_order
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    candidate_rows[
        "evaluation_order"
    ] = np.arange(
        1,
        n_trials + 1,
    )

    # Random Search evaluates all n_trials
    # and then takes the highest validation
    # accuracy. Deterministic tie-break is
    # lexical config ID.
    ranked_candidates = (
        candidate_rows
        .sort_values(
            by=[
                "validation_accuracy_mean",
                "config_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    selected = (
        ranked_candidates.iloc[0]
    )

    return (
        rng_seed,
        evaluation_order,
        candidate_rows,
        selected,
    )


def main():
    args = parse_args()

    if not (
        1 <= args.n_trials <= 64
    ):
        raise ValueError(
            "n-trials must be in "
            "[1, 64]."
        )

    if (
        len(
            args.search_seeds
        )
        != len(
            set(
                args.search_seeds
            )
        )
    ):
        raise ValueError(
            "search-seeds contains "
            "duplicates."
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CPU is sufficient because Step 2 is
    # a replay over REAL previously measured
    # full-budget validation evaluations.
    adapter = BaselineAdapterV2(
        anchor_path=(
            DEFAULT_ANCHORS
        ),
        manifest_root=(
            DEFAULT_MANIFEST_ROOT
        ),
        device="cpu",
        image_size=84,
        split_seed=42,
    )

    anchor_ids = (
        adapter.anchor_ids()
    )

    if len(anchor_ids) != 64:
        raise RuntimeError(
            "Frozen anchor portfolio "
            "does not contain 64 anchors."
        )

    if args.task_id is None:
        task_ids = (
            adapter.task_ids(
                split="test"
            )
        )
    else:
        task = adapter.get_task(
            args.task_id
        )

        if task.split != "test":
            raise ValueError(
                "--task-id must identify "
                "a held-out meta-test task."
            )

        task_ids = [
            args.task_id
        ]

    search_start = (
        time.perf_counter()
    )

    selection_rows = []
    evaluation_rows = []

    for task_index, task_id in enumerate(
        task_ids,
        start=1,
    ):
        task = adapter.get_task(
            task_id
        )

        stage1_path = (
            args.stage1_root
            / task.dataset
            / task.task_id
            / "stage1_seed101_full64.csv"
        )

        if not stage1_path.exists():
            raise FileNotFoundError(
                stage1_path
            )

        stage1 = pd.read_csv(
            stage1_path
        )

        validate_stage1_table(
            table=stage1,
            task_id=task_id,
            expected_anchor_ids=(
                anchor_ids
            ),
        )

        stage1_hash = (
            sha256_file(
                stage1_path
            )
        )

        global_ranked = (
            stage1_global_ranking(
                stage1
            )
        )

        global_best = (
            global_ranked.iloc[0]
        )

        global_best_config = str(
            global_best[
                "config_id"
            ]
        )

        global_best_accuracy = float(
            global_best[
                "validation_accuracy_mean"
            ]
        )

        global_rank_map = dict(
            zip(
                global_ranked[
                    "config_id"
                ].astype(str),
                global_ranked[
                    "stage1_global_rank"
                ].astype(int),
            )
        )

        print(
            f"[{task_index:3d}/"
            f"{len(task_ids):3d}] "
            f"{task_id}"
        )

        for search_seed in (
            args.search_seeds
        ):
            algorithm_start = (
                time.perf_counter()
            )

            (
                derived_seed,
                evaluation_order,
                candidate_rows,
                selected,
            ) = select_random_search(
                stage1=stage1,
                anchor_ids=(
                    anchor_ids
                ),
                task_id=task_id,
                search_seed=(
                    int(
                        search_seed
                    )
                ),
                n_trials=(
                    args.n_trials
                ),
            )

            algorithm_overhead = (
                time.perf_counter()
                - algorithm_start
            )

            selected_config = str(
                selected[
                    "config_id"
                ]
            )

            selected_accuracy = float(
                selected[
                    "validation_accuracy_mean"
                ]
            )

            measured_serial_time = float(
                candidate_rows[
                    "total_elapsed_seconds"
                ]
                .astype(float)
                .sum()
            )

            best_value_hit = bool(
                np.isclose(
                    selected_accuracy,
                    global_best_accuracy,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

            candidate_payload = [
                str(config_id)
                for config_id
                in evaluation_order
            ]

            candidate_json = (
                canonical_json(
                    candidate_payload
                )
            )

            candidate_sha256 = (
                hashlib.sha256(
                    candidate_json.encode(
                        "utf-8"
                    )
                )
                .hexdigest()
            )

            selection_rows.append(
                {
                    "method":
                        "random_search",

                    "task_id":
                        task.task_id,

                    "dataset":
                        task.dataset,

                    "regime":
                        (
                            f"{task.n_way}w"
                            f"{task.n_shot}s"
                        ),

                    "n_way":
                        task.n_way,

                    "n_shot":
                        task.n_shot,

                    "n_query":
                        task.n_query,

                    "search_seed":
                        int(
                            search_seed
                        ),

                    "derived_rng_seed":
                        int(
                            derived_seed
                        ),

                    "n_trials":
                        int(
                            args.n_trials
                        ),

                    "selected_config_id":
                        selected_config,

                    "selected_validation_accuracy":
                        selected_accuracy,

                    "stage1_global_best_config_id":
                        global_best_config,

                    "stage1_global_best_validation_accuracy":
                        global_best_accuracy,

                    "selected_global_rank":
                        int(
                            global_rank_map[
                                selected_config
                            ]
                        ),

                    "validation_regret_to_stage1_best_pp":
                        float(
                            100.0
                            * (
                                global_best_accuracy
                                - selected_accuracy
                            )
                        ),

                    "exact_global_best_config_hit":
                        bool(
                            selected_config
                            == global_best_config
                        ),

                    "global_best_value_hit":
                        best_value_hit,

                    "configuration_evaluations":
                        int(
                            args.n_trials
                        ),

                    "full_budget_equivalent_evaluations":
                        float(
                            args.n_trials
                        ),

                    "training_episodes_consumed":
                        int(
                            args.n_trials
                            * FULL_TRAIN_EPISODES
                        ),

                    # This is the sum of REAL measured
                    # evaluator runtimes from the
                    # frozen stage1 cache.
                    "measured_serial_search_time_seconds":
                        measured_serial_time,

                    # This is only replay/sampling
                    # overhead, not model-training cost.
                    "replay_algorithm_overhead_seconds":
                        float(
                            algorithm_overhead
                        ),

                    "candidate_ids_json":
                        candidate_json,

                    "candidate_ids_sha256":
                        candidate_sha256,

                    "stage1_source":
                        str(
                            stage1_path
                        ),

                    "stage1_sha256":
                        stage1_hash,

                    "search_model_seed":
                        int(
                            DEFAULT_SEARCH_MODEL_SEED
                        ),

                    "train_episodes_per_evaluation":
                        int(
                            FULL_TRAIN_EPISODES
                        ),

                    "validation_episodes_per_evaluation":
                        int(
                            VALIDATION_EPISODES
                        ),

                    "test_bank_used":
                        False,
                }
            )

            for _, row in (
                candidate_rows.iterrows()
            ):
                config_id = str(
                    row[
                        "config_id"
                    ]
                )

                evaluation_rows.append(
                    {
                        "method":
                            "random_search",

                        "task_id":
                            task.task_id,

                        "dataset":
                            task.dataset,

                        "regime":
                            (
                                f"{task.n_way}w"
                                f"{task.n_shot}s"
                            ),

                        "search_seed":
                            int(
                                search_seed
                            ),

                        "derived_rng_seed":
                            int(
                                derived_seed
                            ),

                        "evaluation_order":
                            int(
                                row[
                                    "evaluation_order"
                                ]
                            ),

                        "config_id":
                            config_id,

                        "validation_accuracy":
                            float(
                                row[
                                    "validation_accuracy_mean"
                                ]
                            ),

                        "stage1_global_rank":
                            int(
                                global_rank_map[
                                    config_id
                                ]
                            ),

                        "elapsed_seconds":
                            float(
                                row[
                                    "total_elapsed_seconds"
                                ]
                            ),

                        "training_episodes":
                            int(
                                FULL_TRAIN_EPISODES
                            ),

                        "validation_episodes":
                            int(
                                VALIDATION_EPISODES
                            ),

                        "model_seed":
                            int(
                                DEFAULT_SEARCH_MODEL_SEED
                            ),

                        "test_bank_used":
                            False,
                    }
                )

    selections = pd.DataFrame(
        selection_rows
    )

    evaluations = pd.DataFrame(
        evaluation_rows
    )

    selection_path = (
        args.output_root
        / "search_selections.csv"
    )

    evaluation_path = (
        args.output_root
        / "candidate_evaluations.csv"
    )

    selections.to_csv(
        selection_path,
        index=False,
    )

    evaluations.to_csv(
        evaluation_path,
        index=False,
    )

    protocol = {
        "schema_version":
            1,

        "method":
            "random_search",

        "algorithm":
            (
                "uniform random search "
                "without replacement over "
                "the frozen 64-anchor "
                "portfolio"
            ),

        "task_split":
            "test",

        "task_count":
            len(
                task_ids
            ),

        "search_seeds":
            [
                int(seed)
                for seed
                in args.search_seeds
            ],

        "search_seed_role":
            (
                "Controls candidate subset "
                "sampling only."
            ),

        "search_model_seed":
            int(
                DEFAULT_SEARCH_MODEL_SEED
            ),

        "final_model_seeds_reserved_for_later_test_evaluation":
            [
                int(seed)
                for seed
                in FINAL_MODEL_SEEDS
            ],

        "n_trials":
            int(
                args.n_trials
            ),

        "sampling":
            "without replacement",

        "search_space":
            "frozen V2 64-anchor portfolio",

        "anchor_manifest":
            str(
                DEFAULT_ANCHORS
            ),

        "anchor_manifest_sha256":
            sha256_file(
                DEFAULT_ANCHORS
            ),

        "resource_type":
            "training episodes",

        "train_episodes_per_evaluation":
            int(
                FULL_TRAIN_EPISODES
            ),

        "validation_episodes_per_evaluation":
            int(
                VALIDATION_EPISODES
            ),

        "objective":
            "maximize validation accuracy",

        "tie_break":
            (
                "validation accuracy "
                "descending, then config_id "
                "ascending"
            ),

        "validation_source":
            (
                "real frozen "
                "stage1_seed101_full64.csv "
                "evaluations"
            ),

        "test_bank_used":
            False,

        "test_evaluation_performed":
            False,

        "important_note":
            (
                "No held-out test accuracy "
                "is read or used by this "
                "selection script. Final "
                "test evaluation is deferred "
                "until all baseline methods "
                "have frozen selections."
            ),
    }

    protocol_path = (
        args.output_root
        / "protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    expected_selection_rows = (
        len(
            task_ids
        )
        * len(
            args.search_seeds
        )
    )

    expected_evaluation_rows = (
        expected_selection_rows
        * args.n_trials
    )

    if (
        len(selections)
        != expected_selection_rows
    ):
        raise RuntimeError(
            "Unexpected selection "
            "row count."
        )

    if (
        len(evaluations)
        != expected_evaluation_rows
    ):
        raise RuntimeError(
            "Unexpected candidate "
            "evaluation row count."
        )

    # Every task/run must contain exactly
    # n_trials UNIQUE configurations.
    candidate_counts = (
        evaluations
        .groupby(
            [
                "task_id",
                "search_seed",
            ]
        )[
            "config_id"
        ]
        .nunique()
    )

    if not (
        candidate_counts
        == args.n_trials
    ).all():
        raise RuntimeError(
            "Random Search contained "
            "duplicate candidates within "
            "at least one task/run."
        )

    if (
        selections[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test bank leakage detected."
        )

    if (
        evaluations[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test bank leakage detected."
        )

    # --------------------------------------------------------
    # Freeze/lock the Random Search selections BEFORE any
    # held-out test evaluation.
    # --------------------------------------------------------

    lock_manifest = {
        "schema_version":
            1,

        "method":
            "random_search",

        "selection_frozen":
            True,

        "test_evaluation_performed":
            False,

        "files": {
            "protocol.json":
                sha256_file(
                    protocol_path
                ),

            "search_selections.csv":
                sha256_file(
                    selection_path
                ),

            "candidate_evaluations.csv":
                sha256_file(
                    evaluation_path
                ),
        },
    }

    lock_path = (
        args.output_root
        / "selection_lock_manifest.json"
    )

    lock_path.write_text(
        json.dumps(
            lock_manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sha_path = (
        args.output_root
        / "selection_files.sha256"
    )

    lines = []

    for filename in (
        "protocol.json",
        "search_selections.csv",
        "candidate_evaluations.csv",
        "selection_lock_manifest.json",
    ):
        path = (
            args.output_root
            / filename
        )

        lines.append(
            f"{sha256_file(path)}  "
            f"{filename}"
        )

    sha_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    elapsed = (
        time.perf_counter()
        - search_start
    )

    print()
    print("=" * 90)
    print(
        "RANDOM SEARCH SELECTION SUMMARY"
    )
    print("=" * 90)

    print(
        "Tasks:",
        selections[
            "task_id"
        ].nunique(),
    )

    print(
        "Search seeds:",
        sorted(
            selections[
                "search_seed"
            ].unique()
            .tolist()
        ),
    )

    print(
        "Selections:",
        len(
            selections
        ),
    )

    print(
        "Candidate evaluations:",
        len(
            evaluations
        ),
    )

    print(
        "Trials per task/run:",
        args.n_trials,
    )

    print(
        "Exact global-best config hit rate:",
        f"{100.0 * selections['exact_global_best_config_hit'].mean():.2f}%",
    )

    print(
        "Global-best validation-value hit rate:",
        f"{100.0 * selections['global_best_value_hit'].mean():.2f}%",
    )

    print(
        "Mean validation regret to "
        "stage1 best:",
        f"{selections['validation_regret_to_stage1_best_pp'].mean():.4f} pp",
    )

    print(
        "Median validation regret:",
        f"{selections['validation_regret_to_stage1_best_pp'].median():.4f} pp",
    )

    print(
        "Mean measured serial "
        "search time:",
        f"{selections['measured_serial_search_time_seconds'].mean():.2f} s",
    )

    print(
        "Unique selected anchors:",
        selections[
            "selected_config_id"
        ].nunique(),
    )

    print(
        "Replay script elapsed:",
        f"{elapsed:.2f} s",
    )

    print(
        "Test bank used: False"
    )

    print(
        "Selection lock:",
        lock_path,
    )

    print()
    print(
        "RANDOM SEARCH SELECTION: PASS"
    )


if __name__ == "__main__":
    main()
