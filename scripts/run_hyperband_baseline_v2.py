from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

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
    "results/hpo_baselines_v2/hyperband"
)

DEFAULT_CACHE_ROOT = Path(
    "results/hpo_baselines_v2/"
    "hyperband_cache"
)

DEFAULT_SEARCH_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

MIN_RESOURCE = 25
MAX_RESOURCE = 200
ETA = 2


def parse_args():
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
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )

    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--search-seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SEARCH_SEEDS
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
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


def stable_seed(
    *,
    task_id: str,
    search_seed: int,
    bracket: int,
) -> int:
    payload = (
        "exact_hyperband_v2|"
        f"{task_id}|"
        f"{search_seed}|"
        f"{bracket}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def hyperband_schedule():
    """
    Exact Hyperband bracket construction over
    the declared resource range [25, 200]
    with eta=2.

    Generalized s_max:
        floor(log_eta(R / r_min))
    """

    ratio = (
        MAX_RESOURCE
        / MIN_RESOURCE
    )

    s_max = int(
        math.floor(
            math.log(
                ratio,
                ETA,
            )
            + 1e-12
        )
    )

    if (
        MIN_RESOURCE
        * (ETA ** s_max)
        != MAX_RESOURCE
    ):
        raise RuntimeError(
            "Current implementation "
            "expects max/min resource "
            "to be an exact eta power."
        )

    B = (
        s_max + 1
    ) * MAX_RESOURCE

    brackets = []

    for s in reversed(
        range(
            s_max + 1
        )
    ):
        n = int(
            math.ceil(
                (
                    B
                    / MAX_RESOURCE
                )
                * (
                    ETA ** s
                )
                / (
                    s + 1
                )
            )
        )

        r = (
            MAX_RESOURCE
            * ETA ** (-s)
        )

        rungs = []

        for i in range(
            s + 1
        ):
            n_i = int(
                math.floor(
                    n
                    * ETA ** (-i)
                )
            )

            r_i = int(
                round(
                    r
                    * ETA ** i
                )
            )

            if (
                r_i
                < MIN_RESOURCE
                or
                r_i
                > MAX_RESOURCE
            ):
                raise RuntimeError(
                    "Invalid Hyperband "
                    "resource schedule."
                )

            if i < s:
                keep = max(
                    1,
                    int(
                        math.floor(
                            n_i
                            / ETA
                        )
                    ),
                )
            else:
                keep = None

            rungs.append(
                {
                    "rung":
                        i,

                    "n_configs":
                        n_i,

                    "resource":
                        r_i,

                    "keep":
                        keep,
                }
            )

        brackets.append(
            {
                "s":
                    s,

                "initial_n":
                    n,

                "initial_resource":
                    int(
                        round(r)
                    ),

                "rungs":
                    rungs,
            }
        )

    return {
        "s_max":
            s_max,

        "B":
            B,

        "brackets":
            brackets,
    }


def print_schedule(
    schedule,
):
    print()
    print(
        "=" * 88
    )

    print(
        "EXACT HYPERBAND SCHEDULE"
    )

    print(
        "=" * 88
    )

    print(
        "min resource:",
        MIN_RESOURCE,
    )

    print(
        "max resource:",
        MAX_RESOURCE,
    )

    print(
        "eta:",
        ETA,
    )

    print(
        "s_max:",
        schedule[
            "s_max"
        ],
    )

    print(
        "B:",
        schedule[
            "B"
        ],
    )

    total_resource = 0
    total_evaluations = 0

    for bracket in (
        schedule[
            "brackets"
        ]
    ):
        pieces = []

        bracket_resource = 0

        for rung in (
            bracket[
                "rungs"
            ]
        ):
            pieces.append(
                f"{rung['n_configs']}"
                f"@{rung['resource']}"
            )

            bracket_resource += (
                rung[
                    "n_configs"
                ]
                * rung[
                    "resource"
                ]
            )

            total_evaluations += (
                rung[
                    "n_configs"
                ]
            )

        total_resource += (
            bracket_resource
        )

        print(
            f"s={bracket['s']}: "
            + " -> ".join(
                pieces
            )
            + " | resource="
            + str(
                bracket_resource
            )
        )

    print()
    print(
        "Logical evaluations:",
        total_evaluations,
    )

    print(
        "Total training episodes:",
        total_resource,
    )

    print(
        "Full-budget equivalents:",
        total_resource
        / MAX_RESOURCE,
    )

    return (
        total_evaluations,
        total_resource,
    )


def cache_path(
    *,
    cache_root: Path,
    task_id: str,
    config_id: str,
    budget: int,
    model_seed: int,
) -> Path:

    return (
        cache_root
        / task_id
        / (
            f"{config_id}"
            f"__budget{budget:03d}"
            f"__seed{model_seed}.json"
        )
    )


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.parent
        / (
            path.name
            + f".tmp.{os.getpid()}"
        )
    )

    temp_path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp_path,
        path,
    )


def validate_cached_result(
    *,
    value: dict,
    task_id: str,
    config_id: str,
    budget: int,
    model_seed: int,
):
    required = {
        "task_id",
        "config_id",
        "training_episodes",
        "validation_episodes",
        "model_seed",
        "validation_accuracy",
        "elapsed_seconds",
        "test_bank_used",
    }

    missing = (
        required
        - set(
            value
        )
    )

    if missing:
        raise RuntimeError(
            "Incomplete Hyperband cache: "
            f"{sorted(missing)}"
        )

    if (
        str(
            value[
                "task_id"
            ]
        )
        != task_id
    ):
        raise RuntimeError(
            "Cached task mismatch."
        )

    if (
        str(
            value[
                "config_id"
            ]
        )
        != config_id
    ):
        raise RuntimeError(
            "Cached config mismatch."
        )

    if (
        int(
            value[
                "training_episodes"
            ]
        )
        != int(
            budget
        )
    ):
        raise RuntimeError(
            "Cached budget mismatch."
        )

    if (
        int(
            value[
                "validation_episodes"
            ]
        )
        != VALIDATION_EPISODES
    ):
        raise RuntimeError(
            "Cached validation "
            "budget mismatch."
        )

    if (
        int(
            value[
                "model_seed"
            ]
        )
        != int(
            model_seed
        )
    ):
        raise RuntimeError(
            "Cached model seed mismatch."
        )

    if bool(
        value[
            "test_bank_used"
        ]
    ):
        raise RuntimeError(
            "Test leakage in cache."
        )

    accuracy = float(
        value[
            "validation_accuracy"
        ]
    )

    if not (
        0.0 <= accuracy <= 1.0
    ):
        raise RuntimeError(
            "Cached validation "
            "accuracy outside [0,1]."
        )


class HyperbandEvaluator:
    def __init__(
        self,
        *,
        adapter: BaselineAdapterV2,
        stage1_root: Path,
        cache_root: Path,
    ):
        self.adapter = adapter

        self.stage1_root = (
            stage1_root
        )

        self.cache_root = (
            cache_root
        )

        self._stage1_tables = {}

        self.new_evaluations = 0
        self.cache_hits = 0
        self.stage1_hits = 0

    def _load_stage1(
        self,
        task_id: str,
    ) -> pd.DataFrame:
        if (
            task_id
            in self._stage1_tables
        ):
            return (
                self._stage1_tables[
                    task_id
                ]
            )

        task = (
            self.adapter.get_task(
                task_id
            )
        )

        path = (
            self.stage1_root
            / task.dataset
            / task.task_id
            / "stage1_seed101_full64.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        table = pd.read_csv(
            path
        )

        if len(table) != 64:
            raise RuntimeError(
                f"{task_id}: expected "
                "64 stage1 rows."
            )

        table[
            "config_id"
        ] = (
            table[
                "config_id"
            ].astype(str)
        )

        expected = set(
            self.adapter.anchor_ids()
        )

        observed = set(
            table[
                "config_id"
            ]
        )

        if observed != expected:
            raise RuntimeError(
                f"{task_id}: stage1 "
                "portfolio mismatch."
            )

        if (
            "train_episodes"
            in table.columns
        ):
            budgets = set(
                pd.to_numeric(
                    table[
                        "train_episodes"
                    ]
                ).astype(int)
            )

            if budgets != {
                MAX_RESOURCE
            }:
                raise RuntimeError(
                    f"{task_id}: stage1 "
                    "max-budget mismatch."
                )

        if (
            "validation_episodes"
            in table.columns
        ):
            budgets = set(
                pd.to_numeric(
                    table[
                        "validation_episodes"
                    ]
                ).astype(int)
            )

            if budgets != {
                VALIDATION_EPISODES
            }:
                raise RuntimeError(
                    f"{task_id}: stage1 "
                    "validation-budget "
                    "mismatch."
                )

        self._stage1_tables[
            task_id
        ] = table

        return table

    def evaluate(
        self,
        *,
        task_id: str,
        config_id: str,
        budget: int,
        model_seed: int,
    ) -> dict[str, Any]:

        budget = int(
            budget
        )

        if budget not in {
            25,
            50,
            100,
            200,
        }:
            raise ValueError(
                "Unexpected Hyperband "
                f"budget: {budget}"
            )

        # ---------------------------------
        # Full budget already genuinely
        # exists from final oracle stage1.
        # ---------------------------------

        if budget == MAX_RESOURCE:
            table = (
                self._load_stage1(
                    task_id
                )
            )

            rows = table[
                table[
                    "config_id"
                ]
                == config_id
            ]

            if len(rows) != 1:
                raise RuntimeError(
                    "Could not uniquely "
                    "resolve full-budget "
                    "stage1 result."
                )

            row = rows.iloc[0]

            self.stage1_hits += 1

            return {
                "task_id":
                    task_id,

                "config_id":
                    config_id,

                "training_episodes":
                    MAX_RESOURCE,

                "validation_episodes":
                    VALIDATION_EPISODES,

                "model_seed":
                    int(
                        model_seed
                    ),

                "validation_accuracy":
                    float(
                        row[
                            "validation_accuracy_mean"
                        ]
                    ),

                "elapsed_seconds":
                    float(
                        row[
                            "total_elapsed_seconds"
                        ]
                    ),

                "source":
                    "frozen_stage1_full64",

                "physically_executed_now":
                    False,

                "test_bank_used":
                    False,
            }

        path = cache_path(
            cache_root=(
                self.cache_root
            ),

            task_id=task_id,

            config_id=config_id,

            budget=budget,

            model_seed=model_seed,
        )

        # ---------------------------------
        # Reuse a genuine previously run
        # partial-budget evaluation.
        # ---------------------------------

        if path.exists():
            value = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            validate_cached_result(
                value=value,

                task_id=task_id,

                config_id=config_id,

                budget=budget,

                model_seed=model_seed,
            )

            value = dict(
                value
            )

            value[
                "source"
            ] = "partial_budget_cache"

            value[
                "physically_executed_now"
            ] = False

            self.cache_hits += 1

            return value

        # ---------------------------------
        # Genuine new ProtoNet run.
        # ---------------------------------

        result = (
            self.adapter
            .train_validate_anchor(
                task_id=task_id,

                config_id=config_id,

                budget=budget,

                model_seed=model_seed,

                validation_episodes=(
                    VALIDATION_EPISODES
                ),
            )
        )

        value = {
            "task_id":
                task_id,

            "config_id":
                config_id,

            "training_episodes":
                int(
                    budget
                ),

            "validation_episodes":
                int(
                    VALIDATION_EPISODES
                ),

            "model_seed":
                int(
                    model_seed
                ),

            "validation_accuracy":
                float(
                    result[
                        "validation_accuracy_mean"
                    ]
                ),

            "elapsed_seconds":
                float(
                    result[
                        "total_elapsed_seconds"
                    ]
                ),

            "source":
                "new_partial_budget_evaluation",

            "physically_executed_now":
                True,

            "test_bank_used":
                False,
        }

        atomic_write_json(
            path,
            value,
        )

        self.new_evaluations += 1

        return value


def deterministic_rank(
    rows,
):
    return sorted(
        rows,
        key=lambda row: (
            -float(
                row[
                    "validation_accuracy"
                ]
            ),
            str(
                row[
                    "config_id"
                ]
            ),
        ),
    )


def sample_initial_configs(
    *,
    anchor_ids,
    n,
    task_id,
    search_seed,
    bracket,
):
    derived_seed = (
        stable_seed(
            task_id=task_id,

            search_seed=(
                search_seed
            ),

            bracket=bracket,
        )
    )

    rng = np.random.default_rng(
        derived_seed
    )

    configs = (
        rng.choice(
            np.asarray(
                anchor_ids,
                dtype=object,
            ),

            size=int(
                n
            ),

            replace=False,
        )
        .tolist()
    )

    return (
        int(
            derived_seed
        ),
        [
            str(config)
            for config
            in configs
        ],
    )


def run_hyperband_once(
    *,
    task_id,
    search_seed,
    schedule,
    evaluator,
    anchor_ids,
):
    trace = []
    terminal_rows = []

    logical_eval_index = 0
    logical_training_episodes = 0
    logical_elapsed_seconds = 0.0

    physical_new_at_start = (
        evaluator.new_evaluations
    )

    cache_hits_at_start = (
        evaluator.cache_hits
    )

    stage1_hits_at_start = (
        evaluator.stage1_hits
    )

    run_start = (
        time.perf_counter()
    )

    for bracket in (
        schedule[
            "brackets"
        ]
    ):
        s = int(
            bracket[
                "s"
            ]
        )

        (
            bracket_seed,
            current_configs,
        ) = sample_initial_configs(
            anchor_ids=anchor_ids,

            n=(
                bracket[
                    "initial_n"
                ]
            ),

            task_id=task_id,

            search_seed=(
                search_seed
            ),

            bracket=s,
        )

        for rung in (
            bracket[
                "rungs"
            ]
        ):
            expected_n = int(
                rung[
                    "n_configs"
                ]
            )

            budget = int(
                rung[
                    "resource"
                ]
            )

            if (
                len(
                    current_configs
                )
                != expected_n
            ):
                raise RuntimeError(
                    "Exact Hyperband "
                    "population-size mismatch: "
                    f"task={task_id}, "
                    f"seed={search_seed}, "
                    f"s={s}, "
                    f"rung={rung['rung']}, "
                    f"expected={expected_n}, "
                    f"observed="
                    f"{len(current_configs)}"
                )

            scored = []

            for config_id in (
                current_configs
            ):
                logical_eval_index += 1

                result = evaluator.evaluate(
                    task_id=task_id,

                    config_id=config_id,

                    budget=budget,

                    model_seed=(
                        DEFAULT_SEARCH_MODEL_SEED
                    ),
                )

                logical_training_episodes += (
                    budget
                )

                logical_elapsed_seconds += float(
                    result[
                        "elapsed_seconds"
                    ]
                )

                row = {
                    "method":
                        "hyperband",

                    "task_id":
                        task_id,

                    "search_seed":
                        int(
                            search_seed
                        ),

                    "bracket_s":
                        s,

                    "bracket_rng_seed":
                        int(
                            bracket_seed
                        ),

                    "rung":
                        int(
                            rung[
                                "rung"
                            ]
                        ),

                    "logical_evaluation_index":
                        int(
                            logical_eval_index
                        ),

                    "config_id":
                        config_id,

                    "training_episodes":
                        budget,

                    "validation_episodes":
                        VALIDATION_EPISODES,

                    "model_seed":
                        DEFAULT_SEARCH_MODEL_SEED,

                    "validation_accuracy":
                        float(
                            result[
                                "validation_accuracy"
                            ]
                        ),

                    "elapsed_seconds":
                        float(
                            result[
                                "elapsed_seconds"
                            ]
                        ),

                    "evaluation_source":
                        str(
                            result[
                                "source"
                            ]
                        ),

                    "physically_executed_now":
                        bool(
                            result[
                                "physically_executed_now"
                            ]
                        ),

                    "test_bank_used":
                        False,
                }

                trace.append(
                    row
                )

                scored.append(
                    row
                )

            ranked = (
                deterministic_rank(
                    scored
                )
            )

            if (
                budget
                == MAX_RESOURCE
            ):
                terminal_rows.extend(
                    ranked
                )

            if (
                rung[
                    "keep"
                ]
                is not None
            ):
                keep = int(
                    rung[
                        "keep"
                    ]
                )

                current_configs = [
                    str(
                        row[
                            "config_id"
                        ]
                    )
                    for row
                    in ranked[:keep]
                ]

    if not terminal_rows:
        raise RuntimeError(
            "Hyperband produced no "
            "full-budget terminal "
            "evaluations."
        )

    ranked_terminal = (
        deterministic_rank(
            terminal_rows
        )
    )

    selected = (
        ranked_terminal[0]
    )

    # All terminal comparisons are at the
    # same full 200-episode budget.
    if any(
        int(
            row[
                "training_episodes"
            ]
        )
        != MAX_RESOURCE
        for row
        in terminal_rows
    ):
        raise RuntimeError(
            "Non-full-budget result "
            "entered Hyperband final "
            "selection."
        )

    return {
        "selected_config_id":
            str(
                selected[
                    "config_id"
                ]
            ),

        "selected_validation_accuracy":
            float(
                selected[
                    "validation_accuracy"
                ]
            ),

        "terminal_full_budget_count":
            int(
                len(
                    terminal_rows
                )
            ),

        "terminal_unique_config_count":
            int(
                len(
                    {
                        row[
                            "config_id"
                        ]
                        for row
                        in terminal_rows
                    }
                )
            ),

        "logical_evaluations":
            int(
                logical_eval_index
            ),

        "logical_training_episodes":
            int(
                logical_training_episodes
            ),

        "full_budget_equivalent_evaluations":
            float(
                logical_training_episodes
                / MAX_RESOURCE
            ),

        "logical_measured_serial_time_seconds":
            float(
                logical_elapsed_seconds
            ),

        "physical_new_partial_evaluations":
            int(
                evaluator.new_evaluations
                - physical_new_at_start
            ),

        "cache_hits":
            int(
                evaluator.cache_hits
                - cache_hits_at_start
            ),

        "stage1_hits":
            int(
                evaluator.stage1_hits
                - stage1_hits_at_start
            ),

        "actual_run_wall_time_seconds":
            float(
                time.perf_counter()
                - run_start
            ),

        "trace":
            trace,
    }


def global_stage1_best(
    *,
    evaluator,
    task_id,
):
    table = (
        evaluator._load_stage1(
            task_id
        )
        .copy()
    )

    table = table.sort_values(
        by=[
            "validation_accuracy_mean",
            "config_id",
        ],

        ascending=[
            False,
            True,
        ],

        kind="mergesort",
    ).reset_index(
        drop=True
    )

    row = table.iloc[0]

    return (
        str(
            row[
                "config_id"
            ]
        ),
        float(
            row[
                "validation_accuracy_mean"
            ]
        ),
    )


def main():
    args = parse_args()

    if len(
        args.search_seeds
    ) != len(
        set(
            args.search_seeds
        )
    ):
        raise ValueError(
            "Duplicate search seeds."
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule = (
        hyperband_schedule()
    )

    (
        logical_evals_expected,
        resource_expected,
    ) = print_schedule(
        schedule
    )

    if (
        logical_evals_expected
        != 35
    ):
        raise RuntimeError(
            "Expected 35 logical "
            "Hyperband evaluations "
            "per task/run."
        )

    if (
        resource_expected
        != 3200
    ):
        raise RuntimeError(
            "Expected 3200 training "
            "episodes per task/run."
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

    anchor_ids = (
        adapter.anchor_ids()
    )

    if len(
        anchor_ids
    ) != 64:
        raise RuntimeError(
            "Expected 64 anchors."
        )

    evaluator = (
        HyperbandEvaluator(
            adapter=adapter,

            stage1_root=(
                args.stage1_root
            ),

            cache_root=(
                args.cache_root
            ),
        )
    )

    if args.task_id is None:
        task_ids = (
            adapter.task_ids(
                split="test"
            )
        )
    else:
        task = (
            adapter.get_task(
                args.task_id
            )
        )

        if task.split != "test":
            raise ValueError(
                "--task-id must belong "
                "to test meta-split."
            )

        task_ids = [
            args.task_id
        ]

    selections = []
    trace_rows = []

    overall_start = (
        time.perf_counter()
    )

    for task_index, task_id in (
        enumerate(
            task_ids,
            start=1,
        )
    ):
        task = adapter.get_task(
            task_id
        )

        (
            best_stage1_id,
            best_stage1_value,
        ) = global_stage1_best(
            evaluator=evaluator,
            task_id=task_id,
        )

        print()
        print(
            f"[{task_index:3d}/"
            f"{len(task_ids):3d}] "
            f"{task_id}"
        )

        for search_seed in (
            args.search_seeds
        ):
            result = (
                run_hyperband_once(
                    task_id=task_id,

                    search_seed=int(
                        search_seed
                    ),

                    schedule=schedule,

                    evaluator=evaluator,

                    anchor_ids=anchor_ids,
                )
            )

            selected_id = (
                result[
                    "selected_config_id"
                ]
            )

            selected_value = float(
                result[
                    "selected_validation_accuracy"
                ]
            )

            best_value_hit = bool(
                np.isclose(
                    selected_value,
                    best_stage1_value,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

            selections.append(
                {
                    "method":
                        "hyperband",

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

                    "selected_config_id":
                        selected_id,

                    "selected_validation_accuracy":
                        selected_value,

                    "stage1_global_best_config_id":
                        best_stage1_id,

                    "stage1_global_best_validation_accuracy":
                        best_stage1_value,

                    "validation_regret_to_stage1_best_pp":
                        float(
                            100.0
                            * (
                                best_stage1_value
                                - selected_value
                            )
                        ),

                    "exact_global_best_config_hit":
                        bool(
                            selected_id
                            == best_stage1_id
                        ),

                    "global_best_validation_value_hit":
                        best_value_hit,

                    "logical_evaluations":
                        result[
                            "logical_evaluations"
                        ],

                    "logical_training_episodes":
                        result[
                            "logical_training_episodes"
                        ],

                    "full_budget_equivalent_evaluations":
                        result[
                            "full_budget_equivalent_evaluations"
                        ],

                    "terminal_full_budget_count":
                        result[
                            "terminal_full_budget_count"
                        ],

                    "terminal_unique_config_count":
                        result[
                            "terminal_unique_config_count"
                        ],

                    "logical_measured_serial_time_seconds":
                        result[
                            "logical_measured_serial_time_seconds"
                        ],

                    "physical_new_partial_evaluations":
                        result[
                            "physical_new_partial_evaluations"
                        ],

                    "cache_hits":
                        result[
                            "cache_hits"
                        ],

                    "stage1_hits":
                        result[
                            "stage1_hits"
                        ],

                    "actual_run_wall_time_seconds":
                        result[
                            "actual_run_wall_time_seconds"
                        ],

                    "search_model_seed":
                        DEFAULT_SEARCH_MODEL_SEED,

                    "test_bank_used":
                        False,
                }
            )

            trace_rows.extend(
                result[
                    "trace"
                ]
            )

            print(
                "  seed="
                f"{search_seed} "
                "selected="
                f"{selected_id} "
                "val="
                f"{selected_value:.4f} "
                "regret="
                f"{100*(best_stage1_value-selected_value):.4f}"
                "pp "
                "new="
                f"{result['physical_new_partial_evaluations']} "
                "cache="
                f"{result['cache_hits']} "
                "stage1="
                f"{result['stage1_hits']}"
            )

        # Incremental checkpoint after every task.
        pd.DataFrame(
            selections
        ).to_csv(
            args.output_root
            / "search_selections.partial.csv",
            index=False,
        )

        pd.DataFrame(
            trace_rows
        ).to_csv(
            args.output_root
            / "search_trace.partial.csv",
            index=False,
        )

    selections_df = (
        pd.DataFrame(
            selections
        )
    )

    trace_df = (
        pd.DataFrame(
            trace_rows
        )
    )

    expected_selections = (
        len(
            task_ids
        )
        * len(
            args.search_seeds
        )
    )

    expected_trace = (
        expected_selections
        * logical_evals_expected
    )

    if (
        len(
            selections_df
        )
        != expected_selections
    ):
        raise RuntimeError(
            "Unexpected selection "
            "count."
        )

    if (
        len(
            trace_df
        )
        != expected_trace
    ):
        raise RuntimeError(
            "Unexpected Hyperband "
            "trace count."
        )

    if not (
        selections_df[
            "logical_evaluations"
        ]
        == 35
    ).all():
        raise RuntimeError(
            "Logical evaluation "
            "count mismatch."
        )

    if not (
        selections_df[
            "logical_training_episodes"
        ]
        == 3200
    ).all():
        raise RuntimeError(
            "Hyperband resource "
            "accounting mismatch."
        )

    if not np.allclose(
        selections_df[
            "full_budget_equivalent_evaluations"
        ],
        16.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Full-budget-equivalent "
            "cost mismatch."
        )

    if (
        selections_df[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test leakage detected."
        )

    if (
        trace_df[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test leakage detected."
        )

    selection_path = (
        args.output_root
        / "search_selections.csv"
    )

    trace_path = (
        args.output_root
        / "search_trace.csv"
    )

    selections_df.to_csv(
        selection_path,
        index=False,
    )

    trace_df.to_csv(
        trace_path,
        index=False,
    )

    protocol = {
        "schema_version":
            1,

        "method":
            "hyperband",

        "algorithm":
            (
                "Exact Hyperband bracket "
                "construction over a finite "
                "64-anchor configuration "
                "portfolio."
            ),

        "eta":
            ETA,

        "minimum_resource":
            MIN_RESOURCE,

        "maximum_resource":
            MAX_RESOURCE,

        "resource_unit":
            "ProtoNet training episodes",

        "s_max":
            schedule[
                "s_max"
            ],

        "B":
            schedule[
                "B"
            ],

        "brackets":
            schedule[
                "brackets"
            ],

        "logical_evaluations_per_task_run":
            logical_evals_expected,

        "training_episodes_per_task_run":
            resource_expected,

        "full_budget_equivalent_evaluations":
            resource_expected
            / MAX_RESOURCE,

        "search_space":
            (
                "Frozen V2 64-anchor "
                "ProtoNet portfolio"
            ),

        "anchor_manifest":
            str(
                DEFAULT_ANCHORS
            ),

        "anchor_manifest_sha256":
            sha256_file(
                DEFAULT_ANCHORS
            ),

        "sampling":
            (
                "uniform without replacement "
                "inside each Hyperband bracket; "
                "independent bracket samples"
            ),

        "search_seeds":
            [
                int(seed)
                for seed
                in args.search_seeds
            ],

        "search_model_seed":
            DEFAULT_SEARCH_MODEL_SEED,

        "validation_episodes_per_evaluation":
            VALIDATION_EPISODES,

        "objective":
            (
                "maximize validation "
                "accuracy"
            ),

        "successive_halving_tie_break":
            (
                "validation accuracy "
                "descending, then config_id "
                "ascending"
            ),

        "final_selection":
            (
                "best validation accuracy "
                "among terminal 200-episode "
                "Hyperband evaluations only"
            ),

        "full_budget_source":
            (
                "frozen stage1 seed101 "
                "64-anchor evaluations"
            ),

        "partial_budget_source":
            (
                "genuine ProtoNet evaluations "
                "through BaselineAdapterV2"
            ),

        "partial_evaluation_cache":
            str(
                args.cache_root
            ),

        "final_model_seeds_reserved_for_later_test_evaluation":
            [
                int(seed)
                for seed
                in FINAL_MODEL_SEEDS
            ],

        "test_bank_used":
            False,

        "test_evaluation_performed":
            False,
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

    lock = {
        "schema_version":
            1,

        "method":
            "hyperband",

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

            "search_trace.csv":
                sha256_file(
                    trace_path
                ),
        },
    }

    lock_path = (
        args.output_root
        / "selection_lock_manifest.json"
    )

    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sha_path = (
        args.output_root
        / "selection_files.sha256"
    )

    files_to_hash = [
        protocol_path,
        selection_path,
        trace_path,
        lock_path,
    ]

    sha_path.write_text(
        "".join(
            (
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )
            for path
            in files_to_hash
        ),
        encoding="utf-8",
    )

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print(
        "=" * 90
    )

    print(
        "EXACT HYPERBAND "
        "SELECTION SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        "Tasks:",
        selections_df[
            "task_id"
        ].nunique(),
    )

    print(
        "Search seeds:",
        sorted(
            selections_df[
                "search_seed"
            ]
            .unique()
            .tolist()
        ),
    )

    print(
        "Selections:",
        len(
            selections_df
        ),
    )

    print(
        "Logical evaluation rows:",
        len(
            trace_df
        ),
    )

    print(
        "Logical evaluations "
        "per task/run:",
        logical_evals_expected,
    )

    print(
        "Training episodes "
        "per task/run:",
        resource_expected,
    )

    print(
        "Full-budget equivalents "
        "per task/run:",
        resource_expected
        / MAX_RESOURCE,
    )

    print(
        "Exact global-best "
        "config hit rate:",
        f"{100 * selections_df['exact_global_best_config_hit'].mean():.2f}%"
    )

    print(
        "Global-best validation-"
        "value hit rate:",
        f"{100 * selections_df['global_best_validation_value_hit'].mean():.2f}%"
    )

    print(
        "Mean validation regret:",
        f"{selections_df['validation_regret_to_stage1_best_pp'].mean():.4f} pp"
    )

    print(
        "Median validation regret:",
        f"{selections_df['validation_regret_to_stage1_best_pp'].median():.4f} pp"
    )

    print(
        "Mean logical measured "
        "serial search time:",
        f"{selections_df['logical_measured_serial_time_seconds'].mean():.2f} s"
    )

    print(
        "New partial-budget "
        "evaluations executed:",
        evaluator.new_evaluations,
    )

    print(
        "Partial cache hits:",
        evaluator.cache_hits,
    )

    print(
        "Stage1 full-budget hits:",
        evaluator.stage1_hits,
    )

    print(
        "Unique selected anchors:",
        selections_df[
            "selected_config_id"
        ].nunique(),
    )

    print(
        "Actual script wall time:",
        f"{elapsed:.2f} s",
    )

    print(
        "Test bank used: False"
    )

    print()
    print(
        "Selection lock:",
        lock_path,
    )

    print()
    print(
        "EXACT HYPERBAND "
        "SELECTION: PASS"
    )


if __name__ == "__main__":
    main()
