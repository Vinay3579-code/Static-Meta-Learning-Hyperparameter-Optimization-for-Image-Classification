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

from sklearn.neighbors import (
    KernelDensity,
)

from sml_hpo.baselines.adapter_v2 import (
    BaselineAdapterV2,
    DEFAULT_ANCHORS,
    DEFAULT_MANIFEST_ROOT,
    VALIDATION_EPISODES,
    DEFAULT_SEARCH_MODEL_SEED,
    FINAL_MODEL_SEEDS,
)

from run_bayesian_optimization_baseline_v2 import (
    AnchorFeatureEncoder,
)


DEFAULT_STAGE1_ROOT = Path(
    "results/oracles/final800_v2/test"
)

DEFAULT_CACHE_ROOT = Path(
    "results/hpo_baselines_v2/"
    "multifidelity_cache"
)

DEFAULT_OUTPUT_ROOT = Path(
    "results/hpo_baselines_v2/bohb"
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

# BOHB-style model parameters.
TOP_FRACTION = 0.15

MIN_POINTS_IN_MODEL = 6

KDE_BANDWIDTH = 0.5


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=DEFAULT_STAGE1_ROOT,
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
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


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
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


def stable_seed(
    *parts,
) -> int:
    payload = "|".join(
        str(x)
        for x in parts
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
        * ETA ** s_max
        != MAX_RESOURCE
    ):
        raise RuntimeError(
            "Resource range must form "
            "an exact eta power."
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

            keep = (
                None
                if i == s
                else max(
                    1,
                    int(
                        math.floor(
                            n_i
                            / ETA
                        )
                    ),
                )
            )

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


def cache_path(
    *,
    cache_root,
    task_id,
    config_id,
    budget,
    model_seed,
):
    return (
        cache_root
        / task_id
        / (
            f"{config_id}"
            f"__budget{budget:03d}"
            f"__seed{model_seed}.json"
        )
    )


class MultiFidelityEvaluator:
    def __init__(
        self,
        *,
        adapter,
        stage1_root,
        cache_root,
    ):
        self.adapter = adapter
        self.stage1_root = Path(
            stage1_root
        )
        self.cache_root = Path(
            cache_root
        )

        self._stage1 = {}

        self.new_evaluations = 0
        self.cache_hits = 0
        self.stage1_hits = 0

    def load_stage1(
        self,
        task_id,
    ):
        if task_id in self._stage1:
            return self._stage1[
                task_id
            ]

        task = self.adapter.get_task(
            task_id
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

        table[
            "config_id"
        ] = (
            table[
                "config_id"
            ].astype(str)
        )

        if len(table) != 64:
            raise RuntimeError(
                "Expected 64 stage1 "
                "results."
            )

        if set(
            table[
                "config_id"
            ]
        ) != set(
            self.adapter.anchor_ids()
        ):
            raise RuntimeError(
                "Stage1 portfolio mismatch."
            )

        self._stage1[
            task_id
        ] = table

        return table

    def evaluate(
        self,
        *,
        task_id,
        config_id,
        budget,
    ):
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
                f"Invalid budget: {budget}"
            )

        # Full-budget result already exists.
        if budget == 200:
            table = self.load_stage1(
                task_id
            )

            rows = table[
                table[
                    "config_id"
                ]
                == config_id
            ]

            if len(rows) != 1:
                raise RuntimeError(
                    "Unable to resolve "
                    "stage1 config."
                )

            row = rows.iloc[0]

            self.stage1_hits += 1

            return {
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
            }

        path = cache_path(
            cache_root=self.cache_root,

            task_id=task_id,

            config_id=config_id,

            budget=budget,

            model_seed=(
                DEFAULT_SEARCH_MODEL_SEED
            ),
        )

        if path.exists():
            value = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
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
                    "Cache task mismatch."
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
                    "Cache config mismatch."
                )

            if (
                int(
                    value[
                        "training_episodes"
                    ]
                )
                != budget
            ):
                raise RuntimeError(
                    "Cache budget mismatch."
                )

            if bool(
                value[
                    "test_bank_used"
                ]
            ):
                raise RuntimeError(
                    "Test leakage in cache."
                )

            self.cache_hits += 1

            return {
                "validation_accuracy":
                    float(
                        value[
                            "validation_accuracy"
                        ]
                    ),

                "elapsed_seconds":
                    float(
                        value[
                            "elapsed_seconds"
                        ]
                    ),

                "source":
                    "partial_budget_cache",

                "physically_executed_now":
                    False,
            }

        result = (
            self.adapter
            .train_validate_anchor(
                task_id=task_id,

                config_id=config_id,

                budget=budget,

                model_seed=(
                    DEFAULT_SEARCH_MODEL_SEED
                ),

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
                budget,

            "validation_episodes":
                VALIDATION_EPISODES,

            "model_seed":
                DEFAULT_SEARCH_MODEL_SEED,

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

            "test_bank_used":
                False,
        }

        atomic_write_json(
            path,
            value,
        )

        self.new_evaluations += 1

        return {
            "validation_accuracy":
                value[
                    "validation_accuracy"
                ],

            "elapsed_seconds":
                value[
                    "elapsed_seconds"
                ],

            "source":
                "new_partial_budget_evaluation",

            "physically_executed_now":
                True,
        }


class BOHBHistory:
    """
    Stores observations separately by
    fidelity.

    Model construction always prefers the
    highest budget with enough observations,
    matching BOHB's general principle of
    modeling the most informative available
    fidelity.
    """

    def __init__(
        self,
    ):
        self.rows = []

    def add(
        self,
        *,
        config_id,
        budget,
        validation_accuracy,
    ):
        self.rows.append(
            {
                "config_id":
                    str(
                        config_id
                    ),

                "budget":
                    int(
                        budget
                    ),

                "validation_accuracy":
                    float(
                        validation_accuracy
                    ),
            }
        )

    def dataframe(
        self,
    ):
        return pd.DataFrame(
            self.rows
        )

    def model_budget(
        self,
    ):
        if not self.rows:
            return None

        df = self.dataframe()

        for budget in (
            200,
            100,
            50,
            25,
        ):
            subset = df[
                df[
                    "budget"
                ] == budget
            ]

            unique_count = (
                subset[
                    "config_id"
                ]
                .nunique()
            )

            if (
                unique_count
                >= MIN_POINTS_IN_MODEL
            ):
                return budget

        return None

    def observations_at(
        self,
        budget,
    ):
        df = self.dataframe()

        subset = df[
            df[
                "budget"
            ] == int(
                budget
            )
        ].copy()

        # A config can appear in multiple
        # brackets at one fidelity.
        # It is deterministic under the
        # frozen seed, so collapse duplicates.
        subset = (
            subset.sort_values(
                [
                    "config_id",
                    "validation_accuracy",
                ]
            )
            .drop_duplicates(
                subset=[
                    "config_id"
                ],
                keep="last",
            )
        )

        return subset


def select_model_based_anchor(
    *,
    history,
    available_ids,
    encoder,
    rng,
):
    model_budget = (
        history.model_budget()
    )

    if (
        model_budget
        is None
    ):
        selected = str(
            rng.choice(
                np.asarray(
                    available_ids,
                    dtype=object,
                )
            )
        )

        return {
            "config_id":
                selected,

            "proposal_mode":
                "random_warmup",

            "model_budget":
                None,

            "density_ratio":
                np.nan,
        }

    observations = (
        history.observations_at(
            model_budget
        )
    )

    if (
        len(observations)
        < MIN_POINTS_IN_MODEL
    ):
        raise RuntimeError(
            "Insufficient BOHB "
            "model observations."
        )

    observations = (
        observations.sort_values(
            by=[
                "validation_accuracy",
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

    n_good = max(
        2,
        int(
            math.ceil(
                TOP_FRACTION
                * len(
                    observations
                )
            )
        ),
    )

    n_good = min(
        n_good,
        len(
            observations
        ) - 2,
    )

    if n_good < 2:
        selected = str(
            rng.choice(
                np.asarray(
                    available_ids,
                    dtype=object,
                )
            )
        )

        return {
            "config_id":
                selected,

            "proposal_mode":
                "random_warmup",

            "model_budget":
                model_budget,

            "density_ratio":
                np.nan,
        }

    good_ids = (
        observations.iloc[
            :n_good
        ][
            "config_id"
        ].tolist()
    )

    bad_ids = (
        observations.iloc[
            n_good:
        ][
            "config_id"
        ].tolist()
    )

    if len(
        bad_ids
    ) < 2:
        selected = str(
            rng.choice(
                np.asarray(
                    available_ids,
                    dtype=object,
                )
            )
        )

        return {
            "config_id":
                selected,

            "proposal_mode":
                "random_warmup",

            "model_budget":
                model_budget,

            "density_ratio":
                np.nan,
        }

    X_good = np.vstack(
        [
            encoder.encode(
                config_id
            )
            for config_id
            in good_ids
        ]
    )

    X_bad = np.vstack(
        [
            encoder.encode(
                config_id
            )
            for config_id
            in bad_ids
        ]
    )

    kde_good = KernelDensity(
        kernel="gaussian",
        bandwidth=(
            KDE_BANDWIDTH
        ),
    ).fit(
        X_good
    )

    kde_bad = KernelDensity(
        kernel="gaussian",
        bandwidth=(
            KDE_BANDWIDTH
        ),
    ).fit(
        X_bad
    )

    X_available = np.vstack(
        [
            encoder.encode(
                config_id
            )
            for config_id
            in available_ids
        ]
    )

    log_good = (
        kde_good.score_samples(
            X_available
        )
    )

    log_bad = (
        kde_bad.score_samples(
            X_available
        )
    )

    log_ratio = (
        log_good
        - log_bad
    )

    table = pd.DataFrame(
        {
            "config_id":
                available_ids,

            "log_density_ratio":
                log_ratio,
        }
    ).sort_values(
        by=[
            "log_density_ratio",
            "config_id",
        ],

        ascending=[
            False,
            True,
        ],

        kind="mergesort",
    )

    row = table.iloc[0]

    return {
        "config_id":
            str(
                row[
                    "config_id"
                ]
            ),

        "proposal_mode":
            "kde_density_ratio",

        "model_budget":
            int(
                model_budget
            ),

        "density_ratio":
            float(
                row[
                    "log_density_ratio"
                ]
            ),
    }


def propose_initial_population(
    *,
    n,
    history,
    anchor_ids,
    already_in_bracket,
    encoder,
    rng,
):
    chosen = []
    metadata = []

    for _ in range(
        n
    ):
        unavailable = (
            set(
                already_in_bracket
            )
            |
            set(
                chosen
            )
        )

        available = [
            config_id
            for config_id
            in anchor_ids
            if config_id
            not in unavailable
        ]

        if not available:
            raise RuntimeError(
                "BOHB exhausted anchor "
                "portfolio."
            )

        proposal = (
            select_model_based_anchor(
                history=history,

                available_ids=(
                    available
                ),

                encoder=encoder,

                rng=rng,
            )
        )

        chosen.append(
            proposal[
                "config_id"
            ]
        )

        metadata.append(
            proposal
        )

    return (
        chosen,
        metadata,
    )


def rank_rows(
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


def stage1_best(
    evaluator,
    task_id,
):
    table = (
        evaluator.load_stage1(
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


def run_bohb_once(
    *,
    task_id,
    search_seed,
    schedule,
    evaluator,
    anchor_ids,
    encoder,
):
    history = BOHBHistory()

    trace = []
    terminal_rows = []

    logical_eval_index = 0
    logical_training_episodes = 0
    logical_elapsed = 0.0

    new_start = (
        evaluator.new_evaluations
    )

    cache_start = (
        evaluator.cache_hits
    )

    stage1_start = (
        evaluator.stage1_hits
    )

    run_start = (
        time.perf_counter()
    )

    run_rng = (
        np.random.default_rng(
            stable_seed(
                "bohb",
                task_id,
                search_seed,
            )
        )
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

        current_configs = []
        proposal_metadata = []

        (
            current_configs,
            proposal_metadata,
        ) = propose_initial_population(
            n=(
                bracket[
                    "initial_n"
                ]
            ),

            history=history,

            anchor_ids=anchor_ids,

            already_in_bracket=[],

            encoder=encoder,

            rng=run_rng,
        )

        proposal_by_config = {
            item[
                "config_id"
            ]:
                item

            for item
            in proposal_metadata
        }

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
                    "BOHB population-size "
                    "mismatch."
                )

            scored = []

            for config_id in (
                current_configs
            ):
                logical_eval_index += 1

                result = (
                    evaluator.evaluate(
                        task_id=task_id,

                        config_id=config_id,

                        budget=budget,
                    )
                )

                accuracy = float(
                    result[
                        "validation_accuracy"
                    ]
                )

                history.add(
                    config_id=config_id,

                    budget=budget,

                    validation_accuracy=(
                        accuracy
                    ),
                )

                logical_training_episodes += (
                    budget
                )

                logical_elapsed += float(
                    result[
                        "elapsed_seconds"
                    ]
                )

                proposal = (
                    proposal_by_config.get(
                        config_id,
                        {
                            "proposal_mode":
                                "successive_halving_survivor",

                            "model_budget":
                                None,

                            "density_ratio":
                                np.nan,
                        },
                    )
                )

                row = {
                    "method":
                        "bohb_finite_portfolio",

                    "task_id":
                        task_id,

                    "search_seed":
                        int(
                            search_seed
                        ),

                    "bracket_s":
                        s,

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

                    "validation_accuracy":
                        accuracy,

                    "proposal_mode":
                        proposal[
                            "proposal_mode"
                        ],

                    "proposal_model_budget":
                        proposal[
                            "model_budget"
                        ],

                    "proposal_log_density_ratio":
                        proposal[
                            "density_ratio"
                        ],

                    "evaluation_source":
                        result[
                            "source"
                        ],

                    "physically_executed_now":
                        bool(
                            result[
                                "physically_executed_now"
                            ]
                        ),

                    "elapsed_seconds":
                        float(
                            result[
                                "elapsed_seconds"
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

            ranked = rank_rows(
                scored
            )

            if budget == 200:
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
                    row[
                        "config_id"
                    ]

                    for row
                    in ranked[
                        :keep
                    ]
                ]

                proposal_by_config = {
                    config_id: {
                        "config_id":
                            config_id,

                        "proposal_mode":
                            "successive_halving_survivor",

                        "model_budget":
                            None,

                        "density_ratio":
                            np.nan,
                    }

                    for config_id
                    in current_configs
                }

    if not terminal_rows:
        raise RuntimeError(
            "BOHB produced no "
            "full-budget terminal runs."
        )

    terminal_rows = (
        rank_rows(
            terminal_rows
        )
    )

    selected = (
        terminal_rows[0]
    )

    return {
        "selected_config_id":
            selected[
                "config_id"
            ],

        "selected_validation_accuracy":
            float(
                selected[
                    "validation_accuracy"
                ]
            ),

        "logical_evaluations":
            logical_eval_index,

        "logical_training_episodes":
            logical_training_episodes,

        "full_budget_equivalent_evaluations":
            float(
                logical_training_episodes
                / 200
            ),

        "logical_measured_serial_time_seconds":
            float(
                logical_elapsed
            ),

        "terminal_full_budget_count":
            len(
                terminal_rows
            ),

        "terminal_unique_config_count":
            len(
                {
                    row[
                        "config_id"
                    ]
                    for row
                    in terminal_rows
                }
            ),

        "physical_new_partial_evaluations":
            (
                evaluator.new_evaluations
                - new_start
            ),

        "cache_hits":
            (
                evaluator.cache_hits
                - cache_start
            ),

        "stage1_hits":
            (
                evaluator.stage1_hits
                - stage1_start
            ),

        "actual_run_wall_time_seconds":
            float(
                time.perf_counter()
                - run_start
            ),

        "trace":
            trace,
    }


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

    adapter = (
        BaselineAdapterV2(
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
    )

    anchor_ids = (
        adapter.anchor_ids()
    )

    encoder = (
        AnchorFeatureEncoder(
            adapter
        )
    )

    evaluator = (
        MultiFidelityEvaluator(
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
        task = adapter.get_task(
            args.task_id
        )

        if task.split != "test":
            raise ValueError(
                "--task-id must be "
                "meta-test."
            )

        task_ids = [
            args.task_id
        ]

    selections = []
    traces = []

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
            reference_id,
            reference_value,
        ) = stage1_best(
            evaluator,
            task_id,
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
                run_bohb_once(
                    task_id=task_id,

                    search_seed=int(
                        search_seed
                    ),

                    schedule=schedule,

                    evaluator=evaluator,

                    anchor_ids=(
                        anchor_ids
                    ),

                    encoder=encoder,
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
                    reference_value,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

            selections.append(
                {
                    "method":
                        "bohb_finite_portfolio",

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
                        reference_id,

                    "stage1_global_best_validation_accuracy":
                        reference_value,

                    "validation_regret_to_stage1_best_pp":
                        float(
                            100
                            * (
                                reference_value
                                - selected_value
                            )
                        ),

                    "exact_global_best_config_hit":
                        bool(
                            selected_id
                            == reference_id
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

                    "test_bank_used":
                        False,
                }
            )

            traces.extend(
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
                f"{100*(reference_value-selected_value):.4f}"
                "pp "
                "new="
                f"{result['physical_new_partial_evaluations']} "
                "cache="
                f"{result['cache_hits']} "
                "stage1="
                f"{result['stage1_hits']}"
            )

        pd.DataFrame(
            selections
        ).to_csv(
            args.output_root
            / "search_selections.partial.csv",
            index=False,
        )

        pd.DataFrame(
            traces
        ).to_csv(
            args.output_root
            / "search_trace.partial.csv",
            index=False,
        )

    sel = pd.DataFrame(
        selections
    )

    trace = pd.DataFrame(
        traces
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
        * 35
    )

    if len(sel) != expected_selections:
        raise RuntimeError(
            "Unexpected BOHB selection "
            "count."
        )

    if len(trace) != expected_trace:
        raise RuntimeError(
            "Unexpected BOHB trace "
            "count."
        )

    if not (
        sel[
            "logical_training_episodes"
        ] == 3200
    ).all():
        raise RuntimeError(
            "BOHB resource accounting "
            "mismatch."
        )

    if not np.allclose(
        sel[
            "full_budget_equivalent_evaluations"
        ],
        16.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "BOHB FBE mismatch."
        )

    if (
        sel[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test leakage."
        )

    if (
        trace[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test leakage."
        )

    selection_path = (
        args.output_root
        / "search_selections.csv"
    )

    trace_path = (
        args.output_root
        / "search_trace.csv"
    )

    sel.to_csv(
        selection_path,
        index=False,
    )

    trace.to_csv(
        trace_path,
        index=False,
    )

    protocol = {
        "schema_version":
            1,

        "method":
            "bohb_finite_portfolio",

        "description":
            (
                "Finite-portfolio adaptation "
                "of BOHB: exact Hyperband "
                "resource allocation plus "
                "KDE density-ratio "
                "configuration proposals."
            ),

        "search_space":
            (
                "Frozen V2 64-anchor "
                "ProtoNet portfolio"
            ),

        "minimum_resource":
            MIN_RESOURCE,

        "maximum_resource":
            MAX_RESOURCE,

        "eta":
            ETA,

        "resource_unit":
            "ProtoNet training episodes",

        "brackets":
            schedule[
                "brackets"
            ],

        "logical_evaluations_per_task_run":
            35,

        "training_episodes_per_task_run":
            3200,

        "full_budget_equivalent_evaluations":
            16.0,

        "model":
            "KDE density ratio",

        "top_fraction":
            TOP_FRACTION,

        "minimum_points_in_model":
            MIN_POINTS_IN_MODEL,

        "kde_bandwidth":
            KDE_BANDWIDTH,

        "model_fidelity_rule":
            (
                "highest resource level "
                "with sufficient unique "
                "observations"
            ),

        "anchor_encoding_feature_names":
            encoder.feature_names,

        "search_seeds":
            [
                int(x)
                for x
                in args.search_seeds
            ],

        "search_model_seed":
            DEFAULT_SEARCH_MODEL_SEED,

        "validation_episodes":
            VALIDATION_EPISODES,

        "full_budget_source":
            (
                "frozen stage1 seed101 "
                "64-anchor evaluations"
            ),

        "partial_budget_source":
            (
                "genuine ProtoNet "
                "evaluations through "
                "BaselineAdapterV2"
            ),

        "shared_multifidelity_cache":
            str(
                args.cache_root
            ),

        "anchor_manifest":
            str(
                DEFAULT_ANCHORS
            ),

        "anchor_manifest_sha256":
            sha256_file(
                DEFAULT_ANCHORS
            ),

        "final_selection":
            (
                "best validation score "
                "among terminal 200-episode "
                "evaluations"
            ),

        "final_model_seeds_reserved_for_later_test_evaluation":
            [
                int(x)
                for x
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
            "bohb_finite_portfolio",

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

    paths = [
        protocol_path,
        selection_path,
        trace_path,
        lock_path,
    ]

    sha_path.write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.name}\n"
            for path
            in paths
        ),
        encoding="utf-8",
    )

    elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 90)
    print(
        "BOHB FINITE-PORTFOLIO "
        "SELECTION SUMMARY"
    )
    print("=" * 90)

    print(
        "Tasks:",
        sel[
            "task_id"
        ].nunique(),
    )

    print(
        "Search seeds:",
        sorted(
            sel[
                "search_seed"
            ].unique()
            .tolist()
        ),
    )

    print(
        "Selections:",
        len(sel),
    )

    print(
        "Logical evaluation rows:",
        len(trace),
    )

    print(
        "Logical evaluations/run:",
        sel[
            "logical_evaluations"
        ].mean(),
    )

    print(
        "Training episodes/run:",
        sel[
            "logical_training_episodes"
        ].mean(),
    )

    print(
        "Full-budget equivalents/run:",
        sel[
            "full_budget_equivalent_evaluations"
        ].mean(),
    )

    print(
        "Exact global-best config hit:",
        f"{100 * sel['exact_global_best_config_hit'].mean():.2f}%"
    )

    print(
        "Best-value hit:",
        f"{100 * sel['global_best_validation_value_hit'].mean():.2f}%"
    )

    print(
        "Mean validation regret:",
        f"{sel['validation_regret_to_stage1_best_pp'].mean():.4f} pp"
    )

    print(
        "Median validation regret:",
        f"{sel['validation_regret_to_stage1_best_pp'].median():.4f} pp"
    )

    print(
        "New partial evaluations:",
        evaluator.new_evaluations,
    )

    print(
        "Shared partial-cache hits:",
        evaluator.cache_hits,
    )

    print(
        "Stage1 hits:",
        evaluator.stage1_hits,
    )

    print(
        "Model-based proposals:",
        int(
            (
                trace[
                    "proposal_mode"
                ]
                == "kde_density_ratio"
            ).sum()
        ),
    )

    print(
        "Random warmup proposals:",
        int(
            (
                trace[
                    "proposal_mode"
                ]
                == "random_warmup"
            ).sum()
        ),
    )

    print(
        "Unique selected anchors:",
        sel[
            "selected_config_id"
        ].nunique(),
    )

    print(
        "Script wall time:",
        f"{elapsed:.2f} s",
    )

    print(
        "Test bank used: False"
    )

    print()
    print(
        "BOHB FINITE-PORTFOLIO "
        "SELECTION: PASS"
    )


if __name__ == "__main__":
    main()
