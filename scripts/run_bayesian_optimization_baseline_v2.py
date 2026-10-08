from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.stats import norm

from sklearn.exceptions import (
    ConvergenceWarning,
)
from sklearn.gaussian_process import (
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

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
    "results/hpo_baselines_v2/"
    "bayesian_optimization"
)

DEFAULT_SEARCH_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

# Recommended Bayesian-optimization budget.
DEFAULT_N_CALLS = 20

DEFAULT_N_INITIAL = 5

# Expected Improvement exploration parameter.
EI_XI = 0.001


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
        "--task-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n-calls",
        type=int,
        default=DEFAULT_N_CALLS,
    )

    parser.add_argument(
        "--n-initial",
        type=int,
        default=DEFAULT_N_INITIAL,
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
    payload = (
        "discrete_gp_ei_bo_v2|"
        f"{search_seed}|"
        f"{task_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    # sklearn expects a 32-bit-compatible
    # random state.
    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def safe_log10(
    value: float,
) -> float:
    value = float(value)

    return float(
        np.log10(
            max(
                value,
                1e-12,
            )
        )
    )


class AnchorFeatureEncoder:
    """
    Fixed feature representation of the
    frozen 64-anchor search space.

    No task performance information is used
    when constructing these features.
    """

    def __init__(
        self,
        adapter: BaselineAdapterV2,
    ):
        self.adapter = adapter

        self.anchor_ids = (
            adapter.anchor_ids()
        )

        configs = [
            adapter.get_config(
                config_id
            )
            for config_id
            in self.anchor_ids
        ]

        self.optimizers = sorted(
            {
                str(
                    c.optimizer
                )
                for c in configs
            }
        )

        self.schedulers = sorted(
            {
                str(
                    c.scheduler
                )
                for c in configs
            }
        )

        self.distance_metrics = sorted(
            {
                str(
                    c.distance_metric
                )
                for c in configs
            }
        )

        numeric = np.asarray(
            [
                self._numeric_raw(c)
                for c in configs
            ],
            dtype=np.float64,
        )

        self.numeric_mean = (
            numeric.mean(
                axis=0
            )
        )

        self.numeric_std = (
            numeric.std(
                axis=0,
                ddof=0,
            )
        )

        self.numeric_std[
            self.numeric_std
            < 1e-12
        ] = 1.0

        self.feature_names = [
            "log10_learning_rate",
            "log10_weight_decay",
            "log2_hidden_channels",
            "log2_embedding_dim",
            "dropout",
            "log10_temperature",
            "label_smoothing",
        ]

        self.feature_names += [
            f"optimizer={value}"
            for value
            in self.optimizers
        ]

        self.feature_names += [
            f"scheduler={value}"
            for value
            in self.schedulers
        ]

        self.feature_names += [
            f"distance_metric={value}"
            for value
            in self.distance_metrics
        ]

        self.matrix = np.vstack(
            [
                self.encode(
                    config_id
                )
                for config_id
                in self.anchor_ids
            ]
        )

        if not np.isfinite(
            self.matrix
        ).all():
            raise RuntimeError(
                "Non-finite values in "
                "anchor feature matrix."
            )

        if (
            self.matrix.shape[0]
            != 64
        ):
            raise RuntimeError(
                "Expected 64 anchor "
                "feature vectors."
            )

        self.row_by_id = {
            config_id: index
            for index, config_id
            in enumerate(
                self.anchor_ids
            )
        }

    @staticmethod
    def _numeric_raw(
        config,
    ):
        return np.asarray(
            [
                safe_log10(
                    config.learning_rate
                ),

                safe_log10(
                    config.weight_decay
                ),

                np.log2(
                    float(
                        config.hidden_channels
                    )
                ),

                np.log2(
                    float(
                        config.embedding_dim
                    )
                ),

                float(
                    config.dropout
                ),

                safe_log10(
                    config.temperature
                ),

                float(
                    config.label_smoothing
                ),
            ],
            dtype=np.float64,
        )

    def encode(
        self,
        config_id: str,
    ) -> np.ndarray:
        config = (
            self.adapter.get_config(
                config_id
            )
        )

        numeric = (
            self._numeric_raw(
                config
            )
            - self.numeric_mean
        ) / self.numeric_std

        optimizer = np.asarray(
            [
                float(
                    str(
                        config.optimizer
                    )
                    == value
                )
                for value
                in self.optimizers
            ],
            dtype=np.float64,
        )

        scheduler = np.asarray(
            [
                float(
                    str(
                        config.scheduler
                    )
                    == value
                )
                for value
                in self.schedulers
            ],
            dtype=np.float64,
        )

        metric = np.asarray(
            [
                float(
                    str(
                        config.distance_metric
                    )
                    == value
                )
                for value
                in self.distance_metrics
            ],
            dtype=np.float64,
        )

        return np.concatenate(
            [
                numeric,
                optimizer,
                scheduler,
                metric,
            ]
        )


def validate_stage1_table(
    *,
    table: pd.DataFrame,
    task_id: str,
    anchor_ids: list[str],
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
            f"columns {sorted(missing)}"
        )

    if len(table) != 64:
        raise RuntimeError(
            f"{task_id}: expected "
            "64 anchor rows."
        )

    ids = (
        table[
            "config_id"
        ]
        .astype(str)
        .tolist()
    )

    if len(ids) != len(
        set(ids)
    ):
        raise RuntimeError(
            f"{task_id}: duplicate "
            "config IDs."
        )

    if set(ids) != set(
        anchor_ids
    ):
        raise RuntimeError(
            f"{task_id}: stage1 "
            "anchor portfolio mismatch."
        )

    accuracy = pd.to_numeric(
        table[
            "validation_accuracy_mean"
        ],
        errors="raise",
    )

    if (
        not np.isfinite(
            accuracy
        ).all()
    ):
        raise RuntimeError(
            f"{task_id}: invalid "
            "validation accuracy."
        )

    if (
        (accuracy < 0).any()
        or
        (accuracy > 1).any()
    ):
        raise RuntimeError(
            f"{task_id}: validation "
            "accuracy outside [0,1]."
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
            FULL_TRAIN_EPISODES
        }:
            raise RuntimeError(
                f"{task_id}: stage1 "
                "training budget mismatch."
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
                f"{task_id}: validation "
                "budget mismatch."
            )


def expected_improvement(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    xi: float,
):
    mean = np.asarray(
        mean,
        dtype=np.float64,
    )

    std = np.asarray(
        std,
        dtype=np.float64,
    )

    improvement = (
        mean
        - float(
            best_observed
        )
        - float(
            xi
        )
    )

    ei = np.zeros_like(
        mean
    )

    valid = (
        std > 1e-12
    )

    z = np.zeros_like(
        mean
    )

    z[valid] = (
        improvement[valid]
        / std[valid]
    )

    ei[valid] = (
        improvement[valid]
        * norm.cdf(
            z[valid]
        )
        +
        std[valid]
        * norm.pdf(
            z[valid]
        )
    )

    ei[
        ~np.isfinite(ei)
    ] = 0.0

    return ei


def build_gp(
    random_state: int,
):
    # Inputs are standardized numeric
    # features plus 0/1 categorical
    # one-hot encodings.
    #
    # Isotropic Matern avoids fitting
    # dozens of independent length scales
    # from only 5-20 observations.
    kernel = (
        ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(
                1e-2,
                1e2,
            ),
        )
        *
        Matern(
            length_scale=1.0,
            length_scale_bounds=(
                1e-2,
                1e2,
            ),
            nu=2.5,
        )
        +
        WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(
                1e-8,
                1e-2,
            ),
        )
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=int(
            random_state
        ),
    )


def incumbent(
    observed_ids,
    observed_values,
):
    table = pd.DataFrame(
        {
            "config_id":
                observed_ids,

            "value":
                observed_values,
        }
    )

    table = table.sort_values(
        by=[
            "value",
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
                "value"
            ]
        ),
    )


def run_discrete_gp_ei(
    *,
    task_id: str,
    search_seed: int,
    n_calls: int,
    n_initial: int,
    anchor_ids: list[str],
    encoder: AnchorFeatureEncoder,
    value_by_id: dict[str, float],
    elapsed_by_id: dict[str, float],
):
    derived_seed = (
        stable_rng_seed(
            task_id=task_id,
            search_seed=(
                search_seed
            ),
        )
    )

    rng = np.random.default_rng(
        derived_seed
    )

    initial_ids = (
        rng.choice(
            np.asarray(
                anchor_ids,
                dtype=object,
            ),
            size=n_initial,
            replace=False,
        )
        .tolist()
    )

    observed_ids = []
    observed_values = []

    trace_rows = []

    algorithm_start = (
        time.perf_counter()
    )

    # ---------------------------------
    # Initial random design.
    # ---------------------------------

    for config_id in (
        initial_ids
    ):
        value = float(
            value_by_id[
                config_id
            ]
        )

        observed_ids.append(
            config_id
        )

        observed_values.append(
            value
        )

        (
            current_best_id,
            current_best_value,
        ) = incumbent(
            observed_ids,
            observed_values,
        )

        trace_rows.append(
            {
                "step":
                    len(
                        observed_ids
                    ),

                "phase":
                    "initial_random",

                "config_id":
                    config_id,

                "validation_accuracy":
                    value,

                "predicted_mean_before_query":
                    np.nan,

                "predicted_std_before_query":
                    np.nan,

                "expected_improvement":
                    np.nan,

                "incumbent_config_id":
                    current_best_id,

                "incumbent_validation_accuracy":
                    current_best_value,

                "elapsed_seconds":
                    float(
                        elapsed_by_id[
                            config_id
                        ]
                    ),
            }
        )

    # ---------------------------------
    # Sequential GP-EI queries.
    # ---------------------------------

    while (
        len(
            observed_ids
        )
        < n_calls
    ):
        observed_set = set(
            observed_ids
        )

        remaining_ids = [
            config_id
            for config_id
            in anchor_ids
            if config_id
            not in observed_set
        ]

        if not remaining_ids:
            raise RuntimeError(
                "No unevaluated anchors "
                "remain."
            )

        X_train = np.vstack(
            [
                encoder.encode(
                    config_id
                )
                for config_id
                in observed_ids
            ]
        )

        y_train = np.asarray(
            observed_values,
            dtype=np.float64,
        )

        X_remaining = np.vstack(
            [
                encoder.encode(
                    config_id
                )
                for config_id
                in remaining_ids
            ]
        )

        gp = build_gp(
            random_state=(
                derived_seed
                + len(
                    observed_ids
                )
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=(
                    ConvergenceWarning
                ),
            )

            gp.fit(
                X_train,
                y_train,
            )

        (
            predicted_mean,
            predicted_std,
        ) = gp.predict(
            X_remaining,
            return_std=True,
        )

        best_observed = float(
            np.max(
                y_train
            )
        )

        ei = expected_improvement(
            mean=predicted_mean,
            std=predicted_std,
            best_observed=(
                best_observed
            ),
            xi=EI_XI,
        )

        acquisition = (
            pd.DataFrame(
                {
                    "config_id":
                        remaining_ids,

                    "predicted_mean":
                        predicted_mean,

                    "predicted_std":
                        predicted_std,

                    "ei":
                        ei,
                }
            )
            .sort_values(
                by=[
                    "ei",
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
            acquisition.iloc[0]
        )

        config_id = str(
            selected[
                "config_id"
            ]
        )

        value = float(
            value_by_id[
                config_id
            ]
        )

        observed_ids.append(
            config_id
        )

        observed_values.append(
            value
        )

        (
            current_best_id,
            current_best_value,
        ) = incumbent(
            observed_ids,
            observed_values,
        )

        trace_rows.append(
            {
                "step":
                    len(
                        observed_ids
                    ),

                "phase":
                    "gp_expected_improvement",

                "config_id":
                    config_id,

                "validation_accuracy":
                    value,

                "predicted_mean_before_query":
                    float(
                        selected[
                            "predicted_mean"
                        ]
                    ),

                "predicted_std_before_query":
                    float(
                        selected[
                            "predicted_std"
                        ]
                    ),

                "expected_improvement":
                    float(
                        selected[
                            "ei"
                        ]
                    ),

                "incumbent_config_id":
                    current_best_id,

                "incumbent_validation_accuracy":
                    current_best_value,

                "elapsed_seconds":
                    float(
                        elapsed_by_id[
                            config_id
                        ]
                    ),
            }
        )

    (
        best_config_id,
        best_validation_accuracy,
    ) = incumbent(
        observed_ids,
        observed_values,
    )

    algorithm_overhead = (
        time.perf_counter()
        - algorithm_start
    )

    return {
        "derived_rng_seed":
            int(
                derived_seed
            ),

        "initial_config_ids":
            list(
                initial_ids
            ),

        "evaluated_config_ids":
            list(
                observed_ids
            ),

        "selected_config_id":
            best_config_id,

        "selected_validation_accuracy":
            float(
                best_validation_accuracy
            ),

        "algorithm_overhead_seconds":
            float(
                algorithm_overhead
            ),

        "trace_rows":
            trace_rows,
    }


def main():
    args = parse_args()

    if not (
        2 <= args.n_initial
        <= args.n_calls
        <= 64
    ):
        raise ValueError(
            "Require "
            "2 <= n_initial "
            "<= n_calls <= 64."
        )

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

    # No GPU is needed because all
    # objective values were genuinely
    # measured previously.
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
            "Expected 64 anchors."
        )

    encoder = (
        AnchorFeatureEncoder(
            adapter
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
                "BO task must belong "
                "to held-out meta-test "
                "split."
            )

        task_ids = [
            args.task_id
        ]

    selection_rows = []
    trace_rows = []

    run_start = (
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
            anchor_ids=(
                anchor_ids
            ),
        )

        stage1 = (
            stage1.copy()
            .assign(
                config_id=lambda df:
                    df[
                        "config_id"
                    ].astype(str)
            )
        )

        value_by_id = dict(
            zip(
                stage1[
                    "config_id"
                ],
                stage1[
                    "validation_accuracy_mean"
                ].astype(float),
            )
        )

        elapsed_by_id = dict(
            zip(
                stage1[
                    "config_id"
                ],
                stage1[
                    "total_elapsed_seconds"
                ].astype(float),
            )
        )

        global_ranked = (
            stage1.sort_values(
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

        global_ranked[
            "global_rank"
        ] = np.arange(
            1,
            len(
                global_ranked
            )
            + 1,
        )

        global_best = (
            global_ranked.iloc[0]
        )

        global_best_id = str(
            global_best[
                "config_id"
            ]
        )

        global_best_value = float(
            global_best[
                "validation_accuracy_mean"
            ]
        )

        global_rank_map = dict(
            zip(
                global_ranked[
                    "config_id"
                ],
                global_ranked[
                    "global_rank"
                ].astype(int),
            )
        )

        stage1_hash = (
            sha256_file(
                stage1_path
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
            result = (
                run_discrete_gp_ei(
                    task_id=task_id,

                    search_seed=int(
                        search_seed
                    ),

                    n_calls=(
                        args.n_calls
                    ),

                    n_initial=(
                        args.n_initial
                    ),

                    anchor_ids=(
                        anchor_ids
                    ),

                    encoder=encoder,

                    value_by_id=(
                        value_by_id
                    ),

                    elapsed_by_id=(
                        elapsed_by_id
                    ),
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

            evaluated_ids = (
                result[
                    "evaluated_config_ids"
                ]
            )

            measured_time = float(
                sum(
                    elapsed_by_id[
                        config_id
                    ]
                    for config_id
                    in evaluated_ids
                )
            )

            best_value_hit = bool(
                np.isclose(
                    selected_value,
                    global_best_value,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

            initial_json = (
                json.dumps(
                    result[
                        "initial_config_ids"
                    ],
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            evaluated_json = (
                json.dumps(
                    evaluated_ids,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            selection_rows.append(
                {
                    "method":
                        "bayesian_optimization_gp_ei",

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
                            result[
                                "derived_rng_seed"
                            ]
                        ),

                    "n_calls":
                        int(
                            args.n_calls
                        ),

                    "n_initial":
                        int(
                            args.n_initial
                        ),

                    "selected_config_id":
                        selected_id,

                    "selected_validation_accuracy":
                        selected_value,

                    "stage1_global_best_config_id":
                        global_best_id,

                    "stage1_global_best_validation_accuracy":
                        global_best_value,

                    "selected_global_rank":
                        int(
                            global_rank_map[
                                selected_id
                            ]
                        ),

                    "validation_regret_to_stage1_best_pp":
                        float(
                            100.0
                            * (
                                global_best_value
                                - selected_value
                            )
                        ),

                    "exact_global_best_config_hit":
                        bool(
                            selected_id
                            == global_best_id
                        ),

                    "global_best_validation_value_hit":
                        best_value_hit,

                    "configuration_evaluations":
                        int(
                            args.n_calls
                        ),

                    "full_budget_equivalent_evaluations":
                        float(
                            args.n_calls
                        ),

                    "training_episodes_consumed":
                        int(
                            args.n_calls
                            * FULL_TRAIN_EPISODES
                        ),

                    "measured_serial_search_time_seconds":
                        measured_time,

                    "bo_algorithm_overhead_seconds":
                        float(
                            result[
                                "algorithm_overhead_seconds"
                            ]
                        ),

                    "initial_config_ids_json":
                        initial_json,

                    "evaluated_config_ids_json":
                        evaluated_json,

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

            for trace in (
                result[
                    "trace_rows"
                ]
            ):
                trace_rows.append(
                    {
                        "method":
                            "bayesian_optimization_gp_ei",

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
                                result[
                                    "derived_rng_seed"
                                ]
                            ),

                        **trace,

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

    trace = pd.DataFrame(
        trace_rows
    )

    selection_path = (
        args.output_root
        / "search_selections.csv"
    )

    trace_path = (
        args.output_root
        / "search_trace.csv"
    )

    selections.to_csv(
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
            "bayesian_optimization_gp_ei",

        "description":
            (
                "Discrete Gaussian-process "
                "Bayesian Optimization with "
                "Expected Improvement over "
                "the frozen 64-anchor "
                "ProtoNet portfolio."
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

        "n_calls":
            int(
                args.n_calls
            ),

        "n_initial_random":
            int(
                args.n_initial
            ),

        "acquisition":
            "expected_improvement",

        "ei_xi":
            float(
                EI_XI
            ),

        "surrogate":
            (
                "GaussianProcessRegressor "
                "with Constant*Matern(nu=2.5)"
                "+WhiteKernel"
            ),

        "anchor_encoding_feature_names":
            encoder.feature_names,

        "numeric_feature_standardization":
            (
                "mean/std computed only "
                "from the fixed 64-anchor "
                "configuration portfolio"
            ),

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

        "objective":
            "maximize validation accuracy",

        "search_model_seed":
            int(
                DEFAULT_SEARCH_MODEL_SEED
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

        "final_model_seeds_reserved_for_later_test_evaluation":
            [
                int(seed)
                for seed
                in FINAL_MODEL_SEEDS
            ],

        "validation_source":
            (
                "real frozen "
                "stage1_seed101_full64.csv "
                "evaluations"
            ),

        "tie_break":
            (
                "highest validation value, "
                "then config_id ascending"
            ),

        "test_bank_used":
            False,

        "test_evaluation_performed":
            False,

        "numpy_version":
            np.__version__,

        "scipy_version":
            scipy.__version__,

        "sklearn_version":
            sklearn.__version__,
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

    expected_trace_rows = (
        expected_selection_rows
        * args.n_calls
    )

    if len(
        selections
    ) != expected_selection_rows:
        raise RuntimeError(
            "Unexpected selection "
            "row count."
        )

    if len(
        trace
    ) != expected_trace_rows:
        raise RuntimeError(
            "Unexpected BO trace "
            "row count."
        )

    unique_counts = (
        trace.groupby(
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
        unique_counts
        == args.n_calls
    ).all():
        raise RuntimeError(
            "Duplicate BO query "
            "detected within a run."
        )

    if (
        selections[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test-bank leakage "
            "detected."
        )

    if (
        trace[
            "test_bank_used"
        ]
        .astype(bool)
        .any()
    ):
        raise RuntimeError(
            "Test-bank leakage "
            "detected."
        )

    # ----------------------------------------------------
    # Freeze selections BEFORE any final test evaluation.
    # ----------------------------------------------------

    lock = {
        "schema_version":
            1,

        "method":
            "bayesian_optimization_gp_ei",

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

    total_elapsed = (
        time.perf_counter()
        - run_start
    )

    print()
    print("=" * 90)
    print(
        "BAYESIAN OPTIMIZATION "
        "SELECTION SUMMARY"
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
            ]
            .unique()
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
        "BO queries:",
        len(
            trace
        ),
    )

    print(
        "Queries per task/run:",
        args.n_calls,
    )

    print(
        "Initial random points:",
        args.n_initial,
    )

    print(
        "Exact global-best "
        "config hit rate:",
        f"{100 * selections['exact_global_best_config_hit'].mean():.2f}%"
    )

    print(
        "Global-best validation-"
        "value hit rate:",
        f"{100 * selections['global_best_validation_value_hit'].mean():.2f}%"
    )

    print(
        "Mean validation regret:",
        f"{selections['validation_regret_to_stage1_best_pp'].mean():.4f} pp"
    )

    print(
        "Median validation regret:",
        f"{selections['validation_regret_to_stage1_best_pp'].median():.4f} pp"
    )

    print(
        "Mean measured serial "
        "search time:",
        f"{selections['measured_serial_search_time_seconds'].mean():.2f} s"
    )

    print(
        "Mean BO algorithm overhead:",
        f"{selections['bo_algorithm_overhead_seconds'].mean():.4f} s"
    )

    print(
        "Unique selected anchors:",
        selections[
            "selected_config_id"
        ].nunique(),
    )

    print(
        "Replay elapsed:",
        f"{total_elapsed:.2f} s",
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
        "BAYESIAN OPTIMIZATION "
        "SELECTION: PASS"
    )


if __name__ == "__main__":
    main()
