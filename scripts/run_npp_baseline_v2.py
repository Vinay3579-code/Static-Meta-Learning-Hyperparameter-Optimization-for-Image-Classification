from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pandas.api.types import (
    is_numeric_dtype,
)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sml_hpo.baselines.adapter_v2 import (
    BaselineAdapterV2,
    DEFAULT_ANCHORS,
    DEFAULT_MANIFEST_ROOT,
)

from run_bayesian_optimization_baseline_v2 import (
    AnchorFeatureEncoder,
)


DESCRIPTOR_ROOT = Path(
    "results/descriptors/final800/merged"
)

ORACLE_ROOT = Path(
    "results/oracles/final800_v2"
)

OUTPUT_ROOT = Path(
    "results/hpo_baselines_v2/npp"
)

MODEL_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

EXPECTED_TASKS = {
    "train": 560,
    "validation": 120,
    "test": 120,
}

EXPECTED_DESCRIPTOR_DIM = 84

PROTOCOL_COLUMNS = [
    "n_way",
    "n_shot",
    "n_query",
]

MAX_EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DROPOUT = 0.1


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_descriptor_file(
    split: str,
) -> Path:

    preferred = (
        DESCRIPTOR_ROOT
        / f"{split}_full84.csv"
    )

    if preferred.exists():
        return preferred

    candidates = sorted(
        DESCRIPTOR_ROOT.glob(
            f"{split}*.csv"
        )
    )

    valid = []

    for path in candidates:
        try:
            df = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        if "task_id" in df.columns:
            valid.append(path)

    if len(valid) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve "
            f"{split} descriptor CSV.\n"
            f"Candidates: {valid}"
        )

    return valid[0]


# ----------------------------------------------------------------------
# Descriptor schema
# ----------------------------------------------------------------------

NON_DESCRIPTOR_COLUMNS = {
    "task_id",
    "dataset",
    "split",
    "regime",

    "n_way",
    "n_shot",
    "n_query",

    "task_seed",

    "schema_version",
    "descriptor_schema_version",
    "descriptor_version",

    "descriptor_dim",
    "descriptor_dimension",
    "static_dim",
    "probe_dim",
    "static_descriptor_dim",
    "probe_descriptor_dim",

    "feature_count",
    "descriptor_count",

    "support_size",
    "query_size",
    "class_pool_size",
    "class_count",

    "probe_seed",
    "probe_episodes",

    "index",
    "row_index",
}


def infer_descriptor_columns(
    df: pd.DataFrame,
):
    """
    Resolve the exact frozen 84-D task descriptor.

    The final descriptor schema is:

        14 pixel-level statistics
        22 texture / edge features
        18 prototype-geometry features
        10 spectral / PCA features
        20 probe optimization signals

    Total:
        14 + 22 + 18 + 10 + 20 = 84

    Numeric extraction/probe metadata such as
    descriptor_episodes, probe_steps, and elapsed
    time are intentionally excluded.
    """

    expected_family_counts = {
        "pixel_": 14,
        "texture_": 22,
        "geometry_": 18,
        "spectral_": 10,
    }

    descriptor_columns = []

    # --------------------------------------------------
    # Static descriptor families: exactly 64 dimensions.
    # --------------------------------------------------

    for prefix, expected_count in (
        expected_family_counts.items()
    ):
        columns = [
            column
            for column in df.columns
            if str(column).startswith(
                prefix
            )
        ]

        if len(columns) != expected_count:
            raise RuntimeError(
                f"Descriptor family "
                f"{prefix!r}: expected "
                f"{expected_count} columns, "
                f"observed {len(columns)}.\n"
                f"Columns: {columns}"
            )

        for column in columns:
            if not is_numeric_dtype(
                df[column]
            ):
                raise RuntimeError(
                    f"Descriptor column "
                    f"{column!r} is not numeric."
                )

        descriptor_columns.extend(
            columns
        )

    # --------------------------------------------------
    # Probe descriptor: exactly 20 dimensions.
    #
    # The following probe fields are execution metadata
    # and MUST NOT enter the descriptor:
    #
    #   probe_steps
    #   probe_validation_episodes
    #   probe_elapsed_seconds
    # --------------------------------------------------

    probe_metadata = {
        "probe_steps",
        "probe_validation_episodes",
        "probe_elapsed_seconds",
    }

    probe_columns = [
        column
        for column in df.columns
        if (
            str(column).startswith(
                "probe_"
            )
            and column
            not in probe_metadata
        )
    ]

    if len(probe_columns) != 20:
        raise RuntimeError(
            "Probe descriptor family: "
            "expected 20 columns, "
            f"observed {len(probe_columns)}.\n"
            f"Columns: {probe_columns}"
        )

    for column in probe_columns:
        if not is_numeric_dtype(
            df[column]
        ):
            raise RuntimeError(
                f"Probe descriptor column "
                f"{column!r} is not numeric."
            )

    descriptor_columns.extend(
        probe_columns
    )

    # --------------------------------------------------
    # Final frozen descriptor dimensionality.
    # --------------------------------------------------

    if len(
        descriptor_columns
    ) != EXPECTED_DESCRIPTOR_DIM:
        raise RuntimeError(
            "Frozen descriptor schema "
            f"expected "
            f"{EXPECTED_DESCRIPTOR_DIM} "
            "features, observed "
            f"{len(descriptor_columns)}."
        )

    if len(
        descriptor_columns
    ) != len(
        set(
            descriptor_columns
        )
    ):
        raise RuntimeError(
            "Duplicate descriptor columns "
            "detected."
        )

    # Explicitly ensure extraction metadata cannot
    # accidentally enter the descriptor.
    forbidden = {
        "descriptor_episodes",
        "descriptor_mean_std",
        "probe_steps",
        "probe_validation_episodes",
        "probe_elapsed_seconds",
    }

    leakage = (
        forbidden
        & set(
            descriptor_columns
        )
    )

    if leakage:
        raise RuntimeError(
            "Metadata leakage into "
            "descriptor features: "
            f"{sorted(leakage)}"
        )

    return descriptor_columns

def load_descriptor_split(
    split: str,
):
    path = resolve_descriptor_file(
        split
    )

    df = pd.read_csv(path)

    expected = EXPECTED_TASKS[
        split
    ]

    if len(df) != expected:
        raise RuntimeError(
            f"{split}: expected "
            f"{expected} descriptor rows, "
            f"observed {len(df)}."
        )

    required = {
        "task_id",
        *PROTOCOL_COLUMNS,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{split}: missing required "
            f"columns {sorted(missing)}"
        )

    if (
        df["task_id"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"{split}: duplicate task IDs."
        )

    if "split" in df.columns:
        values = set(
            df["split"]
            .astype(str)
            .unique()
        )

        if values != {split}:
            raise RuntimeError(
                f"{split}: descriptor "
                f"split mismatch: {values}"
            )

    return path, df


# ----------------------------------------------------------------------
# Historical performance observations
# ----------------------------------------------------------------------

def stage1_files(
    split: str,
):
    # IMPORTANT:
    # This function is only called for
    # train and validation.
    #
    # NPP never reads test-stage1 labels
    # during recommendation generation.

    if split not in {
        "train",
        "validation",
    }:
        raise RuntimeError(
            "NPP historical labels may "
            "only use train/validation."
        )

    files = sorted(
        (
            ORACLE_ROOT / split
        ).glob(
            "*/*/"
            "stage1_seed101_full64.csv"
        )
    )

    expected = EXPECTED_TASKS[
        split
    ]

    if len(files) != expected:
        raise RuntimeError(
            f"{split}: expected "
            f"{expected} stage1 files, "
            f"observed {len(files)}."
        )

    return files


def task_base_vector(
    row: pd.Series,
    descriptor_columns,
):
    descriptor = (
        row[
            descriptor_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    protocol = np.asarray(
        [
            float(
                row["n_way"]
            ),
            float(
                row["n_shot"]
            ),
            float(
                row["n_query"]
            ),
        ],
        dtype=np.float64,
    )

    return np.concatenate(
        [
            descriptor,
            protocol,
        ]
    )


def build_historical_pairs(
    *,
    split,
    descriptor_df,
    descriptor_columns,
    anchor_encoder,
):
    descriptor_lookup = (
        descriptor_df
        .copy()
        .assign(
            task_id=lambda x:
                x[
                    "task_id"
                ].astype(str)
        )
        .set_index(
            "task_id",
            drop=False,
        )
    )

    anchor_ids = list(
        anchor_encoder.anchor_ids
    )

    X_rows = []
    y_rows = []
    task_rows = []
    config_rows = []

    files = stage1_files(
        split
    )

    for file_index, path in enumerate(
        files,
        start=1,
    ):
        table = pd.read_csv(path)

        if len(table) != 64:
            raise RuntimeError(
                f"{path}: expected "
                "64 anchors."
            )

        table[
            "config_id"
        ] = (
            table[
                "config_id"
            ].astype(str)
        )

        if (
            table[
                "config_id"
            ].nunique()
            != 64
        ):
            raise RuntimeError(
                f"{path}: duplicate anchors."
            )

        if set(
            table["config_id"]
        ) != set(anchor_ids):
            raise RuntimeError(
                f"{path}: anchor "
                "portfolio mismatch."
            )

        task_ids = (
            table[
                "task_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(task_ids) != 1:
            raise RuntimeError(
                f"{path}: ambiguous task ID."
            )

        task_id = task_ids[0]

        if task_id not in (
            descriptor_lookup.index
        ):
            raise RuntimeError(
                f"{task_id}: missing "
                "descriptor row."
            )

        descriptor_row = (
            descriptor_lookup.loc[
                task_id
            ]
        )

        base = task_base_vector(
            descriptor_row,
            descriptor_columns,
        )

        performance = dict(
            zip(
                table[
                    "config_id"
                ].astype(str),
                table[
                    "validation_accuracy_mean"
                ].astype(float),
            )
        )

        for config_id in anchor_ids:
            anchor_vector = (
                anchor_encoder.encode(
                    config_id
                )
            )

            X_rows.append(
                np.concatenate(
                    [
                        base,
                        anchor_vector,
                    ]
                )
            )

            y_rows.append(
                float(
                    performance[
                        config_id
                    ]
                )
            )

            task_rows.append(
                task_id
            )

            config_rows.append(
                config_id
            )

    X = np.asarray(
        X_rows,
        dtype=np.float32,
    )

    y = np.asarray(
        y_rows,
        dtype=np.float32,
    )

    return {
        "X": X,
        "y": y,
        "task_ids": np.asarray(
            task_rows,
            dtype=object,
        ),
        "config_ids": np.asarray(
            config_rows,
            dtype=object,
        ),
    }


def build_test_candidates(
    *,
    descriptor_df,
    descriptor_columns,
    anchor_encoder,
):
    anchor_ids = list(
        anchor_encoder.anchor_ids
    )

    rows = []

    descriptor_df = (
        descriptor_df
        .copy()
        .sort_values(
            "task_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    for _, row in (
        descriptor_df.iterrows()
    ):
        task_id = str(
            row["task_id"]
        )

        base = task_base_vector(
            row,
            descriptor_columns,
        )

        for config_id in anchor_ids:
            rows.append(
                {
                    "task_id":
                        task_id,

                    "dataset":
                        (
                            str(row["dataset"])
                            if "dataset"
                            in row.index
                            else ""
                        ),

                    "regime":
                        (
                            f"{int(row['n_way'])}w"
                            f"{int(row['n_shot'])}s"
                        ),

                    "n_way":
                        int(
                            row["n_way"]
                        ),

                    "n_shot":
                        int(
                            row["n_shot"]
                        ),

                    "n_query":
                        int(
                            row["n_query"]
                        ),

                    "config_id":
                        config_id,

                    "X":
                        np.concatenate(
                            [
                                base,
                                anchor_encoder.encode(
                                    config_id
                                ),
                            ]
                        ).astype(
                            np.float32
                        ),
                }
            )

    return rows


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class NeuralPerformancePredictor(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):
        return (
            self.net(x)
            .squeeze(-1)
        )


def fit_scaler(
    X: np.ndarray,
):
    mean = X.mean(
        axis=0
    ).astype(
        np.float32
    )

    std = X.std(
        axis=0
    ).astype(
        np.float32
    )

    std[
        std < 1e-8
    ] = 1.0

    return mean, std


def apply_scaler(
    X,
    mean,
    std,
):
    return (
        X - mean
    ) / std


@torch.no_grad()
def predict_array(
    *,
    model,
    X,
    device,
    batch_size=4096,
):
    model.eval()

    outputs = []

    for start in range(
        0,
        len(X),
        batch_size,
    ):
        batch = torch.from_numpy(
            X[
                start:
                start + batch_size
            ]
        ).float().to(
            device
        )

        outputs.append(
            model(batch)
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


def validation_ranking_metrics(
    *,
    predictions,
    y,
    task_ids,
    config_ids,
):
    frame = pd.DataFrame(
        {
            "task_id":
                task_ids,

            "config_id":
                config_ids,

            "prediction":
                predictions,

            "actual":
                y,
        }
    )

    recommendation_rows = []

    for task_id, group in (
        frame.groupby(
            "task_id",
            sort=True,
        )
    ):
        predicted_ranked = (
            group.sort_values(
                by=[
                    "prediction",
                    "config_id",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="mergesort",
            )
        )

        actual_ranked = (
            group.sort_values(
                by=[
                    "actual",
                    "config_id",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="mergesort",
            )
        )

        selected = (
            predicted_ranked.iloc[0]
        )

        best = (
            actual_ranked.iloc[0]
        )

        selected_value = float(
            selected["actual"]
        )

        best_value = float(
            best["actual"]
        )

        recommendation_rows.append(
            {
                "task_id":
                    task_id,

                "selected_config_id":
                    str(
                        selected[
                            "config_id"
                        ]
                    ),

                "selected_predicted_accuracy":
                    float(
                        selected[
                            "prediction"
                        ]
                    ),

                "selected_actual_accuracy":
                    selected_value,

                "stage1_best_config_id":
                    str(
                        best[
                            "config_id"
                        ]
                    ),

                "stage1_best_accuracy":
                    best_value,

                "regret_pp":
                    100.0
                    * (
                        best_value
                        - selected_value
                    ),

                "hard_hit":
                    bool(
                        str(
                            selected[
                                "config_id"
                            ]
                        )
                        == str(
                            best[
                                "config_id"
                            ]
                        )
                    ),

                "best_value_hit":
                    bool(
                        np.isclose(
                            selected_value,
                            best_value,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ),
            }
        )

    rec = pd.DataFrame(
        recommendation_rows
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    predictions
                    - y
                ) ** 2
            )
        )
    )

    return {
        "rmse":
            rmse,

        "mean_regret_pp":
            float(
                rec[
                    "regret_pp"
                ].mean()
            ),

        "median_regret_pp":
            float(
                rec[
                    "regret_pp"
                ].median()
            ),

        "hard_hit_rate":
            float(
                rec[
                    "hard_hit"
                ].mean()
            ),

        "best_value_hit_rate":
            float(
                rec[
                    "best_value_hit"
                ].mean()
            ),

        "recommendations":
            rec,
    }


def train_with_validation(
    *,
    seed,
    train_data,
    validation_data,
    device,
    output_root,
):
    set_seed(seed)

    X_train = train_data[
        "X"
    ]

    y_train = train_data[
        "y"
    ]

    X_val = validation_data[
        "X"
    ]

    y_val = validation_data[
        "y"
    ]

    mean, std = fit_scaler(
        X_train
    )

    X_train_scaled = apply_scaler(
        X_train,
        mean,
        std,
    ).astype(
        np.float32
    )

    X_val_scaled = apply_scaler(
        X_val,
        mean,
        std,
    ).astype(
        np.float32
    )

    model = (
        NeuralPerformancePredictor(
            input_dim=(
                X_train.shape[1]
            )
        )
        .to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(
            X_train_scaled
        ),
        torch.from_numpy(
            y_train
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    best_state = None
    best_epoch = None
    best_metric = None
    best_validation = None

    history = []

    stale_epochs = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        losses = []

        for xb, yb in loader:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                    .item()
                )
            )

        predictions = (
            predict_array(
                model=model,
                X=X_val_scaled,
                device=device,
            )
        )

        metrics = (
            validation_ranking_metrics(
                predictions=predictions,
                y=y_val,
                task_ids=(
                    validation_data[
                        "task_ids"
                    ]
                ),
                config_ids=(
                    validation_data[
                        "config_ids"
                    ]
                ),
            )
        )

        candidate_metric = (
            round(
                metrics[
                    "mean_regret_pp"
                ],
                12,
            ),
            round(
                metrics[
                    "rmse"
                ],
                12,
            ),
            epoch,
        )

        history.append(
            {
                "seed":
                    seed,

                "epoch":
                    epoch,

                "train_mse":
                    float(
                        np.mean(
                            losses
                        )
                    ),

                "validation_rmse":
                    metrics[
                        "rmse"
                    ],

                "validation_mean_regret_pp":
                    metrics[
                        "mean_regret_pp"
                    ],

                "validation_median_regret_pp":
                    metrics[
                        "median_regret_pp"
                    ],

                "validation_hard_hit_rate":
                    metrics[
                        "hard_hit_rate"
                    ],

                "validation_best_value_hit_rate":
                    metrics[
                        "best_value_hit_rate"
                    ],
            }
        )

        improved = (
            best_metric is None
            or
            candidate_metric
            < best_metric
        )

        if improved:
            best_metric = (
                candidate_metric
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict()
                .items()
            }

            best_validation = (
                metrics
            )

            stale_epochs = 0

            print(
                f"seed={seed} "
                f"epoch={epoch:3d} "
                f"regret="
                f"{metrics['mean_regret_pp']:.4f}pp "
                f"rmse="
                f"{metrics['rmse']:.5f} "
                f"hard="
                f"{100*metrics['hard_hit_rate']:.2f}% "
                f"value="
                f"{100*metrics['best_value_hit_rate']:.2f}%"
            )

        else:
            stale_epochs += 1

        if (
            stale_epochs
            >= PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "NPP failed to produce "
            "a best model."
        )

    history_path = (
        output_root
        / f"training_history_seed{seed}.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    validation_rec = (
        best_validation[
            "recommendations"
        ]
        .copy()
    )

    validation_rec[
        "seed"
    ] = seed

    return {
        "best_epoch":
            int(
                best_epoch
            ),

        "best_state":
            best_state,

        "scaler_mean":
            mean,

        "scaler_std":
            std,

        "metrics":
            best_validation,

        "validation_recommendations":
            validation_rec,

        "history_path":
            history_path,
    }


def refit_model(
    *,
    seed,
    epochs,
    X,
    y,
    device,
):
    set_seed(seed)

    mean, std = fit_scaler(
        X
    )

    X_scaled = apply_scaler(
        X,
        mean,
        std,
    ).astype(
        np.float32
    )

    model = (
        NeuralPerformancePredictor(
            input_dim=X.shape[1]
        )
        .to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(
            X_scaled
        ),
        torch.from_numpy(
            y
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    for _ in range(
        epochs
    ):
        model.train()

        for xb, yb in loader:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

    return (
        model,
        mean,
        std,
    )


# ----------------------------------------------------------------------
# Test recommendation generation
# ----------------------------------------------------------------------

def recommend_test_tasks(
    *,
    model,
    scaler_mean,
    scaler_std,
    seed,
    test_candidates,
    device,
):
    rows = []
    score_rows = []

    by_task = {}

    for item in test_candidates:
        by_task.setdefault(
            item["task_id"],
            [],
        ).append(item)

    for task_id in sorted(
        by_task
    ):
        candidates = (
            by_task[
                task_id
            ]
        )

        X = np.vstack(
            [
                item["X"]
                for item
                in candidates
            ]
        ).astype(
            np.float32
        )

        X_scaled = apply_scaler(
            X,
            scaler_mean,
            scaler_std,
        ).astype(
            np.float32
        )

        start = time.perf_counter()

        predictions = (
            predict_array(
                model=model,
                X=X_scaled,
                device=device,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        ranking = pd.DataFrame(
            {
                "config_id":
                    [
                        item[
                            "config_id"
                        ]
                        for item
                        in candidates
                    ],

                "predicted_validation_accuracy":
                    predictions,
            }
        ).sort_values(
            by=[
                "predicted_validation_accuracy",
                "config_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        selected = (
            ranking.iloc[0]
        )

        meta = candidates[0]

        rows.append(
            {
                "method":
                    "probe_npp",

                "task_id":
                    task_id,

                "dataset":
                    meta[
                        "dataset"
                    ],

                "regime":
                    meta[
                        "regime"
                    ],

                "n_way":
                    meta[
                        "n_way"
                    ],

                "n_shot":
                    meta[
                        "n_shot"
                    ],

                "n_query":
                    meta[
                        "n_query"
                    ],

                "npp_seed":
                    int(seed),

                "selected_config_id":
                    str(
                        selected[
                            "config_id"
                        ]
                    ),

                "predicted_validation_accuracy":
                    float(
                        selected[
                            "predicted_validation_accuracy"
                        ]
                    ),

                "candidate_count":
                    64,

                "ranking_time_seconds":
                    float(
                        elapsed
                    ),

                "target_task_validation_evaluations":
                    0,

                "test_stage1_labels_used":
                    False,

                "test_bank_used":
                    False,
            }
        )

        for _, candidate in (
            ranking.iterrows()
        ):
            score_rows.append(
                {
                    "task_id":
                        task_id,

                    "npp_seed":
                        int(seed),

                    "config_id":
                        str(
                            candidate[
                                "config_id"
                            ]
                        ),

                    "predicted_validation_accuracy":
                        float(
                            candidate[
                                "predicted_validation_accuracy"
                            ]
                        ),

                    "test_stage1_labels_used":
                        False,

                    "test_bank_used":
                        False,
                }
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            score_rows
        ),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if (
            args.device.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------
    # Descriptor loading.
    # --------------------------------------------------

    descriptor_paths = {}
    descriptors = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        (
            path,
            df,
        ) = load_descriptor_split(
            split
        )

        descriptor_paths[
            split
        ] = path

        descriptors[
            split
        ] = df

    descriptor_columns = (
        infer_descriptor_columns(
            descriptors[
                "train"
            ]
        )
    )

    # Same exact descriptor schema
    # across all three splits.
    for split in (
        "validation",
        "test",
    ):
        columns = (
            infer_descriptor_columns(
                descriptors[
                    split
                ]
            )
        )

        if (
            columns
            != descriptor_columns
        ):
            raise RuntimeError(
                f"{split}: descriptor "
                "column order mismatch."
            )

    print()
    print(
        "Descriptor dimension:",
        len(
            descriptor_columns
        ),
    )

    print(
        "Protocol dimension:",
        len(
            PROTOCOL_COLUMNS
        ),
    )

    print(
        "Task-side NPP dimension:",
        len(
            descriptor_columns
        )
        + len(
            PROTOCOL_COLUMNS
        ),
    )

    print()
    print(
        "Descriptor columns:"
    )

    for index, column in enumerate(
        descriptor_columns
    ):
        print(
            f"{index:3d}  {column}"
        )

    # --------------------------------------------------
    # Anchor encoding.
    # --------------------------------------------------

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

    anchor_encoder = (
        AnchorFeatureEncoder(
            adapter
        )
    )

    if len(
        anchor_encoder.anchor_ids
    ) != 64:
        raise RuntimeError(
            "Expected 64 anchors."
        )

    print()
    print(
        "Anchor encoding dimension:",
        anchor_encoder.matrix.shape[
            1
        ],
    )

    # --------------------------------------------------
    # Historical training observations.
    # --------------------------------------------------

    print()
    print(
        "Building meta-train "
        "task-anchor observations..."
    )

    train_data = (
        build_historical_pairs(
            split="train",

            descriptor_df=(
                descriptors[
                    "train"
                ]
            ),

            descriptor_columns=(
                descriptor_columns
            ),

            anchor_encoder=(
                anchor_encoder
            ),
        )
    )

    print(
        "Building meta-validation "
        "task-anchor observations..."
    )

    validation_data = (
        build_historical_pairs(
            split="validation",

            descriptor_df=(
                descriptors[
                    "validation"
                ]
            ),

            descriptor_columns=(
                descriptor_columns
            ),

            anchor_encoder=(
                anchor_encoder
            ),
        )
    )

    if (
        train_data[
            "X"
        ].shape[0]
        != 560 * 64
    ):
        raise RuntimeError(
            "Unexpected train "
            "observation count."
        )

    if (
        validation_data[
            "X"
        ].shape[0]
        != 120 * 64
    ):
        raise RuntimeError(
            "Unexpected validation "
            "observation count."
        )

    if (
        train_data[
            "X"
        ].shape[1]
        != validation_data[
            "X"
        ].shape[1]
    ):
        raise RuntimeError(
            "Train/validation input "
            "dimension mismatch."
        )

    print()
    print("=" * 90)
    print(
        "NPP DATA AUDIT"
    )
    print("=" * 90)

    print(
        "Train tasks:",
        len(
            np.unique(
                train_data[
                    "task_ids"
                ]
            )
        ),
    )

    print(
        "Train observations:",
        len(
            train_data[
                "y"
            ]
        ),
    )

    print(
        "Validation tasks:",
        len(
            np.unique(
                validation_data[
                    "task_ids"
                ]
            )
        ),
    )

    print(
        "Validation observations:",
        len(
            validation_data[
                "y"
            ]
        ),
    )

    print(
        "Input dimension:",
        train_data[
            "X"
        ].shape[1],
    )

    print(
        "Test-stage1 labels used: False"
    )

    print(
        "Test bank used: False"
    )

    schema = {
        "schema_version":
            1,

        "method":
            "probe_npp",

        "descriptor_dimension":
            84,

        "protocol_dimension":
            3,

        "task_side_dimension":
            87,

        "anchor_encoding_dimension":
            int(
                anchor_encoder
                .matrix.shape[1]
            ),

        "total_input_dimension":
            int(
                train_data[
                    "X"
                ].shape[1]
            ),

        "descriptor_columns":
            descriptor_columns,

        "protocol_columns":
            PROTOCOL_COLUMNS,

        "anchor_encoding_features":
            anchor_encoder.feature_names,

        "descriptor_files": {
            split: {
                "path":
                    str(
                        descriptor_paths[
                            split
                        ]
                    ),

                "sha256":
                    sha256_file(
                        descriptor_paths[
                            split
                        ]
                    ),
            }

            for split
            in descriptor_paths
        },

        "anchor_manifest":
            str(
                DEFAULT_ANCHORS
            ),

        "anchor_manifest_sha256":
            sha256_file(
                DEFAULT_ANCHORS
            ),

        "historical_label_splits": [
            "train",
            "validation",
        ],

        "test_stage1_labels_used":
            False,

        "test_bank_used":
            False,
    }

    schema_path = (
        args.output_root
        / "npp_schema.json"
    )

    schema_path.write_text(
        json.dumps(
            schema,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if args.audit_only:
        print()
        print(
            "NPP SCHEMA / DATA AUDIT: PASS"
        )
        return

    # --------------------------------------------------
    # Model selection on 560 / 120.
    # --------------------------------------------------

    validation_summaries = []
    validation_recommendations = []
    selected_epochs = {}

    for seed in MODEL_SEEDS:
        print()
        print("=" * 90)
        print(
            f"NPP SEED {seed}"
        )
        print("=" * 90)

        result = (
            train_with_validation(
                seed=seed,

                train_data=(
                    train_data
                ),

                validation_data=(
                    validation_data
                ),

                device=device,

                output_root=(
                    args.output_root
                ),
            )
        )

        selected_epochs[
            str(seed)
        ] = result[
            "best_epoch"
        ]

        metrics = result[
            "metrics"
        ]

        validation_summaries.append(
            {
                "seed":
                    seed,

                "selected_epoch":
                    result[
                        "best_epoch"
                    ],

                "validation_rmse":
                    metrics[
                        "rmse"
                    ],

                "validation_mean_regret_pp":
                    metrics[
                        "mean_regret_pp"
                    ],

                "validation_median_regret_pp":
                    metrics[
                        "median_regret_pp"
                    ],

                "validation_hard_hit_rate":
                    metrics[
                        "hard_hit_rate"
                    ],

                "validation_best_value_hit_rate":
                    metrics[
                        "best_value_hit_rate"
                    ],
            }
        )

        validation_recommendations.append(
            result[
                "validation_recommendations"
            ]
        )

    validation_summary_df = (
        pd.DataFrame(
            validation_summaries
        )
    )

    validation_rec_df = (
        pd.concat(
            validation_recommendations,
            ignore_index=True,
        )
    )

    validation_summary_path = (
        args.output_root
        / "validation_summary.csv"
    )

    validation_rec_path = (
        args.output_root
        / "validation_recommendations.csv"
    )

    validation_summary_df.to_csv(
        validation_summary_path,
        index=False,
    )

    validation_rec_df.to_csv(
        validation_rec_path,
        index=False,
    )

    # --------------------------------------------------
    # Refit on 680 train + validation tasks.
    # --------------------------------------------------

    X_full = np.concatenate(
        [
            train_data["X"],
            validation_data["X"],
        ],
        axis=0,
    )

    y_full = np.concatenate(
        [
            train_data["y"],
            validation_data["y"],
        ],
        axis=0,
    )

    if len(X_full) != (
        680 * 64
    ):
        raise RuntimeError(
            "Expected 43,520 full "
            "historical observations."
        )

    test_candidates = (
        build_test_candidates(
            descriptor_df=(
                descriptors[
                    "test"
                ]
            ),

            descriptor_columns=(
                descriptor_columns
            ),

            anchor_encoder=(
                anchor_encoder
            ),
        )
    )

    if len(test_candidates) != (
        120 * 64
    ):
        raise RuntimeError(
            "Expected 7,680 "
            "test candidate rows."
        )

    all_test_recommendations = []
    all_test_scores = []

    model_dir = (
        args.output_root
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in MODEL_SEEDS:
        epochs = (
            selected_epochs[
                str(seed)
            ]
        )

        print()
        print(
            f"Refitting seed={seed} "
            f"on 680 tasks for "
            f"{epochs} epochs..."
        )

        (
            model,
            mean,
            std,
        ) = refit_model(
            seed=seed,

            epochs=epochs,

            X=X_full,

            y=y_full,

            device=device,
        )

        checkpoint_path = (
            model_dir
            / f"npp_seed{seed}.pt"
        )

        torch.save(
            {
                "seed":
                    seed,

                "epochs":
                    epochs,

                "input_dim":
                    int(
                        X_full.shape[1]
                    ),

                "state_dict":
                    {
                        key:
                            value.detach()
                            .cpu()

                        for key, value
                        in model.state_dict()
                        .items()
                    },

                "scaler_mean":
                    mean,

                "scaler_std":
                    std,

                "descriptor_columns":
                    descriptor_columns,

                "test_stage1_labels_used":
                    False,

                "test_bank_used":
                    False,
            },
            checkpoint_path,
        )

        (
            recommendations,
            candidate_scores,
        ) = recommend_test_tasks(
            model=model,

            scaler_mean=mean,

            scaler_std=std,

            seed=seed,

            test_candidates=(
                test_candidates
            ),

            device=device,
        )

        all_test_recommendations.append(
            recommendations
        )

        all_test_scores.append(
            candidate_scores
        )

    test_recommendations = (
        pd.concat(
            all_test_recommendations,
            ignore_index=True,
        )
    )

    test_scores = pd.concat(
        all_test_scores,
        ignore_index=True,
    )

    if len(
        test_recommendations
    ) != 120 * 5:
        raise RuntimeError(
            "Expected 600 frozen "
            "NPP recommendations."
        )

    if len(
        test_scores
    ) != 120 * 64 * 5:
        raise RuntimeError(
            "Expected 38,400 "
            "candidate-score rows."
        )

    if (
        test_recommendations[
            "test_stage1_labels_used"
        ].astype(bool).any()
    ):
        raise RuntimeError(
            "Test-stage1 leakage."
        )

    if (
        test_recommendations[
            "test_bank_used"
        ].astype(bool).any()
    ):
        raise RuntimeError(
            "Test-bank leakage."
        )

    recommendations_path = (
        args.output_root
        / "test_recommendations.csv"
    )

    scores_path = (
        args.output_root
        / "test_candidate_scores.csv"
    )

    test_recommendations.to_csv(
        recommendations_path,
        index=False,
    )

    test_scores.to_csv(
        scores_path,
        index=False,
    )

    # --------------------------------------------------
    # Freeze BEFORE final test-bank evaluation.
    # --------------------------------------------------

    protocol = {
        "schema_version":
            1,

        "method":
            "probe_npp",

        "description":
            (
                "Neural Performance "
                "Predictor trained to map "
                "(84-D task descriptor + "
                "3-D episodic context + "
                "candidate-anchor encoding) "
                "to predicted validation "
                "accuracy."
            ),

        "model":
            (
                "MLP: input -> 128 ReLU "
                "Dropout(0.1) -> 64 ReLU "
                "-> scalar"
            ),

        "loss":
            "MSE",

        "optimizer":
            "Adam",

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "maximum_epochs":
            MAX_EPOCHS,

        "early_stopping_patience":
            PATIENCE,

        "model_seeds":
            list(
                MODEL_SEEDS
            ),

        "model_selection":
            (
                "minimum mean validation "
                "recommendation regret; "
                "RMSE then earlier epoch "
                "used as tie-breaks"
            ),

        "selected_epochs":
            selected_epochs,

        "meta_train_tasks":
            560,

        "meta_validation_tasks":
            120,

        "refit_tasks":
            680,

        "historical_train_pairs":
            560 * 64,

        "historical_validation_pairs":
            120 * 64,

        "candidate_count_per_test_task":
            64,

        "test_task_count":
            120,

        "search_space":
            (
                "same frozen V2 "
                "64-anchor portfolio"
            ),

        "target_task_validation_evaluations":
            0,

        "test_stage1_labels_used":
            False,

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
            "probe_npp",

        "selection_frozen":
            True,

        "test_stage1_labels_used":
            False,

        "test_evaluation_performed":
            False,

        "files": {
            "npp_schema.json":
                sha256_file(
                    schema_path
                ),

            "validation_summary.csv":
                sha256_file(
                    validation_summary_path
                ),

            "validation_recommendations.csv":
                sha256_file(
                    validation_rec_path
                ),

            "protocol.json":
                sha256_file(
                    protocol_path
                ),

            "test_recommendations.csv":
                sha256_file(
                    recommendations_path
                ),

            "test_candidate_scores.csv":
                sha256_file(
                    scores_path
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
        schema_path,
        validation_summary_path,
        validation_rec_path,
        protocol_path,
        recommendations_path,
        scores_path,
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

    print()
    print("=" * 90)
    print(
        "NEURAL PERFORMANCE "
        "PREDICTOR SUMMARY"
    )
    print("=" * 90)

    print(
        "Historical train tasks:",
        560,
    )

    print(
        "Historical train pairs:",
        560 * 64,
    )

    print(
        "Validation tasks:",
        120,
    )

    print(
        "Validation pairs:",
        120 * 64,
    )

    print(
        "Refit tasks:",
        680,
    )

    print(
        "NPP seeds:",
        list(
            MODEL_SEEDS
        ),
    )

    print(
        "Selected epochs:",
        selected_epochs,
    )

    print(
        "Mean validation regret:",
        f"{validation_summary_df['validation_mean_regret_pp'].mean():.4f} pp"
    )

    print(
        "Mean validation hard hit:",
        f"{100 * validation_summary_df['validation_hard_hit_rate'].mean():.2f}%"
    )

    print(
        "Mean validation best-value hit:",
        f"{100 * validation_summary_df['validation_best_value_hit_rate'].mean():.2f}%"
    )

    print(
        "Frozen test recommendations:",
        len(
            test_recommendations
        ),
    )

    print(
        "Test candidate-score rows:",
        len(
            test_scores
        ),
    )

    print(
        "Unique selected anchors:",
        test_recommendations[
            "selected_config_id"
        ].nunique(),
    )

    print(
        "Target-task validation "
        "evaluations: 0"
    )

    print(
        "Test-stage1 labels used: False"
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
        "NPP SELECTION / LOCK: PASS"
    )


if __name__ == "__main__":
    main()
