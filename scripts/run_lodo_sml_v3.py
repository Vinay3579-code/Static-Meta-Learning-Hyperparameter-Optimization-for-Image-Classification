from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sml_hpo.descriptors.zero import ZERO_FEATURE_NAMES


# ============================================================
# Frozen experiment definition
# ============================================================

DATASETS = [
    "omniglot",
    "cifar100",
    "miniimagenet",
    "dtd",
    "flowers102",
]

SEEDS = [0, 1, 2, 3, 4]

PROTOCOL_COLUMNS = [
    "n_way",
    "n_shot",
    "n_query",
]

MAX_EPOCHS = 400
PATIENCE = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_dataset(value) -> str:
    value = str(value).strip().lower()

    aliases = {
        "cifar-100": "cifar100",
        "cifar_100": "cifar100",
        "mini-imagenet": "miniimagenet",
        "mini_imagenet": "miniimagenet",
        "flowers-102": "flowers102",
        "flowers_102": "flowers102",
    }

    return aliases.get(value, value)


def parse_equivalent(value) -> list[str]:
    parsed = json.loads(str(value))

    if not isinstance(parsed, list):
        raise ValueError(
            "Expected oracle-equivalent configuration list"
        )

    return [
        str(x)
        for x in parsed
    ]


def determine_hard_label_column(
    table: pd.DataFrame,
) -> str | None:

    candidates = [
        "oracle_config_id",
        "hard_oracle_config_id",
        "best_config_id",
        "oracle_best_config_id",
        "reference_config_id",
    ]

    for column in candidates:
        if column in table.columns:
            return column

    return None


# ============================================================
# Model
# ============================================================

class MetaMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Inputs / targets
# ============================================================

def build_input(
    table: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:

    required = set(
        feature_names
        + PROTOCOL_COLUMNS
    )

    missing = required - set(
        table.columns
    )

    if missing:
        raise RuntimeError(
            "Missing model-input columns: "
            f"{sorted(missing)}"
        )

    descriptor = table[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    protocol = table[
        PROTOCOL_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )

    x = np.concatenate(
        [
            descriptor,
            protocol,
        ],
        axis=1,
    )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Non-finite model input"
        )

    return x


def build_vocabulary(
    source_train: pd.DataFrame,
) -> list[str]:
    """
    CRITICAL:
    Vocabulary comes from SOURCE META-TRAIN ONLY.

    Source validation does not define new classes,
    and the held-out dataset cannot influence it.
    """

    vocabulary = set()

    for value in source_train[
        "oracle_equivalent_config_ids_json"
    ]:
        vocabulary.update(
            parse_equivalent(value)
        )

    vocabulary = sorted(vocabulary)

    if not vocabulary:
        raise RuntimeError(
            "Empty source-training vocabulary"
        )

    return vocabulary


def build_soft_targets(
    table: pd.DataFrame,
    vocabulary: list[str],
) -> np.ndarray:

    vocab_index = {
        config_id: index
        for index, config_id
        in enumerate(vocabulary)
    }

    targets = np.zeros(
        (
            len(table),
            len(vocabulary),
        ),
        dtype=np.float32,
    )

    missing_rows = []

    for row_index, value in enumerate(
        table[
            "oracle_equivalent_config_ids_json"
        ]
    ):
        equivalents = parse_equivalent(
            value
        )

        represented = [
            config
            for config in equivalents
            if config in vocab_index
        ]

        if not represented:
            missing_rows.append(
                row_index
            )
            continue

        probability = (
            1.0
            / len(represented)
        )

        for config in represented:
            targets[
                row_index,
                vocab_index[config],
            ] = probability

    if missing_rows:
        task_ids = table.iloc[
            missing_rows
        ]["task_id"].tolist()

        raise RuntimeError(
            "Some SOURCE tasks have no "
            "oracle-equivalent configuration "
            "inside the SOURCE-TRAIN vocabulary.\n"
            f"Count: {len(task_ids)}\n"
            f"Examples: {task_ids[:10]}"
        )

    return targets


# ============================================================
# Prediction / validation
# ============================================================

def predict_configs(
    model: nn.Module,
    x: np.ndarray,
    vocabulary: list[str],
    device: torch.device,
) -> list[str]:

    model.eval()

    tensor = torch.tensor(
        x,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(tensor)

        indices = (
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

    return [
        vocabulary[int(index)]
        for index in indices
    ]


def validation_metrics(
    predictions: list[str],
    table: pd.DataFrame,
    hard_label_column: str | None,
) -> dict:

    equivalent_correct = []

    hard_correct = []

    for index, prediction in enumerate(
        predictions
    ):
        equivalent_set = set(
            parse_equivalent(
                table.iloc[index][
                    "oracle_equivalent_config_ids_json"
                ]
            )
        )

        equivalent_correct.append(
            prediction
            in equivalent_set
        )

        if hard_label_column is not None:
            hard_correct.append(
                prediction
                == str(
                    table.iloc[index][
                        hard_label_column
                    ]
                )
            )

    equivalent_rate = float(
        np.mean(
            equivalent_correct
        )
    )

    hard_rate = (
        float(
            np.mean(
                hard_correct
            )
        )
        if hard_correct
        else 0.0
    )

    return {
        "equivalent_rate":
            equivalent_rate,

        "hard_accuracy":
            hard_rate,
    }


# ============================================================
# Train with source-domain validation
# ============================================================

def select_epoch(
    *,
    source_train: pd.DataFrame,
    source_validation: pd.DataFrame,
    features: list[str],
    vocabulary: list[str],
    seed: int,
    device: torch.device,
) -> dict:

    set_seed(seed)

    x_train_raw = build_input(
        source_train,
        features,
    )

    x_val_raw = build_input(
        source_validation,
        features,
    )

    # CRITICAL:
    # scaler fit on source TRAIN ONLY.
    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        x_train_raw
    )

    x_val = scaler.transform(
        x_val_raw
    )

    y_train = build_soft_targets(
        source_train,
        vocabulary,
    )

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    loader = DataLoader(
        TensorDataset(
            x_train_tensor,
            y_train_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = MetaMLP(
        input_dim=len(features) + 3,
        output_dim=len(vocabulary),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    hard_column = (
        determine_hard_label_column(
            source_validation
        )
    )

    best_equivalent = -1.0
    best_hard = -1.0
    best_epoch = -1

    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_examples = 0

        for x_batch, y_batch in loader:

            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x_batch
            )

            log_probabilities = (
                torch.log_softmax(
                    logits,
                    dim=1,
                )
            )

            loss = -(
                y_batch
                * log_probabilities
            ).sum(
                dim=1
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            optimizer.step()

            batch_size = (
                x_batch.shape[0]
            )

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_examples += (
                batch_size
            )

        train_loss = (
            total_loss
            / total_examples
        )

        predictions = predict_configs(
            model,
            x_val,
            vocabulary,
            device,
        )

        metrics = validation_metrics(
            predictions,
            source_validation,
            hard_column,
        )

        eq_rate = (
            metrics[
                "equivalent_rate"
            ]
        )

        hard_rate = (
            metrics[
                "hard_accuracy"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation_equivalent_rate":
                    eq_rate,

                "validation_hard_accuracy":
                    hard_rate,
            }
        )

        improved = False

        if (
            eq_rate
            > best_equivalent + 1e-12
        ):
            improved = True

        elif (
            abs(
                eq_rate
                - best_equivalent
            )
            <= 1e-12
            and hard_rate
            > best_hard + 1e-12
        ):
            improved = True

        if improved:

            best_equivalent = (
                eq_rate
            )

            best_hard = (
                hard_rate
            )

            best_epoch = (
                epoch
            )

            best_state = deepcopy(
                {
                    key:
                        value
                        .detach()
                        .cpu()

                    for key, value
                    in model.state_dict()
                    .items()
                }
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            break

    if best_epoch <= 0:
        raise RuntimeError(
            "No valid best epoch selected"
        )

    return {
        "best_epoch":
            best_epoch,

        "best_equivalent_rate":
            best_equivalent,

        "best_hard_accuracy":
            best_hard,

        "epochs_ran":
            len(history),

        "history":
            history,

        "best_state":
            best_state,
    }


# ============================================================
# Refit on source train + source validation
# ============================================================

def refit_and_predict(
    *,
    source_development: pd.DataFrame,
    target_test: pd.DataFrame,
    features: list[str],
    vocabulary: list[str],
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[list[str], StandardScaler, MetaMLP]:

    set_seed(seed)

    x_dev_raw = build_input(
        source_development,
        features,
    )

    x_target_raw = build_input(
        target_test,
        features,
    )

    # Allowed:
    # source train + source validation.
    # Held-out dataset remains absent.
    scaler = StandardScaler()

    x_dev = scaler.fit_transform(
        x_dev_raw
    )

    x_target = scaler.transform(
        x_target_raw
    )

    y_dev = build_soft_targets(
        source_development,
        vocabulary,
    )

    x_dev_tensor = torch.tensor(
        x_dev,
        dtype=torch.float32,
    )

    y_dev_tensor = torch.tensor(
        y_dev,
        dtype=torch.float32,
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    loader = DataLoader(
        TensorDataset(
            x_dev_tensor,
            y_dev_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = MetaMLP(
        input_dim=len(features) + 3,
        output_dim=len(vocabulary),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        for x_batch, y_batch in loader:

            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x_batch
            )

            log_probabilities = (
                torch.log_softmax(
                    logits,
                    dim=1,
                )
            )

            loss = -(
                y_batch
                * log_probabilities
            ).sum(
                dim=1
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )

            optimizer.step()

    predictions = predict_configs(
        model,
        x_target,
        vocabulary,
        device,
    )

    return (
        predictions,
        scaler,
        model,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--development-data",
        type=Path,
        default=Path(
            "results/meta_learning/"
            "datasets_v2/all680.csv"
        ),
    )

    parser.add_argument(
        "--train-descriptors",
        type=Path,
        default=Path(
            "results/descriptors/"
            "final800/merged/"
            "train_full84.csv"
        ),
    )

    parser.add_argument(
        "--validation-descriptors",
        type=Path,
        default=Path(
            "results/descriptors/"
            "final800/merged/"
            "validation_full84.csv"
        ),
    )

    parser.add_argument(
        "--test-descriptors",
        type=Path,
        default=Path(
            "results/descriptors/"
            "final800/merged/"
            "test_full84.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/"
            "lodo_generalization_v3/"
            "zero_sml"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    output_root = (
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load DEVELOPMENT meta-dataset
    # --------------------------------------------------------

    development = pd.read_csv(
        args.development_data
    )

    train_descriptor_meta = (
        pd.read_csv(
            args.train_descriptors
        )[
            [
                "task_id",
                "dataset",
            ]
        ]
        .drop_duplicates()
    )

    validation_descriptor_meta = (
        pd.read_csv(
            args.validation_descriptors
        )[
            [
                "task_id",
                "dataset",
            ]
        ]
        .drop_duplicates()
    )

    target_test = pd.read_csv(
        args.test_descriptors
    )

    for table in [
        development,
        train_descriptor_meta,
        validation_descriptor_meta,
        target_test,
    ]:
        table["dataset"] = (
            table["dataset"]
            .map(
                normalize_dataset
            )
        )

    # --------------------------------------------------------
    # Explicitly reconstruct / verify meta_split
    # --------------------------------------------------------

    train_ids = set(
        train_descriptor_meta[
            "task_id"
        ].astype(str)
    )

    validation_ids = set(
        validation_descriptor_meta[
            "task_id"
        ].astype(str)
    )

    development[
        "task_id"
    ] = development[
        "task_id"
    ].astype(str)

    target_test[
        "task_id"
    ] = target_test[
        "task_id"
    ].astype(str)

    reconstructed_split = []

    for task_id in development[
        "task_id"
    ]:

        if task_id in train_ids:
            reconstructed_split.append(
                "train"
            )

        elif (
            task_id
            in validation_ids
        ):
            reconstructed_split.append(
                "validation"
            )

        else:
            raise RuntimeError(
                "Development task "
                f"{task_id} is not found "
                "in train or validation "
                "descriptor manifests."
            )

    development[
        "_lodo_split"
    ] = reconstructed_split

    # --------------------------------------------------------
    # Global audits
    # --------------------------------------------------------

    if len(development) != 680:
        raise RuntimeError(
            "Expected 680 development tasks, "
            f"found {len(development)}"
        )

    if len(target_test) != 120:
        raise RuntimeError(
            "Expected 120 target-test tasks, "
            f"found {len(target_test)}"
        )

    if development[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate development task IDs"
        )

    if target_test[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate test task IDs"
        )

    if set(
        development[
            "dataset"
        ].unique()
    ) != set(DATASETS):
        raise RuntimeError(
            "Development datasets mismatch"
        )

    if set(
        target_test[
            "dataset"
        ].unique()
    ) != set(DATASETS):
        raise RuntimeError(
            "Test datasets mismatch"
        )

    if len(
        ZERO_FEATURE_NAMES
    ) != 64:
        raise RuntimeError(
            "Zero descriptor must "
            "contain exactly 64 features"
        )

    forbidden_test_columns = [
        column
        for column
        in target_test.columns
        if (
            "test_accuracy"
            in column.lower()
            or "oracle_equivalent"
            in column.lower()
            or "reference_config"
            in column.lower()
        )
    ]

    if forbidden_test_columns:
        raise RuntimeError(
            "Target descriptor table contains "
            "forbidden outcome/reference columns: "
            f"{forbidden_test_columns}"
        )

    print()
    print("=" * 100)
    print(
        "STRICT FIVE-DATASET "
        "LEAVE-ONE-DATASET-OUT SML"
    )
    print("=" * 100)

    print(
        "Development tasks:",
        len(development),
    )

    print(
        "Target descriptor tasks:",
        len(target_test),
    )

    print(
        "Zero descriptor dimensions:",
        len(ZERO_FEATURE_NAMES),
    )

    print(
        "Protocol dimensions:",
        3,
    )

    print(
        "Input dimensions:",
        len(ZERO_FEATURE_NAMES) + 3,
    )

    print(
        "Meta-learner seeds:",
        SEEDS,
    )

    device = torch.device(
        args.device
    )

    all_recommendations = []
    all_validation_rows = []
    fold_metadata = {}

    # ========================================================
    # Five LODO folds
    # ========================================================

    for heldout_dataset in DATASETS:

        print()
        print("=" * 100)
        print(
            "HELD-OUT DATASET:",
            heldout_dataset,
        )
        print("=" * 100)

        source_train = (
            development[
                (
                    development[
                        "dataset"
                    ]
                    != heldout_dataset
                )
                &
                (
                    development[
                        "_lodo_split"
                    ]
                    == "train"
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        source_validation = (
            development[
                (
                    development[
                        "dataset"
                    ]
                    != heldout_dataset
                )
                &
                (
                    development[
                        "_lodo_split"
                    ]
                    == "validation"
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        source_development = (
            pd.concat(
                [
                    source_train,
                    source_validation,
                ],
                ignore_index=True,
            )
        )

        fold_test = (
            target_test[
                target_test[
                    "dataset"
                ]
                == heldout_dataset
            ]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Strict domain-exclusion audits
        # ----------------------------------------------------

        if len(source_train) != 448:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 448 source "
                f"training tasks, found "
                f"{len(source_train)}"
            )

        if len(
            source_validation
        ) != 96:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 96 source "
                "validation tasks, found "
                f"{len(source_validation)}"
            )

        if len(
            source_development
        ) != 544:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 544 source "
                "development tasks"
            )

        if len(fold_test) != 24:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 24 target "
                f"test tasks, found "
                f"{len(fold_test)}"
            )

        for table_name, table in [
            (
                "source_train",
                source_train,
            ),
            (
                "source_validation",
                source_validation,
            ),
            (
                "source_development",
                source_development,
            ),
        ]:

            if (
                heldout_dataset
                in set(
                    table[
                        "dataset"
                    ].unique()
                )
            ):
                raise RuntimeError(
                    f"LEAKAGE: "
                    f"{heldout_dataset} "
                    f"present in "
                    f"{table_name}"
                )

        vocabulary = (
            build_vocabulary(
                source_train
            )
        )

        print(
            "Source train:",
            len(source_train),
        )

        print(
            "Source validation:",
            len(
                source_validation
            ),
        )

        print(
            "Source refit:",
            len(
                source_development
            ),
        )

        print(
            "Target test tasks:",
            len(fold_test),
        )

        print(
            "Fold vocabulary size:",
            len(vocabulary),
        )

        print(
            "Fold vocabulary:",
            vocabulary,
        )

        fold_dir = (
            output_root
            / f"holdout_{heldout_dataset}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save fold membership before training.
        source_train[
            [
                "task_id",
                "dataset",
            ]
        ].to_csv(
            fold_dir
            / "source_train_tasks.csv",
            index=False,
        )

        source_validation[
            [
                "task_id",
                "dataset",
            ]
        ].to_csv(
            fold_dir
            / "source_validation_tasks.csv",
            index=False,
        )

        fold_test[
            [
                "task_id",
                "dataset",
                "n_way",
                "n_shot",
                "n_query",
            ]
        ].to_csv(
            fold_dir
            / "target_test_tasks.csv",
            index=False,
        )

        seed_metadata = {}

        # ====================================================
        # Five independent meta-learner seeds
        # ====================================================

        for seed in SEEDS:

            print()
            print(
                f"[{heldout_dataset}] "
                f"seed {seed}: "
                "selecting epoch..."
            )

            selection = select_epoch(
                source_train=(
                    source_train
                ),
                source_validation=(
                    source_validation
                ),
                features=list(
                    ZERO_FEATURE_NAMES
                ),
                vocabulary=vocabulary,
                seed=seed,
                device=device,
            )

            selected_epoch = int(
                selection[
                    "best_epoch"
                ]
            )

            print(
                f"[{heldout_dataset}] "
                f"seed {seed}: "
                f"epoch={selected_epoch}, "
                "val_equiv="
                f"{100 * selection['best_equivalent_rate']:.2f}%, "
                "val_hard="
                f"{100 * selection['best_hard_accuracy']:.2f}%"
            )

            history_path = (
                fold_dir
                / (
                    f"seed_{seed}_"
                    "validation_history.csv"
                )
            )

            pd.DataFrame(
                selection[
                    "history"
                ]
            ).to_csv(
                history_path,
                index=False,
            )

            # ------------------------------------------------
            # Refit ONLY on four source datasets.
            # ------------------------------------------------

            (
                predictions,
                scaler,
                model,
            ) = refit_and_predict(
                source_development=(
                    source_development
                ),
                target_test=(
                    fold_test
                ),
                features=list(
                    ZERO_FEATURE_NAMES
                ),
                vocabulary=vocabulary,
                seed=seed,
                epochs=selected_epoch,
                device=device,
            )

            if len(
                predictions
            ) != 24:
                raise RuntimeError(
                    "Unexpected target "
                    "prediction count"
                )

            for task_index, prediction in enumerate(
                predictions
            ):
                if prediction not in vocabulary:
                    raise RuntimeError(
                        "Prediction outside "
                        "fold vocabulary"
                    )

                row = (
                    fold_test.iloc[
                        task_index
                    ]
                )

                all_recommendations.append(
                    {
                        "method":
                            "zero_sml_lodo",

                        "heldout_dataset":
                            heldout_dataset,

                        "run_seed":
                            seed,

                        "task_id":
                            str(
                                row[
                                    "task_id"
                                ]
                            ),

                        "regime":
                            (
                                f"{int(row['n_way'])}"
                                f"w"
                                f"{int(row['n_shot'])}"
                                f"s"
                            ),

                        "selected_config_id":
                            prediction,

                        "selected_epoch":
                            selected_epoch,

                        "fold_vocabulary_size":
                            len(vocabulary),
                    }
                )

            all_validation_rows.append(
                {
                    "heldout_dataset":
                        heldout_dataset,

                    "run_seed":
                        seed,

                    "selected_epoch":
                        selected_epoch,

                    "validation_equivalent_rate":
                        selection[
                            "best_equivalent_rate"
                        ],

                    "validation_hard_accuracy":
                        selection[
                            "best_hard_accuracy"
                        ],

                    "epochs_ran":
                        selection[
                            "epochs_ran"
                        ],

                    "vocabulary_size":
                        len(vocabulary),
                }
            )

            seed_metadata[
                str(seed)
            ] = {
                "selected_epoch":
                    selected_epoch,

                "validation_equivalent_rate":
                    float(
                        selection[
                            "best_equivalent_rate"
                        ]
                    ),

                "validation_hard_accuracy":
                    float(
                        selection[
                            "best_hard_accuracy"
                        ]
                    ),
            }

        fold_metadata[
            heldout_dataset
        ] = {
            "source_datasets":
                sorted(
                    set(DATASETS)
                    - {
                        heldout_dataset
                    }
                ),

            "source_train_tasks":
                448,

            "source_validation_tasks":
                96,

            "source_refit_tasks":
                544,

            "target_test_tasks":
                24,

            "vocabulary_source":
                "source meta-train only",

            "vocabulary_size":
                len(vocabulary),

            "vocabulary":
                vocabulary,

            "runs":
                seed_metadata,
        }

    # ========================================================
    # Recommendation lock
    # ========================================================

    recommendations = (
        pd.DataFrame(
            all_recommendations
        )
        .sort_values(
            [
                "heldout_dataset",
                "run_seed",
                "task_id",
            ]
        )
        .reset_index(drop=True)
    )

    validation_summary = (
        pd.DataFrame(
            all_validation_rows
        )
        .sort_values(
            [
                "heldout_dataset",
                "run_seed",
            ]
        )
        .reset_index(drop=True)
    )

    if len(
        recommendations
    ) != 600:
        raise RuntimeError(
            "Expected exactly "
            "5 datasets × 5 seeds × "
            "24 tasks = 600 "
            "recommendations; found "
            f"{len(recommendations)}"
        )

    if len(
        validation_summary
    ) != 25:
        raise RuntimeError(
            "Expected 25 fold/seed "
            "validation rows"
        )

    duplicate_key = [
        "heldout_dataset",
        "run_seed",
        "task_id",
    ]

    if recommendations[
        duplicate_key
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate LODO recommendation"
        )

    recommendation_path = (
        output_root
        / "locked_recommendations.csv"
    )

    recommendations.to_csv(
        recommendation_path,
        index=False,
    )

    validation_path = (
        output_root
        / "validation_run_summary.csv"
    )

    validation_summary.to_csv(
        validation_path,
        index=False,
    )

    recommendation_hash = (
        sha256_file(
            recommendation_path
        )
    )

    validation_hash = (
        sha256_file(
            validation_path
        )
    )

    metadata = {
        "schema_version": 3,

        "experiment":
            "strict_leave_one_dataset_out",

        "method":
            "Zero-SML",

        "datasets":
            DATASETS,

        "meta_learner_seeds":
            SEEDS,

        "descriptor_dimension":
            64,

        "protocol_dimension":
            3,

        "input_dimension":
            67,

        "dataset_identity_used":
            False,

        "target_dataset_used_for_training":
            False,

        "target_dataset_used_for_validation":
            False,

        "target_dataset_used_for_scaler_fit":
            False,

        "target_dataset_used_for_vocabulary":
            False,

        "target_test_scores_read":
            False,

        "target_reference_labels_read":
            False,

        "vocabulary_policy":
            (
                "union of oracle-equivalent "
                "configurations observed in "
                "SOURCE META-TRAIN only"
            ),

        "model_selection":
            (
                "maximize source-validation "
                "oracle-equivalent rate; "
                "hard accuracy used as "
                "secondary tie-break"
            ),

        "refit_policy":
            (
                "refit from scratch on "
                "source meta-train + "
                "source meta-validation "
                "for validation-selected epoch"
            ),

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "gradient_clip":
            GRAD_CLIP,

        "recommendation_rows":
            len(recommendations),

        "validation_rows":
            len(validation_summary),

        "recommendations_sha256":
            recommendation_hash,

        "validation_summary_sha256":
            validation_hash,

        "folds":
            fold_metadata,
    }

    metadata_path = (
        output_root
        / "lock_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    lock_path = (
        output_root
        / "RECOMMENDATION_LOCK.sha256"
    )

    lock_path.write_text(
        (
            f"{recommendation_hash}  "
            f"{recommendation_path}\n"
            f"{validation_hash}  "
            f"{validation_path}\n"
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "LODO ZERO-SML "
        "RECOMMENDATION LOCK"
    )
    print("=" * 100)

    print(
        "Recommendation rows:",
        len(recommendations),
    )

    print(
        "Validation rows:",
        len(validation_summary),
    )

    print(
        "Held-out folds:",
        recommendations[
            "heldout_dataset"
        ].nunique(),
    )

    print(
        "Runs/fold:",
        recommendations.groupby(
            "heldout_dataset"
        )[
            "run_seed"
        ].nunique()
        .to_dict(),
    )

    print(
        "Tasks/fold/run:",
        sorted(
            recommendations.groupby(
                [
                    "heldout_dataset",
                    "run_seed",
                ]
            )[
                "task_id"
            ].nunique()
            .unique()
            .tolist()
        ),
    )

    print()
    print(
        "Recommendation SHA256:",
        recommendation_hash,
    )

    print()
    print(
        "Target test scores read: False"
    )

    print(
        "Target reference labels read: False"
    )

    print()
    print(
        "STRICT LODO ZERO-SML "
        "RECOMMENDATION LOCK: PASS"
    )


if __name__ == "__main__":
    main()
