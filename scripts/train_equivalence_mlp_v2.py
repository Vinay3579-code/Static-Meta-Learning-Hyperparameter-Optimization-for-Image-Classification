from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sml_hpo.descriptors.probe import PROBE_FEATURE_NAMES
from sml_hpo.descriptors.zero import ZERO_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=101,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_equivalent(
    value: str,
) -> set[str]:
    result = json.loads(str(value))

    if not isinstance(result, list):
        raise ValueError(
            "Expected oracle-equivalent JSON list"
        )

    return {
        str(item)
        for item in result
    }


class MetaMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(
                256,
                128,
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(
                128,
                64,
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(
                64,
                output_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


def build_input(
    table: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:

    required = {
        *feature_names,
        "n_way",
        "n_shot",
        "n_query",
    }

    missing = required - set(
        table.columns
    )

    if missing:
        raise RuntimeError(
            "Missing model input columns: "
            f"{sorted(missing)}"
        )

    descriptor = table[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    protocol = table[
        [
            "n_way",
            "n_shot",
            "n_query",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    result = np.concatenate(
        [
            descriptor,
            protocol,
        ],
        axis=1,
    )

    if not np.isfinite(result).all():
        raise FloatingPointError(
            "Non-finite model input"
        )

    return result


def build_soft_targets(
    table: pd.DataFrame,
    vocabulary: list[str],
) -> np.ndarray:

    vocabulary_index = {
        config_id: position
        for position, config_id
        in enumerate(vocabulary)
    }

    targets = np.zeros(
        (
            len(table),
            len(vocabulary),
        ),
        dtype=np.float32,
    )

    for row_index, value in enumerate(
        table[
            "oracle_equivalent_config_ids_json"
        ]
    ):

        equivalent = [
            config_id
            for config_id
            in parse_equivalent(value)
            if config_id
            in vocabulary_index
        ]

        if not equivalent:
            raise RuntimeError(
                "Training task has no "
                "oracle-equivalent configuration "
                "inside output vocabulary."
            )

        probability = (
            1.0 / len(equivalent)
        )

        for config_id in equivalent:
            targets[
                row_index,
                vocabulary_index[
                    config_id
                ],
            ] = probability

    return targets


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    table: pd.DataFrame,
    vocabulary: list[str],
) -> dict:

    model.eval()

    with torch.no_grad():
        logits = model(x)

        predicted_indices = (
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

    predictions = [
        vocabulary[int(index)]
        for index
        in predicted_indices
    ]

    hard_correct = []
    equivalent_correct = []

    for (
        (_, row),
        prediction,
    ) in zip(
        table.iterrows(),
        predictions,
    ):

        hard_correct.append(
            prediction
            == str(
                row["best_config_id"]
            )
        )

        equivalent_correct.append(
            prediction
            in parse_equivalent(
                row[
                    "oracle_equivalent_config_ids_json"
                ]
            )
        )

    return {
        "hard_accuracy": float(
            np.mean(hard_correct)
        ),
        "oracle_equivalent_rate": float(
            np.mean(
                equivalent_correct
            )
        ),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        args.train
    )

    validation = pd.read_csv(
        args.validation
    )

    if len(train) != 560:
        raise RuntimeError(
            f"Expected 560 training tasks, "
            f"found {len(train)}"
        )

    if len(validation) != 120:
        raise RuntimeError(
            f"Expected 120 validation tasks, "
            f"found {len(validation)}"
        )

    feature_names = [
        *ZERO_FEATURE_NAMES,
        *PROBE_FEATURE_NAMES,
    ]

    if len(feature_names) != 84:
        raise RuntimeError(
            f"Expected 84 descriptor features, "
            f"found {len(feature_names)}"
        )

    train_vocabulary = set()

    for value in train[
        "oracle_equivalent_config_ids_json"
    ]:
        train_vocabulary.update(
            parse_equivalent(value)
        )

    vocabulary = sorted(
        train_vocabulary
    )

    validation_vocabulary = set()

    for value in validation[
        "oracle_equivalent_config_ids_json"
    ]:
        validation_vocabulary.update(
            parse_equivalent(value)
        )

    unseen_validation = (
        validation_vocabulary
        - set(vocabulary)
    )

    if unseen_validation:
        raise RuntimeError(
            "Validation equivalent configs "
            "unseen in training: "
            f"{sorted(unseen_validation)}"
        )

    print(
        "Descriptor dimensions:",
        len(feature_names),
    )

    print(
        "Protocol dimensions: 3"
    )

    print(
        "Total input dimensions:",
        len(feature_names) + 3,
    )

    print(
        "Output classes:",
        len(vocabulary),
    )

    print(
        "Class vocabulary:",
        vocabulary,
    )

    x_train = build_input(
        train,
        feature_names,
    )

    x_validation = build_input(
        validation,
        feature_names,
    )

    if x_train.shape != (
        560,
        87,
    ):
        raise RuntimeError(
            f"Unexpected train shape: "
            f"{x_train.shape}"
        )

    if x_validation.shape != (
        120,
        87,
    ):
        raise RuntimeError(
            f"Unexpected validation shape: "
            f"{x_validation.shape}"
        )

    # IMPORTANT:
    # Fit scaler using META-TRAIN ONLY.
    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        x_train
    )

    x_validation = scaler.transform(
        x_validation
    )

    y_train = build_soft_targets(
        train,
        vocabulary,
    )

    device = torch.device(
        args.device
    )

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    x_validation_tensor = torch.tensor(
        x_validation,
        dtype=torch.float32,
        device=device,
    )

    generator = (
        torch.Generator()
        .manual_seed(args.seed)
    )

    loader = DataLoader(
        TensorDataset(
            x_train_tensor,
            y_train_tensor,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = MetaMLP(
        input_dim=87,
        output_dim=len(vocabulary),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_equivalent = -1.0
    best_hard = -1.0
    best_epoch = -1

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
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

            # Soft-label cross entropy.
            loss = -(
                y_batch
                * log_probabilities
            ).sum(
                dim=1
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
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

        metrics = evaluate(
            model,
            x_validation_tensor,
            validation,
            vocabulary,
        )

        equivalent_rate = (
            metrics[
                "oracle_equivalent_rate"
            ]
        )

        hard_accuracy = (
            metrics[
                "hard_accuracy"
            ]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_loss
                ),
                "validation_hard_accuracy": (
                    hard_accuracy
                ),
                "validation_oracle_equivalent_rate": (
                    equivalent_rate
                ),
            }
        )

        improved = (
            equivalent_rate
            > best_equivalent
            + 1e-12
        )

        if (
            abs(
                equivalent_rate
                - best_equivalent
            )
            <= 1e-12
            and hard_accuracy
            > best_hard
            + 1e-12
        ):
            improved = True

        if improved:
            best_equivalent = (
                equivalent_rate
            )

            best_hard = (
                hard_accuracy
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "input_dim": 87,
                    "descriptor_dim": 84,
                    "protocol_dim": 3,
                    "output_dim": len(
                        vocabulary
                    ),
                    "vocabulary": (
                        vocabulary
                    ),
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_hard_accuracy": (
                        best_hard
                    ),
                    "validation_oracle_equivalent_rate": (
                        best_equivalent
                    ),
                },
                args.output_dir
                / "best_model.pt",
            )

        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            print(
                f"epoch={epoch:03d} "
                f"loss={train_loss:.5f} "
                f"hard="
                f"{100 * hard_accuracy:.2f}% "
                f"equiv="
                f"{100 * equivalent_rate:.2f}%"
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                f"Early stopping at "
                f"epoch {epoch}."
            )
            break

    pd.DataFrame(
        history
    ).to_csv(
        args.output_dir
        / "history.csv",
        index=False,
    )

    # Save train-fitted scaler.
    np.savez(
        args.output_dir
        / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_names=np.asarray(
            [
                *feature_names,
                "n_way",
                "n_shot",
                "n_query",
            ],
            dtype=object,
        ),
    )

    checkpoint = torch.load(
        args.output_dir
        / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    final_metrics = evaluate(
        model,
        x_validation_tensor,
        validation,
        vocabulary,
    )

    predictions = validation[
        [
            "task_id",
            "dataset",
            "best_config_id",
            "oracle_equivalent_config_ids_json",
        ]
    ].copy()

    predictions[
        "predicted_config_id"
    ] = final_metrics[
        "predictions"
    ]

    predictions[
        "hard_correct"
    ] = (
        predictions[
            "predicted_config_id"
        ]
        == predictions[
            "best_config_id"
        ]
    )

    predictions[
        "oracle_equivalent"
    ] = [
        prediction
        in parse_equivalent(
            equivalent
        )
        for (
            prediction,
            equivalent,
        ) in zip(
            predictions[
                "predicted_config_id"
            ],
            predictions[
                "oracle_equivalent_config_ids_json"
            ],
        )
    ]

    predictions.to_csv(
        args.output_dir
        / "validation_predictions.csv",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "descriptor_dimension": 84,
        "protocol_dimension": 3,
        "model_input_dimension": 87,
        "output_class_count": len(
            vocabulary
        ),
        "class_vocabulary": (
            vocabulary
        ),
        "best_epoch": int(
            best_epoch
        ),
        "validation_hard_accuracy": float(
            final_metrics[
                "hard_accuracy"
            ]
        ),
        "validation_oracle_equivalent_rate": float(
            final_metrics[
                "oracle_equivalent_rate"
            ]
        ),
    }

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "EQUIVALENCE-AWARE MLP RESULT"
    )
    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Validation hard accuracy:",
        f"{100 * final_metrics['hard_accuracy']:.2f}%",
    )

    print(
        "Validation oracle-equivalent rate:",
        f"{100 * final_metrics['oracle_equivalent_rate']:.2f}%",
    )

    print()
    print(
        "EQUIVALENCE-AWARE MLP TRAINING: PASS"
    )


if __name__ == "__main__":
    main()
