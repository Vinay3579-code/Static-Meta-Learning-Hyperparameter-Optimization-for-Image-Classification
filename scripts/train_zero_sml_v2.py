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
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_equiv(value):
    return set(
        str(x)
        for x in json.loads(str(value))
    )


class MetaMLP(nn.Module):

    def __init__(self, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(67, 256),
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
        return self.net(x)


def make_x(df):
    features = list(ZERO_FEATURE_NAMES)

    if len(features) != 64:
        raise RuntimeError(
            f"Expected 64 zero features, found {len(features)}"
        )

    descriptors = df[
        features
    ].to_numpy(dtype=np.float64)

    protocol = df[
        ["n_way", "n_shot", "n_query"]
    ].to_numpy(dtype=np.float64)

    x = np.concatenate(
        [descriptors, protocol],
        axis=1,
    )

    if x.shape[1] != 67:
        raise RuntimeError(
            f"Expected 67 inputs, got {x.shape[1]}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Non-finite model inputs"
        )

    return x


def make_targets(df, vocab):
    index = {
        config: i
        for i, config in enumerate(vocab)
    }

    y = np.zeros(
        (len(df), len(vocab)),
        dtype=np.float32,
    )

    for row_i, raw in enumerate(
        df["oracle_equivalent_config_ids_json"]
    ):
        ids = [
            x
            for x in parse_equiv(raw)
            if x in index
        ]

        if not ids:
            raise RuntimeError(
                "Task has no equivalent label in vocabulary"
            )

        weight = 1.0 / len(ids)

        for config in ids:
            y[
                row_i,
                index[config],
            ] = weight

    return y


def evaluate(model, x, df, vocab):
    model.eval()

    with torch.no_grad():
        indices = (
            model(x)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

    predictions = [
        vocab[int(i)]
        for i in indices
    ]

    hard = []
    equiv = []

    for (_, row), pred in zip(
        df.iterrows(),
        predictions,
    ):
        hard.append(
            pred
            == str(row["best_config_id"])
        )

        equiv.append(
            pred
            in parse_equiv(
                row[
                    "oracle_equivalent_config_ids_json"
                ]
            )
        )

    return {
        "hard": float(np.mean(hard)),
        "equiv": float(np.mean(equiv)),
        "predictions": predictions,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.validation)

    if len(train) != 560:
        raise RuntimeError("Expected 560 train tasks")

    if len(val) != 120:
        raise RuntimeError("Expected 120 validation tasks")

    vocab_set = set()

    for value in train[
        "oracle_equivalent_config_ids_json"
    ]:
        vocab_set.update(
            parse_equiv(value)
        )

    vocab = sorted(vocab_set)

    val_vocab = set()

    for value in val[
        "oracle_equivalent_config_ids_json"
    ]:
        val_vocab.update(
            parse_equiv(value)
        )

    unseen = val_vocab - set(vocab)

    if unseen:
        raise RuntimeError(
            f"Unseen validation configs: {sorted(unseen)}"
        )

    print("Zero descriptor dimensions: 64")
    print("Protocol dimensions: 3")
    print("Total input dimensions: 67")
    print("Output classes:", len(vocab))

    x_train = make_x(train)
    x_val = make_x(val)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    y_train = make_targets(
        train,
        vocab,
    )

    device = torch.device(args.device)

    train_x = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    train_y = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    val_x = torch.tensor(
        x_val,
        dtype=torch.float32,
        device=device,
    )

    generator = (
        torch.Generator()
        .manual_seed(args.seed)
    )

    loader = DataLoader(
        TensorDataset(
            train_x,
            train_y,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = MetaMLP(
        len(vocab)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_equiv = -1.0
    best_hard = -1.0
    best_epoch = -1
    stale = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        count = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            log_probs = torch.log_softmax(
                logits,
                dim=1,
            )

            loss = -(
                yb * log_probs
            ).sum(
                dim=1
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * xb.shape[0]
            )

            count += xb.shape[0]

        metrics = evaluate(
            model,
            val_x,
            val,
            vocab,
        )

        train_loss = (
            total_loss / count
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_hard_accuracy": metrics["hard"],
                "validation_oracle_equivalent_rate": metrics["equiv"],
            }
        )

        improved = (
            metrics["equiv"]
            > best_equiv + 1e-12
            or (
                abs(
                    metrics["equiv"]
                    - best_equiv
                ) <= 1e-12
                and metrics["hard"]
                > best_hard + 1e-12
            )
        )

        if improved:
            best_equiv = metrics["equiv"]
            best_hard = metrics["hard"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "vocabulary": vocab,
                    "input_dim": 67,
                    "descriptor_dim": 64,
                    "protocol_dim": 3,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_hard_accuracy":
                        best_hard,
                    "validation_oracle_equivalent_rate":
                        best_equiv,
                },
                args.output_dir
                / "best_model.pt",
            )

        else:
            stale += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            print(
                f"epoch={epoch:03d} "
                f"loss={train_loss:.5f} "
                f"hard={100*metrics['hard']:.2f}% "
                f"equiv={100*metrics['equiv']:.2f}%"
            )

        if stale >= args.patience:
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    pd.DataFrame(history).to_csv(
        args.output_dir / "history.csv",
        index=False,
    )

    np.savez(
        args.output_dir / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_names=np.asarray(
            [
                *ZERO_FEATURE_NAMES,
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

    final = evaluate(
        model,
        val_x,
        val,
        vocab,
    )

    summary = {
        "schema_version": 1,
        "method": "Zero-SML",
        "seed": args.seed,
        "descriptor_dimension": 64,
        "protocol_dimension": 3,
        "model_input_dimension": 67,
        "output_class_count": len(vocab),
        "best_epoch": best_epoch,
        "validation_hard_accuracy":
            final["hard"],
        "validation_oracle_equivalent_rate":
            final["equiv"],
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
    print("ZERO-SML RESULT")
    print("=" * 70)
    print("Best epoch:", best_epoch)

    print(
        "Validation hard accuracy:",
        f"{100*final['hard']:.2f}%",
    )

    print(
        "Validation oracle-equivalent rate:",
        f"{100*final['equiv']:.2f}%",
    )

    print()
    print("ZERO-SML TRAINING: PASS")


if __name__ == "__main__":
    main()
