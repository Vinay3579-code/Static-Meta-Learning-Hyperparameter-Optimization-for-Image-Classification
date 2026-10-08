from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import (
    StandardScaler,
)

from torch import nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)

from sml_hpo.descriptors.probe import (
    PROBE_FEATURE_NAMES,
)


class MetaMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(

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

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def parse_equiv(value):
    result = json.loads(
        str(value)
    )

    return [
        str(x)
        for x in result
    ]


def make_x(
    table,
    feature_names,
):

    descriptors = table[
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

    x = np.concatenate(
        [
            descriptors,
            protocol,
        ],
        axis=1,
    )

    if not np.isfinite(
        x
    ).all():
        raise RuntimeError(
            "Non-finite final-fit input"
        )

    return x


def make_targets(
    table,
    vocabulary,
):

    index = {
        config_id: position
        for position, config_id
        in enumerate(vocabulary)
    }

    target = np.zeros(
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

        configs = parse_equiv(
            value
        )

        represented = [
            config
            for config in configs
            if config in index
        ]

        if not represented:
            raise RuntimeError(
                "No represented "
                "equivalent configuration"
            )

        weight = (
            1.0
            / len(represented)
        )

        for config in represented:
            target[
                row_index,
                index[config],
            ] = weight

    return target


def fit_variant(
    *,
    table,
    features,
    variant_name,
    epochs_by_seed,
    batch_size,
    learning_rate,
    weight_decay,
    device,
    output_root,
):

    output_dir = (
        output_root
        / variant_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary_set = set()

    for value in table[
        "oracle_equivalent_config_ids_json"
    ]:
        vocabulary_set.update(
            parse_equiv(value)
        )

    vocabulary = sorted(
        vocabulary_set
    )

    if len(vocabulary) != 19:
        raise RuntimeError(
            f"Expected 19 classes, "
            f"found {len(vocabulary)}"
        )

    x = make_x(
        table,
        features,
    )

    input_dim = (
        len(features) + 3
    )

    if x.shape != (
        680,
        input_dim,
    ):
        raise RuntimeError(
            f"Unexpected {variant_name} "
            f"shape {x.shape}"
        )

    scaler = StandardScaler()

    x = scaler.fit_transform(
        x
    )

    y = make_targets(
        table,
        vocabulary,
    )

    np.savez(
        output_dir
        / "scaler.npz",

        mean=scaler.mean_,
        scale=scaler.scale_,

        feature_names=np.asarray(
            [
                *features,
                "n_way",
                "n_shot",
                "n_query",
            ],
            dtype=object,
        ),
    )

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    run_metadata = {}

    for seed_string, epochs in (
        epochs_by_seed.items()
    ):

        seed = int(
            seed_string
        )

        epochs = int(
            epochs
        )

        set_seed(
            seed
        )

        generator = (
            torch.Generator()
            .manual_seed(seed)
        )

        loader = DataLoader(
            TensorDataset(
                x_tensor,
                y_tensor,
            ),

            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        model = MetaMLP(
            input_dim,
            len(vocabulary),
        ).to(device)

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        )

        print()
        print(
            f"{variant_name} "
            f"seed={seed} "
            f"epochs={epochs}"
        )

        final_loss = None

        for epoch in range(
            1,
            epochs + 1,
        ):

            model.train()

            total_loss = 0.0
            count = 0

            for xb, yb in loader:

                xb = xb.to(
                    device
                )

                yb = yb.to(
                    device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    xb
                )

                log_prob = (
                    torch.log_softmax(
                        logits,
                        dim=1,
                    )
                )

                loss = -(
                    yb
                    * log_prob
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
                    float(
                        loss.item()
                    )
                    * xb.shape[0]
                )

                count += (
                    xb.shape[0]
                )

            final_loss = (
                total_loss
                / count
            )

            print(
                f"  epoch "
                f"{epoch}/{epochs} "
                f"loss="
                f"{final_loss:.6f}"
            )

        model_path = (
            output_dir
            / f"model_seed{seed}.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "seed":
                    seed,

                "epochs":
                    epochs,

                "input_dim":
                    input_dim,

                "output_dim":
                    len(vocabulary),

                "vocabulary":
                    vocabulary,

                "variant":
                    variant_name,

                "training_tasks":
                    680,
            },
            model_path,
        )

        run_metadata[
            str(seed)
        ] = {
            "epochs": epochs,
            "final_training_loss":
                float(final_loss),
            "model_path":
                str(model_path),
        }

    metadata = {
        "variant":
            variant_name,

        "training_tasks":
            680,

        "descriptor_dimension":
            len(features),

        "protocol_dimension":
            3,

        "input_dimension":
            input_dim,

        "output_classes":
            len(vocabulary),

        "vocabulary":
            vocabulary,

        "runs":
            run_metadata,
    }

    (
        output_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    args = parser.parse_args()

    protocol = json.loads(
        args.protocol.read_text(
            encoding="utf-8"
        )
    )

    if (
        protocol["status"]
        != "FROZEN_BEFORE_TEST_ORACLE"
    ):
        raise RuntimeError(
            "Meta protocol is not frozen"
        )

    if protocol[
        "test_oracle_labels_seen"
    ]:
        raise RuntimeError(
            "Protocol says test "
            "labels were already seen"
        )

    table = pd.read_csv(
        args.data
    )

    if len(table) != 680:
        raise RuntimeError(
            f"Expected 680 tasks, "
            f"found {len(table)}"
        )

    if table[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate task IDs"
        )

    zero_features = list(
        ZERO_FEATURE_NAMES
    )

    probe_features = [
        *ZERO_FEATURE_NAMES,
        *PROBE_FEATURE_NAMES,
    ]

    if len(
        zero_features
    ) != 64:
        raise RuntimeError(
            "Zero descriptor dimension "
            "is not 64"
        )

    if len(
        probe_features
    ) != 84:
        raise RuntimeError(
            "Probe descriptor dimension "
            "is not 84"
        )

    device = torch.device(
        args.device
    )

    primary = protocol[
        "primary_method"
    ]

    probe = protocol[
        "probe_ablation"
    ]

    fit_variant(
        table=table,

        features=zero_features,

        variant_name="zero_sml",

        epochs_by_seed=(
            primary[
                "final_refit_epochs"
            ]
        ),

        batch_size=int(
            primary[
                "batch_size"
            ]
        ),

        learning_rate=float(
            primary[
                "learning_rate"
            ]
        ),

        weight_decay=float(
            primary[
                "weight_decay"
            ]
        ),

        device=device,

        output_root=(
            args.output_root
        ),
    )

    fit_variant(
        table=table,

        features=probe_features,

        variant_name="probe_sml",

        epochs_by_seed=(
            probe[
                "final_refit_epochs"
            ]
        ),

        batch_size=int(
            probe[
                "batch_size"
            ]
        ),

        learning_rate=float(
            probe[
                "learning_rate"
            ]
        ),

        weight_decay=float(
            probe[
                "weight_decay"
            ]
        ),

        device=device,

        output_root=(
            args.output_root
        ),
    )

    print()
    print(
        "FROZEN META ENSEMBLE "
        "REFIT: PASS"
    )


if __name__ == "__main__":
    main()
