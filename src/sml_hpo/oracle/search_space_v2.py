from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import qmc


VALID_OPTIMIZERS = (
    "Adam",
    "AdamW",
    "SGD",
)

VALID_HIDDEN_CHANNELS = (
    32,
    64,
    96,
    128,
)

VALID_EMBEDDING_DIMS = (
    32,
    64,
    128,
    256,
)

VALID_DROPOUTS = (
    0.0,
    0.1,
    0.3,
)

VALID_SCHEDULERS = (
    "none",
    "cosine",
)

VALID_DISTANCE_METRICS = (
    "euclidean",
    "cosine",
)

VALID_LABEL_SMOOTHING = (
    0.0,
    0.05,
    0.1,
)


@dataclass(frozen=True)
class AnchorConfigV2:
    schema_version: int
    config_id: str

    learning_rate: float
    weight_decay: float

    optimizer: str
    hidden_channels: int
    embedding_dim: int
    dropout: float
    scheduler: str

    distance_metric: str
    temperature: float
    label_smoothing: float

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(
                "V2 anchors require schema_version=2"
            )

        if not self.config_id:
            raise ValueError(
                "config_id must not be empty"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative"
            )

        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}"
            )

        if self.hidden_channels not in (
            VALID_HIDDEN_CHANNELS
        ):
            raise ValueError(
                "Invalid hidden channel count"
            )

        if self.embedding_dim not in (
            VALID_EMBEDDING_DIMS
        ):
            raise ValueError(
                "Invalid embedding dimension"
            )

        if self.dropout not in VALID_DROPOUTS:
            raise ValueError(
                "Invalid dropout"
            )

        if self.scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                "Invalid scheduler"
            )

        if self.distance_metric not in (
            VALID_DISTANCE_METRICS
        ):
            raise ValueError(
                "Invalid distance metric"
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        if self.label_smoothing not in (
            VALID_LABEL_SMOOTHING
        ):
            raise ValueError(
                "Invalid label smoothing"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "AnchorConfigV2":
        return cls(**value)


def _log_scale(
    value: float,
    low: float,
    high: float,
) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "Latin-hypercube value outside [0, 1]"
        )

    return float(
        10 ** (
            math.log10(low)
            + value
            * (
                math.log10(high)
                - math.log10(low)
            )
        )
    )


def _balanced_values(
    values: Sequence[Any],
    count: int,
    rng: np.random.Generator,
) -> list[Any]:
    repeated = [
        values[index % len(values)]
        for index in range(count)
    ]

    permutation = rng.permutation(count)

    return [
        repeated[int(index)]
        for index in permutation
    ]


def generate_anchor_configs_v2(
    *,
    count: int = 64,
    seed: int = 20260804,
) -> list[AnchorConfigV2]:
    if count <= 0:
        raise ValueError(
            "count must be positive"
        )

    rng = np.random.default_rng(seed)

    continuous = qmc.LatinHypercube(
        d=3,
        seed=seed,
    ).random(n=count)

    optimizers = _balanced_values(
        VALID_OPTIMIZERS,
        count,
        rng,
    )

    hidden_channels = _balanced_values(
        VALID_HIDDEN_CHANNELS,
        count,
        rng,
    )

    embedding_dims = _balanced_values(
        VALID_EMBEDDING_DIMS,
        count,
        rng,
    )

    dropouts = _balanced_values(
        VALID_DROPOUTS,
        count,
        rng,
    )

    schedulers = _balanced_values(
        VALID_SCHEDULERS,
        count,
        rng,
    )

    distance_metrics = _balanced_values(
        VALID_DISTANCE_METRICS,
        count,
        rng,
    )

    label_smoothing_values = _balanced_values(
        VALID_LABEL_SMOOTHING,
        count,
        rng,
    )

    configs: list[AnchorConfigV2] = []

    for index in range(count):
        optimizer = str(
            optimizers[index]
        )

        distance_metric = str(
            distance_metrics[index]
        )

        if optimizer == "SGD":
            learning_rate = _log_scale(
                float(continuous[index, 0]),
                1e-3,
                2e-1,
            )
        else:
            learning_rate = _log_scale(
                float(continuous[index, 0]),
                1e-5,
                5e-3,
            )

        weight_decay = _log_scale(
            float(continuous[index, 1]),
            1e-7,
            1e-2,
        )

        if distance_metric == "cosine":
            temperature = _log_scale(
                float(continuous[index, 2]),
                0.05,
                1.0,
            )
        else:
            temperature = _log_scale(
                float(continuous[index, 2]),
                0.25,
                4.0,
            )

        configs.append(
            AnchorConfigV2(
                schema_version=2,
                config_id=f"anchor_v2_{index:03d}",
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer=optimizer,
                hidden_channels=int(
                    hidden_channels[index]
                ),
                embedding_dim=int(
                    embedding_dims[index]
                ),
                dropout=float(
                    dropouts[index]
                ),
                scheduler=str(
                    schedulers[index]
                ),
                distance_metric=distance_metric,
                temperature=temperature,
                label_smoothing=float(
                    label_smoothing_values[index]
                ),
            )
        )

    signatures = [
        tuple(config.to_dict().values())
        for config in configs
    ]

    if len(signatures) != len(set(signatures)):
        raise RuntimeError(
            "Duplicate V2 anchor configurations"
        )

    return configs


def save_anchor_manifest_v2(
    configs: Sequence[AnchorConfigV2],
    path: str | Path,
    *,
    generation_seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 2,
        "method": "mixed_latin_hypercube_v2",
        "generation_seed": int(
            generation_seed
        ),
        "configuration_count": len(configs),
        "continuous_space": {
            "learning_rate": {
                "Adam": [1e-5, 5e-3],
                "AdamW": [1e-5, 5e-3],
                "SGD": [1e-3, 2e-1],
                "scale": "log10",
            },
            "weight_decay": {
                "range": [1e-7, 1e-2],
                "scale": "log10",
            },
            "temperature": {
                "euclidean": [0.25, 4.0],
                "cosine": [0.05, 1.0],
                "scale": "log10",
            },
        },
        "categorical_space": {
            "optimizer": list(
                VALID_OPTIMIZERS
            ),
            "hidden_channels": list(
                VALID_HIDDEN_CHANNELS
            ),
            "embedding_dim": list(
                VALID_EMBEDDING_DIMS
            ),
            "dropout": list(
                VALID_DROPOUTS
            ),
            "scheduler": list(
                VALID_SCHEDULERS
            ),
            "distance_metric": list(
                VALID_DISTANCE_METRICS
            ),
            "label_smoothing": list(
                VALID_LABEL_SMOOTHING
            ),
        },
        "fixed_hyperparameters": {
            "input_channels": 3,
            "encoder_depth": 4,
            "normalization": "GroupNorm",
            "gradient_clip_norm": 5.0,
            "sgd_momentum": 0.9,
            "cosine_eta_min_ratio": 0.01,
        },
        "configurations": [
            config.to_dict()
            for config in configs
        ],
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_anchor_manifest_v2(
    path: str | Path,
) -> list[AnchorConfigV2]:
    path = Path(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("schema_version") != 2:
        raise ValueError(
            "Expected V2 anchor manifest"
        )

    configs = [
        AnchorConfigV2.from_dict(item)
        for item in payload["configurations"]
    ]

    if payload.get(
        "configuration_count"
    ) != len(configs):
        raise ValueError(
            "configuration_count mismatch"
        )

    return configs
