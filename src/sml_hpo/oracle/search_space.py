from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import qmc


VALID_OPTIMIZERS = ("Adam", "AdamW", "SGD")
VALID_EMBEDDING_DIMS = (64, 128, 256)
VALID_DROPOUTS = (0.0, 0.1, 0.3)
VALID_SCHEDULERS = ("none", "cosine")


@dataclass(frozen=True)
class AnchorConfig:
    schema_version: int
    config_id: str

    learning_rate: float
    weight_decay: float
    optimizer: str
    embedding_dim: int
    dropout: float
    scheduler: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version}"
            )

        if not self.config_id:
            raise ValueError("config_id must not be empty")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {self.optimizer}"
            )

        if self.embedding_dim not in VALID_EMBEDDING_DIMS:
            raise ValueError(
                f"Unsupported embedding dimension: "
                f"{self.embedding_dim}"
            )

        if self.dropout not in VALID_DROPOUTS:
            raise ValueError(
                f"Unsupported dropout: {self.dropout}"
            )

        if self.scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {self.scheduler}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "AnchorConfig":
        return cls(**value)


def _log_scale(
    unit_value: float,
    low: float,
    high: float,
) -> float:
    if not 0.0 <= unit_value <= 1.0:
        raise ValueError("unit_value must be in [0, 1]")

    if low <= 0 or high <= low:
        raise ValueError("Invalid log-scale bounds")

    log_low = math.log10(low)
    log_high = math.log10(high)

    return float(
        10 ** (
            log_low
            + unit_value * (log_high - log_low)
        )
    )


def _balanced_values(
    values: Sequence[Any],
    count: int,
    rng: np.random.Generator,
) -> list[Any]:
    if not values:
        raise ValueError("values must not be empty")

    repeated = [
        values[index % len(values)]
        for index in range(count)
    ]

    permutation = rng.permutation(count)

    return [
        repeated[int(index)]
        for index in permutation
    ]


def generate_anchor_configs(
    *,
    count: int = 40,
    seed: int = 20260803,
) -> list[AnchorConfig]:
    """
    Generate a reproducible mixed search space.

    Continuous dimensions:
        - optimizer-conditional learning rate
        - weight decay

    Categorical dimensions:
        - optimizer
        - embedding dimension
        - dropout
        - scheduler

    Adam/AdamW and SGD use different learning-rate ranges because using one
    range for all optimizers would systematically disadvantage SGD.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    rng = np.random.default_rng(seed)

    continuous_sampler = qmc.LatinHypercube(
        d=2,
        seed=seed,
    )

    continuous_samples = continuous_sampler.random(
        n=count
    )

    optimizers = _balanced_values(
        VALID_OPTIMIZERS,
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

    configs: list[AnchorConfig] = []

    for index in range(count):
        optimizer = str(optimizers[index])

        learning_rate_unit = float(
            continuous_samples[index, 0]
        )

        weight_decay_unit = float(
            continuous_samples[index, 1]
        )

        if optimizer == "SGD":
            learning_rate = _log_scale(
                learning_rate_unit,
                low=1e-3,
                high=1e-1,
            )
        else:
            learning_rate = _log_scale(
                learning_rate_unit,
                low=1e-5,
                high=3e-3,
            )

        weight_decay = _log_scale(
            weight_decay_unit,
            low=1e-6,
            high=1e-2,
        )

        configs.append(
            AnchorConfig(
                schema_version=1,
                config_id=f"anchor_{index:03d}",
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer=optimizer,
                embedding_dim=int(
                    embedding_dims[index]
                ),
                dropout=float(dropouts[index]),
                scheduler=str(schedulers[index]),
            )
        )

    config_ids = [
        config.config_id
        for config in configs
    ]

    if len(config_ids) != len(set(config_ids)):
        raise RuntimeError("Duplicate configuration IDs")

    numerical_signatures = [
        (
            round(config.learning_rate, 14),
            round(config.weight_decay, 14),
            config.optimizer,
            config.embedding_dim,
            config.dropout,
            config.scheduler,
        )
        for config in configs
    ]

    if len(numerical_signatures) != len(
        set(numerical_signatures)
    ):
        raise RuntimeError(
            "Duplicate anchor configurations generated"
        )

    return configs


def save_anchor_manifest(
    configs: Sequence[AnchorConfig],
    path: str | Path,
    *,
    generation_seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "method": "mixed_latin_hypercube",
        "generation_seed": int(generation_seed),
        "configuration_count": len(configs),
        "continuous_space": {
            "learning_rate": {
                "Adam": [1e-5, 3e-3],
                "AdamW": [1e-5, 3e-3],
                "SGD": [1e-3, 1e-1],
                "scale": "log10",
            },
            "weight_decay": {
                "range": [1e-6, 1e-2],
                "scale": "log10",
            },
        },
        "categorical_space": {
            "optimizer": list(VALID_OPTIMIZERS),
            "embedding_dim": list(
                VALID_EMBEDDING_DIMS
            ),
            "dropout": list(VALID_DROPOUTS),
            "scheduler": list(VALID_SCHEDULERS),
        },
        "fixed_hyperparameters": {
            "input_channels": 3,
            "hidden_channels": 64,
            "sgd_momentum": 0.9,
            "gradient_clip_norm": 5.0,
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


def load_anchor_manifest(
    path: str | Path,
) -> list[AnchorConfig]:
    path = Path(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported anchor-manifest schema"
        )

    configs = [
        AnchorConfig.from_dict(item)
        for item in payload["configurations"]
    ]

    if payload.get(
        "configuration_count"
    ) != len(configs):
        raise ValueError(
            "configuration_count does not match "
            "the configuration list"
        )

    return configs
