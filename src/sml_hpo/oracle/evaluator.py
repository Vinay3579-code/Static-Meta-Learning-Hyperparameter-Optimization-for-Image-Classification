from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.models.protonet import ProtoNet
from sml_hpo.oracle.search_space import AnchorConfig
from sml_hpo.tasks.spec import TaskSpec
from sml_hpo.training.engine import (
    evaluate_episode_bank,
    train_episode,
)
from sml_hpo.utils.seed import seed_everything


def build_optimizer(
    model: ProtoNet,
    config: AnchorConfig,
) -> torch.optim.Optimizer:
    parameters = model.parameters()

    if config.optimizer == "Adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "SGD":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {config.optimizer}"
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: AnchorConfig,
    train_episodes: int,
):
    if config.scheduler == "none":
        return None

    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_episodes),
            eta_min=config.learning_rate * 0.01,
        )

    raise ValueError(
        f"Unsupported scheduler: {config.scheduler}"
    )


def evaluate_anchor_config(
    *,
    task: TaskSpec,
    sampler: EpisodeSampler,
    config: AnchorConfig,
    device: torch.device,
    model_seeds: Sequence[int],
    train_episodes: int,
    validation_episodes: int,
) -> dict[str, Any]:
    if not model_seeds:
        raise ValueError(
            "model_seeds must not be empty"
        )

    if train_episodes <= 0:
        raise ValueError(
            "train_episodes must be positive"
        )

    if validation_episodes <= 0:
        raise ValueError(
            "validation_episodes must be positive"
        )

    training_seed_bank = task.training_seeds()
    validation_seed_bank = task.validation_seeds()

    if train_episodes > len(training_seed_bank):
        raise ValueError(
            f"Requested {train_episodes} training episodes, "
            f"but task contains {len(training_seed_bank)}"
        )

    if validation_episodes > len(
        validation_seed_bank
    ):
        raise ValueError(
            f"Requested {validation_episodes} validation "
            f"episodes, but task contains "
            f"{len(validation_seed_bank)}"
        )

    training_seed_bank = training_seed_bank[
        :train_episodes
    ]

    validation_seed_bank = validation_seed_bank[
        :validation_episodes
    ]

    per_seed_results: list[dict[str, float]] = []

    configuration_start = time.perf_counter()

    for model_seed in model_seeds:
        seed_everything(
            int(model_seed),
            deterministic=True,
        )

        model = ProtoNet(
            input_channels=3,
            hidden_channels=64,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        ).to(device)

        optimizer = build_optimizer(
            model,
            config,
        )

        scheduler = build_scheduler(
            optimizer,
            config,
            train_episodes=train_episodes,
        )

        train_losses: list[float] = []
        train_accuracies: list[float] = []

        seed_start = time.perf_counter()

        for episode_seed in training_seed_bank:
            loss, accuracy = train_episode(
                model=model,
                optimizer=optimizer,
                episode=sampler.sample(
                    episode_seed
                ),
                n_way=task.n_way,
                device=device,
                gradient_clip_norm=5.0,
            )

            train_losses.append(loss)
            train_accuracies.append(accuracy)

            if scheduler is not None:
                scheduler.step()

        validation_result = evaluate_episode_bank(
            model=model,
            sampler=sampler,
            episode_seeds=validation_seed_bank,
            n_way=task.n_way,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        seed_elapsed = (
            time.perf_counter() - seed_start
        )

        per_seed_results.append(
            {
                "model_seed": float(model_seed),
                "final_train_loss": float(
                    train_losses[-1]
                ),
                "mean_train_loss": float(
                    np.mean(train_losses)
                ),
                "final_train_accuracy": float(
                    train_accuracies[-1]
                ),
                "mean_train_accuracy": float(
                    np.mean(train_accuracies)
                ),
                "validation_accuracy": float(
                    validation_result[
                        "mean_accuracy"
                    ]
                ),
                "validation_episode_std": float(
                    validation_result[
                        "std_accuracy"
                    ]
                ),
                "elapsed_seconds": float(
                    seed_elapsed
                ),
            }
        )

        del model
        del optimizer
        del scheduler

        if device.type == "cuda":
            torch.cuda.empty_cache()

    validation_accuracies = np.asarray(
        [
            result["validation_accuracy"]
            for result in per_seed_results
        ],
        dtype=np.float64,
    )

    final_train_losses = np.asarray(
        [
            result["final_train_loss"]
            for result in per_seed_results
        ],
        dtype=np.float64,
    )

    mean_train_accuracies = np.asarray(
        [
            result["mean_train_accuracy"]
            for result in per_seed_results
        ],
        dtype=np.float64,
    )

    episode_stds = np.asarray(
        [
            result["validation_episode_std"]
            for result in per_seed_results
        ],
        dtype=np.float64,
    )

    elapsed_values = np.asarray(
        [
            result["elapsed_seconds"]
            for result in per_seed_results
        ],
        dtype=np.float64,
    )

    total_elapsed = (
        time.perf_counter()
        - configuration_start
    )

    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "split": task.split,
        "config_id": config.config_id,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer,
        "embedding_dim": config.embedding_dim,
        "dropout": config.dropout,
        "scheduler": config.scheduler,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "model_seed_count": len(model_seeds),
        "model_seeds": ",".join(
            str(seed)
            for seed in model_seeds
        ),
        "model_seeds_json": json.dumps(
            [int(seed) for seed in model_seeds]
        ),
        "validation_accuracies_by_seed_json": json.dumps(
            [
                float(value)
                for value in validation_accuracies
            ]
        ),
        "validation_episode_stds_by_seed_json": json.dumps(
            [
                float(value)
                for value in episode_stds
            ]
        ),
        "seed_elapsed_seconds_json": json.dumps(
            [
                float(value)
                for value in elapsed_values
            ]
        ),
        "validation_accuracy_mean": float(
            validation_accuracies.mean()
        ),
        "validation_accuracy_seed_std": float(
            validation_accuracies.std(ddof=1)
            if len(validation_accuracies) > 1
            else 0.0
        ),
        "validation_episode_std_mean": float(
            episode_stds.mean()
        ),
        "final_train_loss_mean": float(
            final_train_losses.mean()
        ),
        "mean_train_accuracy_mean": float(
            mean_train_accuracies.mean()
        ),
        "mean_seed_elapsed_seconds": float(
            elapsed_values.mean()
        ),
        "total_elapsed_seconds": float(
            total_elapsed
        ),
    }
