from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.models.protonet import ProtoNet
from sml_hpo.oracle.search_space_v2 import (
    AnchorConfigV2,
)
from sml_hpo.tasks.spec import TaskSpec
from sml_hpo.training.engine import (
    evaluate_episode_bank,
)
from sml_hpo.utils.seed import seed_everything


def build_optimizer_v2(
    model: ProtoNet,
    config: AnchorConfigV2,
) -> torch.optim.Optimizer:
    if config.optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.weight_decay,
        )

    raise ValueError(config.optimizer)


def build_scheduler_v2(
    optimizer: torch.optim.Optimizer,
    config: AnchorConfigV2,
    train_episodes: int,
):
    if config.scheduler == "none":
        return None

    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_episodes),
            eta_min=(
                config.learning_rate * 0.01
            ),
        )

    raise ValueError(config.scheduler)


def train_episode_v2(
    *,
    model: ProtoNet,
    optimizer: torch.optim.Optimizer,
    sampler: EpisodeSampler,
    episode_seed: int,
    task: TaskSpec,
    config: AnchorConfigV2,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    episode = sampler.sample(
        episode_seed
    )

    support_images = (
        episode.support_images.to(
            device,
            non_blocking=True,
        )
    )

    support_labels = (
        episode.support_labels.to(
            device,
            non_blocking=True,
        )
    )

    query_images = (
        episode.query_images.to(
            device,
            non_blocking=True,
        )
    )

    query_labels = (
        episode.query_labels.to(
            device,
            non_blocking=True,
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model.episode_logits(
        support_images=support_images,
        support_labels=support_labels,
        query_images=query_images,
        n_way=task.n_way,
    )

    loss = F.cross_entropy(
        logits,
        query_labels,
        label_smoothing=(
            config.label_smoothing
        ),
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss for "
            f"{config.config_id}"
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

    accuracy = (
        logits.argmax(dim=1)
        == query_labels
    ).float().mean()

    return (
        float(loss.detach().item()),
        float(accuracy.detach().item()),
    )


def evaluate_anchor_config_v2(
    *,
    task: TaskSpec,
    sampler: EpisodeSampler,
    config: AnchorConfigV2,
    device: torch.device,
    model_seeds: Sequence[int],
    train_episodes: int,
    validation_episodes: int,
) -> dict[str, Any]:
    if not model_seeds:
        raise ValueError(
            "model_seeds must not be empty"
        )

    train_seed_bank = task.training_seeds()[
        :train_episodes
    ]

    validation_seed_bank = (
        task.validation_seeds()[
            :validation_episodes
        ]
    )

    if len(train_seed_bank) != train_episodes:
        raise ValueError(
            "Insufficient training seeds"
        )

    if len(
        validation_seed_bank
    ) != validation_episodes:
        raise ValueError(
            "Insufficient validation seeds"
        )

    per_seed_results = []
    configuration_start = time.perf_counter()

    for model_seed in model_seeds:
        seed_everything(
            int(model_seed),
            deterministic=True,
        )

        model = ProtoNet(
            input_channels=3,
            hidden_channels=(
                config.hidden_channels
            ),
            embedding_dim=(
                config.embedding_dim
            ),
            dropout=config.dropout,
            distance_metric=(
                config.distance_metric
            ),
            temperature=(
                config.temperature
            ),
        ).to(device)

        optimizer = build_optimizer_v2(
            model,
            config,
        )

        scheduler = build_scheduler_v2(
            optimizer,
            config,
            train_episodes,
        )

        train_losses = []
        train_accuracies = []

        seed_start = time.perf_counter()

        for episode_seed in train_seed_bank:
            loss, accuracy = train_episode_v2(
                model=model,
                optimizer=optimizer,
                sampler=sampler,
                episode_seed=episode_seed,
                task=task,
                config=config,
                device=device,
            )

            train_losses.append(loss)
            train_accuracies.append(
                accuracy
            )

            if scheduler is not None:
                scheduler.step()

        validation_result = (
            evaluate_episode_bank(
                model=model,
                sampler=sampler,
                episode_seeds=(
                    validation_seed_bank
                ),
                n_way=task.n_way,
                device=device,
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        per_seed_results.append(
            {
                "model_seed": int(
                    model_seed
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
                "final_train_loss": float(
                    train_losses[-1]
                ),
                "mean_train_accuracy": float(
                    np.mean(
                        train_accuracies
                    )
                ),
                "elapsed_seconds": float(
                    time.perf_counter()
                    - seed_start
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
            row["validation_accuracy"]
            for row in per_seed_results
        ],
        dtype=np.float64,
    )

    validation_episode_stds = np.asarray(
        [
            row["validation_episode_std"]
            for row in per_seed_results
        ],
        dtype=np.float64,
    )

    final_losses = np.asarray(
        [
            row["final_train_loss"]
            for row in per_seed_results
        ],
        dtype=np.float64,
    )

    mean_train_accuracies = np.asarray(
        [
            row["mean_train_accuracy"]
            for row in per_seed_results
        ],
        dtype=np.float64,
    )

    elapsed_values = np.asarray(
        [
            row["elapsed_seconds"]
            for row in per_seed_results
        ],
        dtype=np.float64,
    )

    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "split": task.split,
        "config_id": config.config_id,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer,
        "hidden_channels": (
            config.hidden_channels
        ),
        "embedding_dim": (
            config.embedding_dim
        ),
        "dropout": config.dropout,
        "scheduler": config.scheduler,
        "distance_metric": (
            config.distance_metric
        ),
        "temperature": (
            config.temperature
        ),
        "label_smoothing": (
            config.label_smoothing
        ),
        "train_episodes": train_episodes,
        "validation_episodes": (
            validation_episodes
        ),
        "model_seed_count": len(
            model_seeds
        ),
        "model_seeds": ",".join(
            str(seed)
            for seed in model_seeds
        ),
        "model_seeds_json": json.dumps(
            [
                int(seed)
                for seed in model_seeds
            ]
        ),
        "validation_accuracies_by_seed_json": (
            json.dumps(
                validation_accuracies.tolist()
            )
        ),
        "validation_episode_stds_by_seed_json": (
            json.dumps(
                validation_episode_stds.tolist()
            )
        ),
        "seed_elapsed_seconds_json": (
            json.dumps(
                elapsed_values.tolist()
            )
        ),
        "validation_accuracy_mean": float(
            validation_accuracies.mean()
        ),
        "validation_accuracy_seed_std": float(
            validation_accuracies.std(
                ddof=1
            )
            if len(
                validation_accuracies
            ) > 1
            else 0.0
        ),
        "validation_episode_std_mean": float(
            validation_episode_stds.mean()
        ),
        "final_train_loss_mean": float(
            final_losses.mean()
        ),
        "mean_train_accuracy_mean": float(
            mean_train_accuracies.mean()
        ),
        "mean_seed_elapsed_seconds": float(
            elapsed_values.mean()
        ),
        "total_elapsed_seconds": float(
            time.perf_counter()
            - configuration_start
        ),
    }
