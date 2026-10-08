from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from sml_hpo.episodes.sampler import Episode, EpisodeSampler
from sml_hpo.models.protonet import ProtoNet


def _move_episode(
    episode: Episode,
    device: torch.device,
) -> Episode:
    return Episode(
        support_images=episode.support_images.to(
            device,
            non_blocking=True,
        ),
        support_labels=episode.support_labels.to(
            device,
            non_blocking=True,
        ),
        query_images=episode.query_images.to(
            device,
            non_blocking=True,
        ),
        query_labels=episode.query_labels.to(
            device,
            non_blocking=True,
        ),
    )


def train_episode(
    model: ProtoNet,
    optimizer: torch.optim.Optimizer,
    episode: Episode,
    n_way: int,
    device: torch.device,
    gradient_clip_norm: float | None = 5.0,
) -> tuple[float, float]:
    model.train()

    episode = _move_episode(episode, device)

    optimizer.zero_grad(set_to_none=True)

    logits = model.episode_logits(
        support_images=episode.support_images,
        support_labels=episode.support_labels,
        query_images=episode.query_images,
        n_way=n_way,
    )

    loss = nn.functional.cross_entropy(
        logits,
        episode.query_labels,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Encountered non-finite training loss: {loss.item()}"
        )

    loss.backward()

    if gradient_clip_norm is not None:
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

    optimizer.step()

    accuracy = (
        logits.argmax(dim=1) == episode.query_labels
    ).float().mean()

    return float(loss.item()), float(accuracy.item())


@torch.no_grad()
def evaluate_episode(
    model: ProtoNet,
    episode: Episode,
    n_way: int,
    device: torch.device,
) -> float:
    model.eval()

    episode = _move_episode(episode, device)

    logits = model.episode_logits(
        support_images=episode.support_images,
        support_labels=episode.support_labels,
        query_images=episode.query_images,
        n_way=n_way,
    )

    accuracy = (
        logits.argmax(dim=1) == episode.query_labels
    ).float().mean()

    return float(accuracy.item())


def evaluate_episode_bank(
    model: ProtoNet,
    sampler: EpisodeSampler,
    episode_seeds: Iterable[int],
    n_way: int,
    device: torch.device,
) -> dict[str, float]:
    accuracies = [
        evaluate_episode(
            model=model,
            episode=sampler.sample(seed),
            n_way=n_way,
            device=device,
        )
        for seed in episode_seeds
    ]

    accuracy_array = np.asarray(accuracies, dtype=np.float64)

    if len(accuracy_array) == 0:
        raise ValueError("episode_seeds must not be empty")

    return {
        "mean_accuracy": float(accuracy_array.mean()),
        "std_accuracy": float(accuracy_array.std(ddof=1))
        if len(accuracy_array) > 1
        else 0.0,
        "episodes": int(len(accuracy_array)),
    }
