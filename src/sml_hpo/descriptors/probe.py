from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from sml_hpo.episodes.sampler import Episode, EpisodeSampler
from sml_hpo.models.protonet import ProtoNet
from sml_hpo.tasks.spec import TaskSpec
from sml_hpo.utils.seed import seed_everything


EPSILON = 1e-12


PROBE_FEATURE_NAMES = (
    "probe_train_loss_initial",
    "probe_train_loss_final",
    "probe_train_loss_mean",
    "probe_train_loss_std",
    "probe_train_loss_slope",
    "probe_train_accuracy_initial",
    "probe_train_accuracy_final",
    "probe_train_accuracy_mean",
    "probe_validation_loss",
    "probe_validation_accuracy",
    "probe_gradient_norm_mean",
    "probe_gradient_norm_std",
    "probe_gradient_norm_max",
    "probe_gradient_variance_mean",
    "probe_update_norm_mean",
    "probe_update_parameter_ratio_mean",
    "probe_gradient_cosine_mean",
    "probe_gradient_cosine_std",
    "probe_directional_curvature",
    "probe_gradient_hessian_alignment",
)

assert len(PROBE_FEATURE_NAMES) == 20


@dataclass(frozen=True)
class ProbeDescriptorResult:
    descriptor: np.ndarray
    elapsed_seconds: float
    probe_steps: int
    validation_episodes: int


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


def _episode_loss_accuracy(
    model: ProtoNet,
    episode: Episode,
    n_way: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    accuracy = (
        logits.argmax(dim=1)
        == episode.query_labels
    ).float().mean()

    return loss, accuracy


def _gradient_vector(
    parameters: list[nn.Parameter],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []

    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(
                torch.zeros_like(parameter).reshape(-1)
            )
        else:
            pieces.append(
                parameter.grad.detach().reshape(-1)
            )

    return torch.cat(pieces)


def _safe_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    denominator = (
        first.norm()
        * second.norm()
    )

    if float(denominator.item()) <= EPSILON:
        return 0.0

    value = torch.dot(first, second) / denominator

    return float(
        torch.clamp(value, -1.0, 1.0).item()
    )


@torch.no_grad()
def evaluate_probe_bank(
    *,
    model: ProtoNet,
    sampler: EpisodeSampler,
    episode_seeds: tuple[int, ...],
    n_way: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    losses: list[float] = []
    accuracies: list[float] = []

    for episode_seed in episode_seeds:
        episode = _move_episode(
            sampler.sample(episode_seed),
            device,
        )

        loss, accuracy = _episode_loss_accuracy(
            model=model,
            episode=episode,
            n_way=n_way,
        )

        losses.append(float(loss.item()))
        accuracies.append(float(accuracy.item()))

    if not losses:
        raise ValueError(
            "Probe validation episode bank is empty"
        )

    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
    )


def _loss_and_gradient_vector(
    *,
    model: ProtoNet,
    episode: Episode,
    n_way: int,
    parameters: list[nn.Parameter],
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)

    loss, _ = _episode_loss_accuracy(
        model=model,
        episode=episode,
        n_way=n_way,
    )

    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )

    gradient_parts: list[torch.Tensor] = []

    for parameter, gradient in zip(
        parameters,
        gradients,
        strict=True,
    ):
        if gradient is None:
            gradient_parts.append(
                torch.zeros_like(parameter).reshape(-1)
            )
        else:
            gradient_parts.append(
                gradient.detach().reshape(-1)
            )

    return (
        float(loss.detach().item()),
        torch.cat(gradient_parts),
    )


def finite_difference_curvature(
    *,
    model: ProtoNet,
    episode: Episode,
    n_way: int,
    relative_epsilon: float = 1e-3,
) -> tuple[float, float]:
    """
    Approximate curvature in the final-gradient direction.

    The finite-difference approximation avoids relying on second-order
    backward support for every operation in the convolutional encoder.

    Returns:
        directional curvature
        cosine alignment between gradient and Hessian-gradient direction
    """
    if relative_epsilon <= 0:
        raise ValueError(
            "relative_epsilon must be positive"
        )

    model.eval()

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    original_vector = parameters_to_vector(
        parameters
    ).detach().clone()

    base_loss, base_gradient = _loss_and_gradient_vector(
        model=model,
        episode=episode,
        n_way=n_way,
        parameters=parameters,
    )

    gradient_norm = float(
        base_gradient.norm().item()
    )

    if gradient_norm <= EPSILON:
        return 0.0, 0.0

    direction = base_gradient / gradient_norm

    parameter_norm = float(
        original_vector.norm().item()
    )

    perturbation_size = (
        relative_epsilon
        * max(parameter_norm, 1.0)
    )

    try:
        with torch.no_grad():
            vector_to_parameters(
                original_vector
                + perturbation_size * direction,
                parameters,
            )

        plus_loss, plus_gradient = (
            _loss_and_gradient_vector(
                model=model,
                episode=episode,
                n_way=n_way,
                parameters=parameters,
            )
        )

        with torch.no_grad():
            vector_to_parameters(
                original_vector
                - perturbation_size * direction,
                parameters,
            )

        minus_loss, minus_gradient = (
            _loss_and_gradient_vector(
                model=model,
                episode=episode,
                n_way=n_way,
                parameters=parameters,
            )
        )

    finally:
        with torch.no_grad():
            vector_to_parameters(
                original_vector,
                parameters,
            )

    directional_curvature = (
        plus_loss
        - 2.0 * base_loss
        + minus_loss
    ) / (
        perturbation_size ** 2
    )

    hessian_direction = (
        plus_gradient
        - minus_gradient
    ) / (
        2.0 * perturbation_size
    )

    alignment = _safe_cosine(
        base_gradient,
        hessian_direction,
    )

    if not math.isfinite(directional_curvature):
        directional_curvature = 0.0

    if not math.isfinite(alignment):
        alignment = 0.0

    return (
        float(directional_curvature),
        float(alignment),
    )


def _linear_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    x_values = np.arange(
        len(values),
        dtype=np.float64,
    )

    y_values = np.asarray(
        values,
        dtype=np.float64,
    )

    slope, _ = np.polyfit(
        x_values,
        y_values,
        deg=1,
    )

    return float(slope)


def extract_probe_descriptor(
    *,
    task: TaskSpec,
    sampler: EpisodeSampler,
    device: torch.device,
    probe_steps: int = 20,
    validation_episodes: int = 5,
    model_seed: int = 20260802,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    curvature_relative_epsilon: float = 1e-3,
) -> ProbeDescriptorResult:
    """
    Run a fixed-budget episodic probe and return 20 optimization signals.

    The probe model, optimizer and update budget are fixed across all
    datasets and tasks. They are not tuned per target task.
    """
    if probe_steps <= 0:
        raise ValueError(
            "probe_steps must be positive"
        )

    if validation_episodes <= 0:
        raise ValueError(
            "validation_episodes must be positive"
        )

    training_seeds = task.training_seeds()
    validation_seeds = task.validation_seeds()

    if probe_steps > len(training_seeds):
        raise ValueError(
            f"Task {task.task_id} contains only "
            f"{len(training_seeds)} training seeds"
        )

    if validation_episodes > len(validation_seeds):
        raise ValueError(
            f"Task {task.task_id} contains only "
            f"{len(validation_seeds)} validation seeds"
        )

    # Reset to the same initialization for every task.
    seed_everything(
        model_seed,
        deterministic=True,
    )

    model = ProtoNet(
        input_channels=3,
        hidden_channels=64,
        embedding_dim=64,
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    train_losses: list[float] = []
    train_accuracies: list[float] = []

    gradient_norms: list[float] = []
    gradient_variances: list[float] = []

    update_norms: list[float] = []
    update_parameter_ratios: list[float] = []

    gradient_cosines: list[float] = []
    previous_gradient: torch.Tensor | None = None

    start_time = time.perf_counter()

    for episode_seed in training_seeds[:probe_steps]:
        model.train()

        episode = _move_episode(
            sampler.sample(episode_seed),
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        loss, accuracy = _episode_loss_accuracy(
            model=model,
            episode=episode,
            n_way=task.n_way,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite probe loss for "
                f"{task.task_id}: {loss.item()}"
            )

        loss.backward()

        gradient = _gradient_vector(parameters)

        gradient_norm = float(
            gradient.norm().item()
        )

        gradient_variance = float(
            gradient.var(
                unbiased=False
            ).item()
        )

        if previous_gradient is not None:
            gradient_cosines.append(
                _safe_cosine(
                    previous_gradient,
                    gradient,
                )
            )

        previous_gradient = (
            gradient.detach().clone()
        )

        parameter_vector_before = (
            parameters_to_vector(parameters)
            .detach()
            .clone()
        )

        parameter_norm_before = float(
            parameter_vector_before.norm().item()
        )

        nn.utils.clip_grad_norm_(
            parameters,
            max_norm=5.0,
        )

        optimizer.step()

        parameter_vector_after = (
            parameters_to_vector(parameters)
            .detach()
        )

        update_norm = float(
            (
                parameter_vector_after
                - parameter_vector_before
            ).norm().item()
        )

        update_ratio = (
            update_norm
            / (
                parameter_norm_before
                + EPSILON
            )
        )

        train_losses.append(
            float(loss.detach().item())
        )

        train_accuracies.append(
            float(accuracy.detach().item())
        )

        gradient_norms.append(gradient_norm)
        gradient_variances.append(
            gradient_variance
        )

        update_norms.append(update_norm)
        update_parameter_ratios.append(
            update_ratio
        )

    validation_seed_bank = tuple(
        validation_seeds[:validation_episodes]
    )

    validation_loss, validation_accuracy = (
        evaluate_probe_bank(
            model=model,
            sampler=sampler,
            episode_seeds=validation_seed_bank,
            n_way=task.n_way,
            device=device,
        )
    )

    curvature_episode = _move_episode(
        sampler.sample(
            validation_seed_bank[0]
        ),
        device,
    )

    directional_curvature, alignment = (
        finite_difference_curvature(
            model=model,
            episode=curvature_episode,
            n_way=task.n_way,
            relative_epsilon=(
                curvature_relative_epsilon
            ),
        )
    )

    cosine_mean = (
        float(np.mean(gradient_cosines))
        if gradient_cosines
        else 0.0
    )

    cosine_std = (
        float(np.std(gradient_cosines))
        if gradient_cosines
        else 0.0
    )

    descriptor = np.asarray(
        [
            train_losses[0],
            train_losses[-1],
            float(np.mean(train_losses)),
            float(np.std(train_losses)),
            _linear_slope(train_losses),
            train_accuracies[0],
            train_accuracies[-1],
            float(np.mean(train_accuracies)),
            validation_loss,
            validation_accuracy,
            float(np.mean(gradient_norms)),
            float(np.std(gradient_norms)),
            float(np.max(gradient_norms)),
            float(np.mean(gradient_variances)),
            float(np.mean(update_norms)),
            float(
                np.mean(
                    update_parameter_ratios
                )
            ),
            cosine_mean,
            cosine_std,
            directional_curvature,
            alignment,
        ],
        dtype=np.float64,
    )

    if descriptor.shape != (20,):
        raise RuntimeError(
            f"Probe descriptor has invalid shape: "
            f"{descriptor.shape}"
        )

    if not np.all(np.isfinite(descriptor)):
        invalid_indices = np.flatnonzero(
            ~np.isfinite(descriptor)
        )

        invalid_names = [
            PROBE_FEATURE_NAMES[index]
            for index in invalid_indices
        ]

        raise FloatingPointError(
            "Non-finite probe features: "
            f"{invalid_names}"
        )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    return ProbeDescriptorResult(
        descriptor=descriptor,
        elapsed_seconds=float(
            elapsed_seconds
        ),
        probe_steps=probe_steps,
        validation_episodes=(
            validation_episodes
        ),
    )
