from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Episode:
    support_images: torch.Tensor
    support_labels: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor


class EpisodeSampler:
    """
    Deterministic N-way K-shot episode sampler.

    Supplying the same episode seed generates the same classes and examples.
    """

    def __init__(
        self,
        dataset: Dataset,
        allowed_classes: Sequence[int],
        n_way: int,
        n_shot: int,
        n_query: int,
    ) -> None:
        if n_way <= 1:
            raise ValueError("n_way must be greater than 1")
        if n_shot <= 0:
            raise ValueError("n_shot must be positive")
        if n_query <= 0:
            raise ValueError("n_query must be positive")

        self.dataset = dataset
        self.n_way = int(n_way)
        self.n_shot = int(n_shot)
        self.n_query = int(n_query)

        allowed_set = {int(value) for value in allowed_classes}

        if len(allowed_set) < self.n_way:
            raise ValueError(
                f"Need at least {self.n_way} classes, "
                f"received {len(allowed_set)}"
            )

        targets = getattr(dataset, "targets", None)

        if targets is None:
            targets = [
                int(dataset[index][1])
                for index in range(len(dataset))
            ]

        class_to_indices: dict[int, list[int]] = defaultdict(list)

        for index, target in enumerate(targets):
            target = int(target)

            if target in allowed_set:
                class_to_indices[target].append(index)

        required_examples = self.n_shot + self.n_query

        invalid_classes = {
            class_id: len(indices)
            for class_id, indices in class_to_indices.items()
            if len(indices) < required_examples
        }

        if invalid_classes:
            raise ValueError(
                "Some classes have insufficient examples: "
                f"{invalid_classes}"
            )

        self.class_to_indices = dict(class_to_indices)
        self.classes = np.asarray(
            sorted(self.class_to_indices),
            dtype=np.int64,
        )

        if len(self.classes) < self.n_way:
            raise RuntimeError(
                "Not enough valid classes after checking sample counts"
            )

    def sample(self, episode_seed: int) -> Episode:
        rng = np.random.default_rng(int(episode_seed))

        selected_classes = rng.choice(
            self.classes,
            size=self.n_way,
            replace=False,
        )

        support_images: list[torch.Tensor] = []
        support_labels: list[int] = []

        query_images: list[torch.Tensor] = []
        query_labels: list[int] = []

        examples_per_class = self.n_shot + self.n_query

        for local_label, global_class in enumerate(selected_classes):
            candidate_indices = self.class_to_indices[int(global_class)]

            sampled_indices = rng.choice(
                candidate_indices,
                size=examples_per_class,
                replace=False,
            )

            support_indices = sampled_indices[: self.n_shot]
            query_indices = sampled_indices[self.n_shot :]

            for index in support_indices:
                image, _ = self.dataset[int(index)]
                support_images.append(image)
                support_labels.append(local_label)

            for index in query_indices:
                image, _ = self.dataset[int(index)]
                query_images.append(image)
                query_labels.append(local_label)

        support_images_tensor = torch.stack(support_images)
        support_labels_tensor = torch.tensor(
            support_labels,
            dtype=torch.long,
        )

        query_images_tensor = torch.stack(query_images)
        query_labels_tensor = torch.tensor(
            query_labels,
            dtype=torch.long,
        )

        # Remove class-contiguous ordering while preserving reproducibility.
        support_permutation = torch.from_numpy(
            rng.permutation(len(support_labels))
        ).long()

        query_permutation = torch.from_numpy(
            rng.permutation(len(query_labels))
        ).long()

        return Episode(
            support_images=support_images_tensor[support_permutation],
            support_labels=support_labels_tensor[support_permutation],
            query_images=query_images_tensor[query_permutation],
            query_labels=query_labels_tensor[query_permutation],
        )
