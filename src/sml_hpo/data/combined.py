from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from torch.utils.data import Dataset


def extract_targets(dataset: Dataset) -> list[int]:
    for attribute in ("targets", "_labels", "labels"):
        values = getattr(dataset, attribute, None)

        if values is not None:
            return [
                int(value)
                for value in values
            ]

    raise AttributeError(
        f"Could not extract targets from "
        f"{type(dataset).__name__}"
    )


class CombinedLabeledDataset(Dataset):
    """
    Concatenate multiple image-level splits that use the same label space.

    The resulting dataset exposes `targets`, `classes` and `class_to_idx`,
    which are required by EpisodeSampler.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        class_names: Sequence[str],
    ) -> None:
        if not datasets:
            raise ValueError(
                "datasets must not be empty"
            )

        if not class_names:
            raise ValueError(
                "class_names must not be empty"
            )

        self.datasets = tuple(datasets)
        self.classes = tuple(
            str(name)
            for name in class_names
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(self.classes)
        }

        self.cumulative_sizes: list[int] = []
        self.targets: list[int] = []

        running_size = 0

        for dataset in self.datasets:
            dataset_targets = extract_targets(
                dataset
            )

            if len(dataset_targets) != len(dataset):
                raise ValueError(
                    "Target count does not match "
                    "dataset length"
                )

            for target in dataset_targets:
                if not 0 <= target < len(self.classes):
                    raise ValueError(
                        f"Target {target} is outside "
                        f"the class range"
                    )

            self.targets.extend(
                dataset_targets
            )

            running_size += len(dataset)

            self.cumulative_sizes.append(
                running_size
            )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        dataset_index = bisect_right(
            self.cumulative_sizes,
            index,
        )

        previous_size = (
            0
            if dataset_index == 0
            else self.cumulative_sizes[
                dataset_index - 1
            ]
        )

        local_index = (
            index - previous_size
        )

        image, target = self.datasets[
            dataset_index
        ][local_index]

        expected_target = self.targets[
            index
        ]

        if int(target) != expected_target:
            raise RuntimeError(
                "Underlying dataset target changed "
                "after construction"
            )

        return image, expected_target
