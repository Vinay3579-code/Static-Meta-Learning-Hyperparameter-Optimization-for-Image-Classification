from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Omniglot


@dataclass(frozen=True)
class OmniglotSplits:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset

    train_classes: tuple[int, ...]
    validation_classes: tuple[int, ...]
    test_classes: tuple[int, ...]


def _extract_targets(dataset: Omniglot) -> list[int]:
    """
    Extract integer class labels from Torchvision's Omniglot dataset.

    `_flat_character_images` contains tuples in the form:
        (image_filename, integer_class_index)
    """
    flat_images = getattr(dataset, "_flat_character_images", None)

    if flat_images is not None:
        return [
            int(character_class)
            for _, character_class in flat_images
        ]

    # Compatibility fallback if Torchvision internals change.
    return [
        int(dataset[index][1])
        for index in range(len(dataset))
    ]

def _build_transform(image_size: int) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def _split_classes(
    classes: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    class_array = np.asarray(sorted(set(classes)), dtype=np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(class_array)

    validation_count = max(
        1,
        int(round(len(class_array) * validation_fraction)),
    )

    validation_classes = np.sort(class_array[:validation_count])
    train_classes = np.sort(class_array[validation_count:])

    if len(train_classes) == 0:
        raise RuntimeError("No training classes remain after splitting")

    return (
        tuple(int(value) for value in train_classes),
        tuple(int(value) for value in validation_classes),
    )


def load_omniglot_splits(
    root: str | Path,
    image_size: int = 84,
    validation_fraction: float = 0.20,
    split_seed: int = 42,
    download: bool = True,
) -> OmniglotSplits:
    """
    Split protocol:

    - Omniglot background classes:
      class-disjoint meta-training and validation classes.
    - Omniglot evaluation classes:
      held-out meta-test classes.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    transform = _build_transform(image_size)

    background_dataset = Omniglot(
        root=str(root),
        background=True,
        transform=transform,
        download=download,
    )

    evaluation_dataset = Omniglot(
        root=str(root),
        background=False,
        transform=transform,
        download=download,
    )

    background_targets = _extract_targets(background_dataset)
    evaluation_targets = _extract_targets(evaluation_dataset)

    # Attach targets so the generic episode sampler can use them.
    background_dataset.targets = background_targets
    evaluation_dataset.targets = evaluation_targets

    background_classes = sorted(set(background_targets))
    evaluation_classes = sorted(set(evaluation_targets))

    train_classes, validation_classes = _split_classes(
        classes=background_classes,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    test_classes = tuple(int(value) for value in evaluation_classes)

    return OmniglotSplits(
        train_dataset=background_dataset,
        validation_dataset=background_dataset,
        test_dataset=evaluation_dataset,
        train_classes=train_classes,
        validation_classes=validation_classes,
        test_classes=test_classes,
    )
