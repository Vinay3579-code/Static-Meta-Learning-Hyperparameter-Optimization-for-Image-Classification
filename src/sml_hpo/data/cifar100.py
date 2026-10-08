from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100


class CombinedCIFAR100(Dataset):
    """
    Combines the official CIFAR-100 train and test image partitions.

    Class-disjoint meta-train, validation and meta-test partitions are
    created afterward. Each class therefore contains all 600 images.
    """

    def __init__(
        self,
        train_dataset: CIFAR100,
        test_dataset: CIFAR100,
    ) -> None:
        if train_dataset.classes != test_dataset.classes:
            raise ValueError(
                "CIFAR-100 train and test class order is inconsistent"
            )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_length = len(train_dataset)

        self.targets = (
            [int(value) for value in train_dataset.targets]
            + [int(value) for value in test_dataset.targets]
        )

        self.classes = tuple(train_dataset.classes)
        self.class_to_idx = dict(train_dataset.class_to_idx)

    def __len__(self) -> int:
        return len(self.train_dataset) + len(self.test_dataset)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        if index < self.train_length:
            return self.train_dataset[index]

        return self.test_dataset[index - self.train_length]


@dataclass(frozen=True)
class CIFAR100Splits:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset

    train_classes: tuple[int, ...]
    validation_classes: tuple[int, ...]
    test_classes: tuple[int, ...]

    class_names: tuple[str, ...]


def _build_transform(image_size: int) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )


def _split_classes(
    number_of_classes: int,
    train_class_count: int,
    validation_class_count: int,
    split_seed: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    if number_of_classes <= 0:
        raise ValueError("number_of_classes must be positive")

    test_class_count = (
        number_of_classes
        - train_class_count
        - validation_class_count
    )

    if min(
        train_class_count,
        validation_class_count,
        test_class_count,
    ) <= 0:
        raise ValueError(
            "Train, validation and test class counts must be positive"
        )

    classes = np.arange(number_of_classes, dtype=np.int64)

    rng = np.random.default_rng(split_seed)
    rng.shuffle(classes)

    train_end = train_class_count
    validation_end = train_end + validation_class_count

    train_classes = np.sort(classes[:train_end])
    validation_classes = np.sort(
        classes[train_end:validation_end]
    )
    test_classes = np.sort(classes[validation_end:])

    return (
        tuple(int(value) for value in train_classes),
        tuple(int(value) for value in validation_classes),
        tuple(int(value) for value in test_classes),
    )


def load_cifar100_splits(
    root: str | Path,
    image_size: int = 84,
    split_seed: int = 42,
    train_class_count: int = 64,
    validation_class_count: int = 16,
    download: bool = True,
) -> CIFAR100Splits:
    """
    Load all 60,000 CIFAR-100 images and create class-disjoint splits.

    Development protocol:
        64 classes for meta-training
        16 classes for validation/HPO
        20 classes for held-out meta-testing

    The exact class identities are deterministic for split_seed.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    transform = _build_transform(image_size)

    official_train = CIFAR100(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )

    official_test = CIFAR100(
        root=str(root),
        train=False,
        transform=transform,
        download=download,
    )

    combined_dataset = CombinedCIFAR100(
        train_dataset=official_train,
        test_dataset=official_test,
    )

    train_classes, validation_classes, test_classes = _split_classes(
        number_of_classes=len(combined_dataset.classes),
        train_class_count=train_class_count,
        validation_class_count=validation_class_count,
        split_seed=split_seed,
    )

    return CIFAR100Splits(
        train_dataset=combined_dataset,
        validation_dataset=combined_dataset,
        test_dataset=combined_dataset,
        train_classes=train_classes,
        validation_classes=validation_classes,
        test_classes=test_classes,
        class_names=combined_dataset.classes,
    )
