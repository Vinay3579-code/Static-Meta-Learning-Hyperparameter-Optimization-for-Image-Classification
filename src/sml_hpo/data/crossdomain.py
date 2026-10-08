from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import DTD, Flowers102

from sml_hpo.data.combined import (
    CombinedLabeledDataset,
)


@dataclass(frozen=True)
class CrossDomainSplits:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset

    train_classes: tuple[int, ...]
    validation_classes: tuple[int, ...]
    test_classes: tuple[int, ...]

    class_names: tuple[str, ...]


def _build_transform(
    image_size: int,
) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError(
            "image_size must be positive"
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _split_classes(
    *,
    number_of_classes: int,
    train_class_count: int,
    validation_class_count: int,
    split_seed: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
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
            "Every class split must be non-empty"
        )

    classes = np.arange(
        number_of_classes,
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        split_seed
    )

    rng.shuffle(classes)

    train_end = train_class_count
    validation_end = (
        train_end
        + validation_class_count
    )

    train_classes = np.sort(
        classes[:train_end]
    )

    validation_classes = np.sort(
        classes[
            train_end:validation_end
        ]
    )

    test_classes = np.sort(
        classes[validation_end:]
    )

    return (
        tuple(
            int(value)
            for value in train_classes
        ),
        tuple(
            int(value)
            for value in validation_classes
        ),
        tuple(
            int(value)
            for value in test_classes
        ),
    )


def load_dtd_splits(
    root: str | Path,
    image_size: int = 84,
    split_seed: int = 42,
    partition: int = 1,
    download: bool = True,
) -> CrossDomainSplits:
    """
    Combine DTD's image-level train/val/test splits, then create
    deterministic 27/10/10 class-disjoint meta splits.
    """
    root = Path(root)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform = _build_transform(
        image_size
    )

    official_train = DTD(
        root=str(root),
        split="train",
        partition=partition,
        transform=transform,
        download=download,
    )

    official_validation = DTD(
        root=str(root),
        split="val",
        partition=partition,
        transform=transform,
        download=download,
    )

    official_test = DTD(
        root=str(root),
        split="test",
        partition=partition,
        transform=transform,
        download=download,
    )

    class_names = tuple(
        official_train.classes
    )

    for dataset in (
        official_validation,
        official_test,
    ):
        if tuple(dataset.classes) != class_names:
            raise ValueError(
                "DTD splits use inconsistent "
                "class ordering"
            )

    combined = CombinedLabeledDataset(
        datasets=(
            official_train,
            official_validation,
            official_test,
        ),
        class_names=class_names,
    )

    (
        train_classes,
        validation_classes,
        test_classes,
    ) = _split_classes(
        number_of_classes=len(
            class_names
        ),
        train_class_count=27,
        validation_class_count=10,
        split_seed=split_seed,
    )

    return CrossDomainSplits(
        train_dataset=combined,
        validation_dataset=combined,
        test_dataset=combined,
        train_classes=train_classes,
        validation_classes=validation_classes,
        test_classes=test_classes,
        class_names=class_names,
    )


def load_flowers102_splits(
    root: str | Path,
    image_size: int = 84,
    split_seed: int = 42,
    download: bool = True,
) -> CrossDomainSplits:
    """
    Combine Flowers102's image-level train/val/test splits, then create
    deterministic 64/18/20 class-disjoint meta splits.
    """
    root = Path(root)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform = _build_transform(
        image_size
    )

    official_train = Flowers102(
        root=str(root),
        split="train",
        transform=transform,
        download=download,
    )

    official_validation = Flowers102(
        root=str(root),
        split="val",
        transform=transform,
        download=download,
    )

    official_test = Flowers102(
        root=str(root),
        split="test",
        transform=transform,
        download=download,
    )

    class_names = tuple(
        official_train.classes
    )

    for dataset in (
        official_validation,
        official_test,
    ):
        if tuple(dataset.classes) != class_names:
            raise ValueError(
                "Flowers102 splits use inconsistent "
                "class ordering"
            )

    combined = CombinedLabeledDataset(
        datasets=(
            official_train,
            official_validation,
            official_test,
        ),
        class_names=class_names,
    )

    (
        train_classes,
        validation_classes,
        test_classes,
    ) = _split_classes(
        number_of_classes=len(
            class_names
        ),
        train_class_count=64,
        validation_class_count=18,
        split_seed=split_seed,
    )

    return CrossDomainSplits(
        train_dataset=combined,
        validation_dataset=combined,
        test_dataset=combined,
        train_classes=train_classes,
        validation_classes=validation_classes,
        test_classes=test_classes,
        class_names=class_names,
    )
