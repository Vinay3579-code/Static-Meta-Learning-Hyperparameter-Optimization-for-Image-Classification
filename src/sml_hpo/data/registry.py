from __future__ import annotations

from pathlib import Path
from typing import Any

from sml_hpo.data.cifar100 import (
    load_cifar100_splits,
)
from sml_hpo.data.crossdomain import (
    load_dtd_splits,
    load_flowers102_splits,
)
from sml_hpo.data.miniimagenet import (
    load_miniimagenet_splits,
)
from sml_hpo.data.omniglot import (
    load_omniglot_splits,
)


SUPPORTED_DATASETS = (
    "omniglot",
    "cifar100",
    "miniimagenet",
    "dtd",
    "flowers102",
)


def load_dataset_splits(
    dataset_name: str,
    root: str | Path,
    image_size: int = 84,
    split_seed: int = 42,
    download: bool = True,
) -> Any:
    normalized_name = (
        dataset_name
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    if normalized_name == "omniglot":
        return load_omniglot_splits(
            root=root,
            image_size=image_size,
            validation_fraction=0.20,
            split_seed=split_seed,
            download=download,
        )

    if normalized_name == "cifar100":
        return load_cifar100_splits(
            root=root,
            image_size=image_size,
            split_seed=split_seed,
            train_class_count=64,
            validation_class_count=16,
            download=download,
        )

    if normalized_name == "miniimagenet":
        return load_miniimagenet_splits(
            root=root,
            image_size=image_size,
        )

    if normalized_name == "dtd":
        return load_dtd_splits(
            root=root,
            image_size=image_size,
            split_seed=split_seed,
            partition=1,
            download=download,
        )

    if normalized_name == "flowers102":
        return load_flowers102_splits(
            root=root,
            image_size=image_size,
            split_seed=split_seed,
            download=download,
        )

    raise ValueError(
        f"Unsupported dataset: {dataset_name!r}. "
        f"Supported datasets: {SUPPORTED_DATASETS}"
    )
