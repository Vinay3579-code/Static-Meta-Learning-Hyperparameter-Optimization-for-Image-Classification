from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _normalise_class_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class MiniImageNetCacheDataset(Dataset):
    """
    Dataset for the commonly used Ravi/Larochelle miniImageNet cache.

    Expected pickle structure:
        {
            "image_data": ndarray [N, 84, 84, 3],
            "class_dict": {
                class_name: [sample indices]
            }
        }
    """

    def __init__(
        self,
        cache_path: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.transform = transform

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"miniImageNet cache not found: {self.cache_path}"
            )

        with self.cache_path.open("rb") as file:
            cache = pickle.load(file)

        if not isinstance(cache, dict):
            raise TypeError(
                f"Expected dictionary in {self.cache_path}, "
                f"received {type(cache)!r}"
            )

        if "image_data" not in cache:
            raise KeyError(
                f"'image_data' missing from {self.cache_path}"
            )

        if "class_dict" not in cache:
            raise KeyError(
                f"'class_dict' missing from {self.cache_path}"
            )

        images = np.asarray(cache["image_data"])
        class_dict = cache["class_dict"]

        if images.ndim != 4:
            raise ValueError(
                f"Expected four-dimensional image array, got "
                f"shape {images.shape}"
            )

        # Accept NHWC, which is used by the standard cache.
        if images.shape[-1] != 3:
            raise ValueError(
                "Expected RGB images in NHWC format; "
                f"received shape {images.shape}"
            )

        if images.dtype != np.uint8:
            if images.min() < 0 or images.max() > 255:
                raise ValueError(
                    "Image values are outside the valid uint8 range"
                )
            images = images.astype(np.uint8)

        if not isinstance(class_dict, dict):
            raise TypeError("'class_dict' must be a dictionary")

        sorted_entries = sorted(
            class_dict.items(),
            key=lambda item: _normalise_class_name(item[0]),
        )

        self.class_names = tuple(
            _normalise_class_name(class_name)
            for class_name, _ in sorted_entries
        )

        targets = np.full(
            shape=len(images),
            fill_value=-1,
            dtype=np.int64,
        )

        for local_label, (_, indices) in enumerate(sorted_entries):
            indices_array = np.asarray(
                list(indices),
                dtype=np.int64,
            )

            if indices_array.ndim != 1:
                raise ValueError(
                    "Each class index collection must be one-dimensional"
                )

            if len(indices_array) == 0:
                raise ValueError(
                    f"Class {self.class_names[local_label]} is empty"
                )

            if indices_array.min() < 0:
                raise ValueError("Negative image index detected")

            if indices_array.max() >= len(images):
                raise ValueError(
                    "Class dictionary contains an out-of-range index"
                )

            if np.any(targets[indices_array] != -1):
                raise ValueError(
                    "An image index belongs to multiple classes"
                )

            targets[indices_array] = local_label

        if np.any(targets == -1):
            missing_count = int(np.sum(targets == -1))
            raise ValueError(
                f"{missing_count} images have no class assignment"
            )

        self.images = images
        self.targets = targets.tolist()
        self.classes = self.class_names

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        image = Image.fromarray(
            self.images[index],
            mode="RGB",
        )

        if self.transform is not None:
            image = self.transform(image)

        target = int(self.targets[index])

        return image, target


@dataclass(frozen=True)
class MiniImageNetSplits:
    train_dataset: MiniImageNetCacheDataset
    validation_dataset: MiniImageNetCacheDataset
    test_dataset: MiniImageNetCacheDataset

    train_classes: tuple[int, ...]
    validation_classes: tuple[int, ...]
    test_classes: tuple[int, ...]

    train_class_names: tuple[str, ...]
    validation_class_names: tuple[str, ...]
    test_class_names: tuple[str, ...]


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
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _resolve_cache(
    root: Path,
    split: str,
) -> Path:
    candidates = [
        root / f"mini-imagenet-cache-{split}.pkl",
        root / f"mini-imagenet-cache-{split}.pickle",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    names = "\n".join(str(path) for path in candidates)

    raise FileNotFoundError(
        f"No cache found for miniImageNet split '{split}'. "
        f"Expected one of:\n{names}"
    )


def load_miniimagenet_splits(
    root: str | Path,
    image_size: int = 84,
) -> MiniImageNetSplits:
    root = Path(root)
    transform = _build_transform(image_size)

    train_dataset = MiniImageNetCacheDataset(
        cache_path=_resolve_cache(root, "train"),
        transform=transform,
    )

    validation_dataset = MiniImageNetCacheDataset(
        cache_path=_resolve_cache(root, "val"),
        transform=transform,
    )

    test_dataset = MiniImageNetCacheDataset(
        cache_path=_resolve_cache(root, "test"),
        transform=transform,
    )

    train_names = set(train_dataset.class_names)
    validation_names = set(validation_dataset.class_names)
    test_names = set(test_dataset.class_names)

    if not train_names.isdisjoint(validation_names):
        raise ValueError(
            "miniImageNet train and validation classes overlap"
        )

    if not train_names.isdisjoint(test_names):
        raise ValueError(
            "miniImageNet train and test classes overlap"
        )

    if not validation_names.isdisjoint(test_names):
        raise ValueError(
            "miniImageNet validation and test classes overlap"
        )

    return MiniImageNetSplits(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_classes=tuple(range(len(train_dataset.classes))),
        validation_classes=tuple(
            range(len(validation_dataset.classes))
        ),
        test_classes=tuple(range(len(test_dataset.classes))),
        train_class_names=train_dataset.class_names,
        validation_class_names=validation_dataset.class_names,
        test_class_names=test_dataset.class_names,
    )
