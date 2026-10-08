from __future__ import annotations

from collections import Counter
from pathlib import Path

from sml_hpo.data.registry import (
    load_dataset_splits,
)


EXPECTED = {
    "dtd": {
        "root": Path("data/raw/dtd"),
        "images": 5640,
        "classes": (27, 10, 10),
    },
    "flowers102": {
        "root": Path(
            "data/raw/flowers102"
        ),
        "images": 8189,
        "classes": (64, 18, 20),
    },
}


def main() -> None:
    for dataset_name, expected in (
        EXPECTED.items()
    ):
        print()
        print("=" * 64)
        print("Preparing:", dataset_name)
        print("=" * 64)

        splits = load_dataset_splits(
            dataset_name=dataset_name,
            root=expected["root"],
            image_size=84,
            split_seed=42,
            download=True,
        )

        total_images = len(
            splits.train_dataset
        )

        class_counts = (
            len(splits.train_classes),
            len(splits.validation_classes),
            len(splits.test_classes),
        )

        targets = splits.train_dataset.targets

        images_per_class = Counter(
            int(target)
            for target in targets
        )

        print("Images:", total_images)
        print(
            "Total classes:",
            len(images_per_class),
        )
        print(
            "Class split:",
            class_counts,
        )
        print(
            "Minimum images/class:",
            min(images_per_class.values()),
        )
        print(
            "Maximum images/class:",
            max(images_per_class.values()),
        )

        assert (
            total_images
            == expected["images"]
        )

        assert (
            class_counts
            == expected["classes"]
        )

        train = set(
            splits.train_classes
        )
        validation = set(
            splits.validation_classes
        )
        test = set(
            splits.test_classes
        )

        assert train.isdisjoint(
            validation
        )
        assert train.isdisjoint(test)
        assert validation.isdisjoint(
            test
        )

        assert len(
            train | validation | test
        ) == len(images_per_class)

        print(
            f"{dataset_name} preparation: PASS"
        )

    print()
    print(
        "CROSS-DOMAIN DATA PREPARATION: PASS"
    )


if __name__ == "__main__":
    main()
