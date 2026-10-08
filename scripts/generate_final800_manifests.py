from __future__ import annotations

import json
import math
from pathlib import Path

from sml_hpo.data.registry import (
    load_dataset_splits,
)
from sml_hpo.tasks.generator import (
    generate_task_specs,
)
from sml_hpo.tasks.spec import (
    save_task_manifest,
)


DATASETS = {
    "omniglot": Path(
        "data/raw/omniglot"
    ),
    "cifar100": Path(
        "data/raw/cifar100"
    ),
    "miniimagenet": Path(
        "data/raw/miniimagenet"
    ),
    "dtd": Path(
        "data/raw/dtd"
    ),
    "flowers102": Path(
        "data/raw/flowers102"
    ),
}


REGIMES = {
    "5w1s": {
        "n_way": 5,
        "n_shot": 1,
        "n_query": 15,
        "preferred_class_pool_size": 10,
    },
    "5w5s": {
        "n_way": 5,
        "n_shot": 5,
        "n_query": 15,
        "preferred_class_pool_size": 10,
    },
    "10w1s": {
        "n_way": 10,
        "n_shot": 1,
        "n_query": 10,
        "preferred_class_pool_size": 15,
    },
    "10w5s": {
        "n_way": 10,
        "n_shot": 5,
        "n_query": 10,
        "preferred_class_pool_size": 15,
    },
}


TASK_COUNTS = {
    "train": 28,
    "validation": 6,
    "test": 6,
}


def classes_for_split(
    splits,
    split_name: str,
) -> tuple[int, ...]:
    if split_name == "train":
        return tuple(
            splits.train_classes
        )

    if split_name == "validation":
        return tuple(
            splits.validation_classes
        )

    if split_name == "test":
        return tuple(
            splits.test_classes
        )

    raise ValueError(split_name)


def main() -> None:
    output_root = Path(
        "data/manifests/final800"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_rows = []
    all_task_ids: set[str] = set()

    total_tasks = 0

    for dataset_name, data_root in (
        DATASETS.items()
    ):
        splits = load_dataset_splits(
            dataset_name=dataset_name,
            root=data_root,
            image_size=84,
            split_seed=42,
            download=False,
        )

        for split_name, task_count in (
            TASK_COUNTS.items()
        ):
            available_classes = (
                classes_for_split(
                    splits,
                    split_name,
                )
            )

            for regime_name, regime in (
                REGIMES.items()
            ):
                class_pool_size = min(
                    regime[
                        "preferred_class_pool_size"
                    ],
                    len(available_classes),
                )

                if (
                    class_pool_size
                    < regime["n_way"]
                ):
                    raise RuntimeError(
                        f"{dataset_name}/{split_name}/"
                        f"{regime_name} has only "
                        f"{len(available_classes)} classes"
                    )

                possible_unique_pools = math.comb(
                    len(available_classes),
                    class_pool_size,
                )

                allow_repeated = (
                    possible_unique_pools
                    < task_count
                )

                tasks = generate_task_specs(
                    dataset=dataset_name,
                    split=split_name,
                    available_classes=(
                        available_classes
                    ),
                    task_count=task_count,
                    class_pool_size=(
                        class_pool_size
                    ),
                    base_seed=20260805,
                    n_way=regime["n_way"],
                    n_shot=regime["n_shot"],
                    n_query=regime["n_query"],
                    train_episodes=300,
                    validation_episodes=100,
                    test_episodes=600,
                    task_tag=regime_name,
                    allow_repeated_class_pools=(
                        allow_repeated
                    ),
                )

                output_path = (
                    output_root
                    / split_name
                    / (
                        f"{dataset_name}_"
                        f"{regime_name}.json"
                    )
                )

                save_task_manifest(
                    tasks,
                    output_path,
                )

                for task in tasks:
                    if task.task_id in all_task_ids:
                        raise RuntimeError(
                            "Duplicate task ID: "
                            f"{task.task_id}"
                        )

                    all_task_ids.add(
                        task.task_id
                    )

                    index_rows.append(
                        {
                            "task_id": task.task_id,
                            "dataset": dataset_name,
                            "split": split_name,
                            "regime": regime_name,
                            "n_way": task.n_way,
                            "n_shot": task.n_shot,
                            "n_query": task.n_query,
                            "class_pool_size": len(
                                task.class_ids
                            ),
                            "task_seed": (
                                task.task_seed
                            ),
                            "manifest": str(
                                output_path
                            ),
                        }
                    )

                total_tasks += len(tasks)

                print(
                    f"{dataset_name:12s} | "
                    f"{split_name:10s} | "
                    f"{regime_name:6s} | "
                    f"tasks={len(tasks):2d} | "
                    f"classes={len(available_classes):3d} | "
                    f"pool={class_pool_size:2d} | "
                    f"repeat={allow_repeated}"
                )

    expected_total = (
        560 + 120 + 120
    )

    if total_tasks != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} tasks, "
            f"generated {total_tasks}"
        )

    split_counts = {
        split_name: sum(
            1
            for row in index_rows
            if row["split"] == split_name
        )
        for split_name in TASK_COUNTS
    }

    if split_counts != {
        "train": 560,
        "validation": 120,
        "test": 120,
    }:
        raise RuntimeError(
            f"Invalid split counts: {split_counts}"
        )

    index_payload = {
        "schema_version": 1,
        "protocol_name": (
            "five_dataset_four_regime_final800"
        ),
        "generation_seed": 20260805,
        "dataset_split_seed": 42,
        "task_count": total_tasks,
        "split_counts": split_counts,
        "dataset_count": len(DATASETS),
        "regime_count": len(REGIMES),
        "datasets": list(DATASETS),
        "regimes": REGIMES,
        "task_budget": {
            "training_episodes": 300,
            "validation_episodes": 100,
            "test_episodes": 600,
        },
        "tasks": index_rows,
    }

    index_path = (
        output_root
        / "index.json"
    )

    index_path.write_text(
        json.dumps(
            index_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Total tasks:", total_tasks)
    print("Split counts:", split_counts)
    print("Unique task IDs:", len(all_task_ids))
    print("Index:", index_path)
    print("FINAL 800-TASK MANIFESTS: PASS")


if __name__ == "__main__":
    main()
