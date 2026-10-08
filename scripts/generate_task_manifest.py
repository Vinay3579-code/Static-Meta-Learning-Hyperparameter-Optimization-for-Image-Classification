from __future__ import annotations

import argparse
from pathlib import Path

from sml_hpo.data.registry import load_dataset_splits
from sml_hpo.tasks.generator import generate_task_specs
from sml_hpo.tasks.spec import save_task_manifest


DEFAULT_ROOTS = {
    "omniglot": Path("data/raw/omniglot"),
    "cifar100": Path("data/raw/cifar100"),
    "miniimagenet": Path("data/raw/miniimagenet"),
    "dtd": Path("data/raw/dtd"),
    "flowers102": Path("data/raw/flowers102"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DEFAULT_ROOTS),
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--class-pool-size", type=int, default=10)
    parser.add_argument(
        "--task-tag",
        type=str,
    )
    parser.add_argument(
        "--allow-repeated-class-pools",
        action="store_true",
    )

    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=15)

    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=100,
    )
    parser.add_argument("--test-episodes", type=int, default=600)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root or DEFAULT_ROOTS[args.dataset]

    splits = load_dataset_splits(
        dataset_name=args.dataset,
        root=data_root,
        image_size=84,
        split_seed=args.base_seed,
        download=True,
    )

    if args.split == "train":
        available_classes = splits.train_classes
    elif args.split == "validation":
        available_classes = splits.validation_classes
    else:
        available_classes = splits.test_classes

    tasks = generate_task_specs(
        dataset=args.dataset,
        split=args.split,
        available_classes=available_classes,
        task_count=args.task_count,
        class_pool_size=args.class_pool_size,
        base_seed=args.base_seed,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes,
        task_tag=args.task_tag,
        allow_repeated_class_pools=(
            args.allow_repeated_class_pools
        ),
    )

    save_task_manifest(tasks, args.output)

    print("Dataset:", args.dataset)
    print("Split:", args.split)
    print("Available classes:", len(available_classes))
    print("Generated tasks:", len(tasks))
    print("Class pool size:", args.class_pool_size)
    print("Output:", args.output)
    print("First task:", tasks[0].task_id)
    print("First class pool:", tasks[0].class_ids)
    print("TASK MANIFEST: PASS")


if __name__ == "__main__":
    main()
