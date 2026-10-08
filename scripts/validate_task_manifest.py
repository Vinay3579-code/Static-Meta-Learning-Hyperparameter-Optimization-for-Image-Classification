from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sml_hpo.data.registry import load_dataset_splits
from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.tasks.spec import load_task_manifest


DEFAULT_ROOTS = {
    "omniglot": Path("data/raw/omniglot"),
    "cifar100": Path("data/raw/cifar100"),
    "miniimagenet": Path("data/raw/miniimagenet"),
    "dtd": Path("data/raw/dtd"),
    "flowers102": Path("data/raw/flowers102"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def select_dataset(splits, split_name: str):
    if split_name == "train":
        return splits.train_dataset
    if split_name == "validation":
        return splits.validation_dataset
    if split_name == "test":
        return splits.test_dataset
    raise ValueError(split_name)


def main() -> None:
    args = parse_args()
    tasks = load_task_manifest(args.manifest)

    if not tasks:
        raise RuntimeError("Manifest is empty")

    dataset_names = {task.dataset for task in tasks}
    split_names = {task.split for task in tasks}

    if len(dataset_names) != 1:
        raise RuntimeError("Manifest mixes multiple datasets")

    if len(split_names) != 1:
        raise RuntimeError("Manifest mixes multiple splits")

    first_task = tasks[0]

    splits = load_dataset_splits(
        dataset_name=first_task.dataset,
        root=DEFAULT_ROOTS[first_task.dataset],
        image_size=84,
        split_seed=42,
        download=False,
    )

    dataset = select_dataset(splits, first_task.split)

    sampler = EpisodeSampler(
        dataset=dataset,
        allowed_classes=first_task.class_ids,
        n_way=first_task.n_way,
        n_shot=first_task.n_shot,
        n_query=first_task.n_query,
    )

    first_seed = first_task.training_seeds()[0]

    episode_a = sampler.sample(first_seed)
    episode_b = sampler.sample(first_seed)

    expected_support = (
        first_task.n_way * first_task.n_shot,
        3,
        84,
        84,
    )

    expected_query = (
        first_task.n_way * first_task.n_query,
        3,
        84,
        84,
    )

    assert tuple(episode_a.support_images.shape) == expected_support
    assert tuple(episode_a.query_images.shape) == expected_query

    assert torch.equal(
        episode_a.support_images,
        episode_b.support_images,
    )
    assert torch.equal(
        episode_a.support_labels,
        episode_b.support_labels,
    )
    assert torch.equal(
        episode_a.query_images,
        episode_b.query_images,
    )
    assert torch.equal(
        episode_a.query_labels,
        episode_b.query_labels,
    )

    all_ids = [task.task_id for task in tasks]
    all_class_pools = [task.class_ids for task in tasks]

    unique_class_pool_count = len(
        set(all_class_pools)
    )

    assert len(all_ids) == len(set(all_ids))
    print("Manifest:", args.manifest)
    print("Dataset:", first_task.dataset)
    print("Split:", first_task.split)
    print("Tasks:", len(tasks))
    print("Support shape:", tuple(episode_a.support_images.shape))
    print("Query shape:", tuple(episode_a.query_images.shape))
    print("Deterministic episode: PASS")
    print("Unique task IDs: PASS")
    print(
        "Unique class pools:",
        unique_class_pool_count,
        "/",
        len(all_class_pools),
    )
    print("TASK MANIFEST VALIDATION: PASS")


if __name__ == "__main__":
    main()
