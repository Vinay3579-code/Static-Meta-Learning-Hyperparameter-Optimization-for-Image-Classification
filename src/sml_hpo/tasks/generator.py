from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from sml_hpo.tasks.spec import TaskSpec


def _normalize_tag(
    task_tag: str | None,
) -> str | None:
    if task_tag is None:
        return None

    normalized = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        task_tag.strip(),
    ).strip("_").lower()

    if not normalized:
        raise ValueError(
            "task_tag must contain letters or numbers"
        )

    return normalized


def _stable_seed(
    *,
    base_seed: int,
    dataset: str,
    split: str,
    task_tag: str | None,
    n_way: int,
    n_shot: int,
    n_query: int,
) -> int:
    """
    Produce a platform-independent seed from the complete task regime.

    Python's built-in hash is deliberately randomized between processes,
    so SHA-256 is used instead.
    """
    material = "|".join(
        [
            str(int(base_seed)),
            dataset,
            split,
            task_tag or "untagged",
            str(n_way),
            str(n_shot),
            str(n_query),
        ]
    )

    digest = hashlib.sha256(
        material.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % (2**63 - 1)


def generate_task_specs(
    *,
    dataset: str,
    split: str,
    available_classes: Sequence[int],
    task_count: int,
    class_pool_size: int,
    base_seed: int,
    n_way: int,
    n_shot: int,
    n_query: int,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    task_tag: str | None = None,
    allow_repeated_class_pools: bool = False,
) -> list[TaskSpec]:
    normalized_dataset = (
        dataset.strip().lower()
    )

    normalized_tag = _normalize_tag(
        task_tag
    )

    if split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            f"Unsupported split: {split!r}"
        )

    if task_count <= 0:
        raise ValueError(
            "task_count must be positive"
        )

    available = np.asarray(
        sorted(
            set(
                int(value)
                for value in available_classes
            )
        ),
        dtype=np.int64,
    )

    if class_pool_size < n_way:
        raise ValueError(
            "class_pool_size must be at least n_way"
        )

    if class_pool_size > len(available):
        raise ValueError(
            f"class_pool_size={class_pool_size} exceeds "
            f"available classes={len(available)}"
        )

    combined_seed = _stable_seed(
        base_seed=base_seed,
        dataset=normalized_dataset,
        split=split,
        task_tag=normalized_tag,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
    )

    rng = np.random.default_rng(
        combined_seed
    )

    task_specs: list[TaskSpec] = []
    used_class_pools: set[
        tuple[int, ...]
    ] = set()

    for task_index in range(task_count):
        class_pool: tuple[int, ...] | None = None

        for _ in range(10_000):
            candidate = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(
                        available,
                        size=class_pool_size,
                        replace=False,
                    )
                )
            )

            if candidate not in used_class_pools:
                class_pool = candidate
                used_class_pools.add(candidate)
                break

        if class_pool is None:
            if not allow_repeated_class_pools:
                raise RuntimeError(
                    "Could not generate another unique class pool. "
                    "Enable allow_repeated_class_pools for limited "
                    "class splits."
                )

            class_pool = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(
                        available,
                        size=class_pool_size,
                        replace=False,
                    )
                )
            )

        # Each repeated class pool still defines a distinct task because
        # it receives independent fixed train/validation/test seed banks.
        task_seed = (
            combined_seed
            + (task_index + 1) * 100_000
        ) % (2**63 - 1)

        identifier_parts = [
            normalized_dataset,
            split,
        ]

        if normalized_tag is not None:
            identifier_parts.append(
                normalized_tag
            )

        identifier_parts.append(
            f"{task_index:04d}"
        )

        task_specs.append(
            TaskSpec(
                schema_version=1,
                task_id="_".join(
                    identifier_parts
                ),
                dataset=normalized_dataset,
                split=split,
                class_ids=class_pool,
                task_seed=task_seed,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                train_episodes=train_episodes,
                validation_episodes=validation_episodes,
                test_episodes=test_episodes,
                train_seed_base=task_seed + 1_000,
                validation_seed_base=task_seed + 20_000,
                test_seed_base=task_seed + 40_000,
            )
        )

    task_ids = [
        task.task_id
        for task in task_specs
    ]

    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError(
            "Generated duplicate task IDs"
        )

    return task_specs
