from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TaskSpec:
    """
    Reproducible few-shot HPO task.

    A task is an episode distribution over a fixed class pool. Every
    hyperparameter configuration evaluated on this task must receive the
    same training, validation and test episode seeds.
    """

    schema_version: int
    task_id: str
    dataset: str
    split: str

    class_ids: tuple[int, ...]
    task_seed: int

    n_way: int
    n_shot: int
    n_query: int

    train_episodes: int
    validation_episodes: int
    test_episodes: int

    train_seed_base: int
    validation_seed_base: int
    test_seed_base: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version}"
            )

        if not self.task_id:
            raise ValueError("task_id must not be empty")

        if not self.dataset:
            raise ValueError("dataset must not be empty")

        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {VALID_SPLITS}, got {self.split!r}"
            )

        if self.n_way <= 1:
            raise ValueError("n_way must be greater than one")

        if self.n_shot <= 0 or self.n_query <= 0:
            raise ValueError("n_shot and n_query must be positive")

        if len(self.class_ids) < self.n_way:
            raise ValueError(
                "The task class pool must contain at least n_way classes"
            )

        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("class_ids contains duplicates")

        if min(
            self.train_episodes,
            self.validation_episodes,
            self.test_episodes,
        ) <= 0:
            raise ValueError("Episode counts must be positive")

    def training_seeds(self) -> tuple[int, ...]:
        return tuple(
            self.train_seed_base + index
            for index in range(self.train_episodes)
        )

    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(
            self.validation_seed_base + index
            for index in range(self.validation_episodes)
        )

    def testing_seeds(self) -> tuple[int, ...]:
        return tuple(
            self.test_seed_base + index
            for index in range(self.test_episodes)
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["class_ids"] = list(self.class_ids)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        copied = dict(value)
        copied["class_ids"] = tuple(
            int(class_id)
            for class_id in copied["class_ids"]
        )
        return cls(**copied)


def save_task_manifest(
    tasks: Iterable[TaskSpec],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    task_list = list(tasks)

    task_ids = [task.task_id for task in task_list]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task manifest contains duplicate task IDs")

    payload = {
        "schema_version": 1,
        "task_count": len(task_list),
        "tasks": [task.to_dict() for task in task_list],
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_task_manifest(
    path: str | Path,
) -> list[TaskSpec]:
    path = Path(path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported task-manifest schema")

    tasks = [
        TaskSpec.from_dict(item)
        for item in payload["tasks"]
    ]

    if payload.get("task_count") != len(tasks):
        raise ValueError("Manifest task_count does not match task data")

    return tasks
