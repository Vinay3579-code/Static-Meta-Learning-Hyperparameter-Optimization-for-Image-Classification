from __future__ import annotations

import json
from pathlib import Path

from sml_hpo.tasks.spec import load_task_manifest


DATASETS = (
    "omniglot",
    "cifar100",
    "miniimagenet",
    "dtd",
    "flowers102",
)

REGIMES = (
    "5w1s",
    "10w5s",
)


def main() -> None:
    manifest_root = Path(
        "data/manifests/final800/train"
    )

    output_path = Path(
        "configs/protocols/"
        "budget_stability_tasks.json"
    )

    selected_tasks = []

    for dataset in DATASETS:
        for regime in REGIMES:
            manifest = (
                manifest_root
                / f"{dataset}_{regime}.json"
            )

            tasks = load_task_manifest(
                manifest
            )

            if not tasks:
                raise RuntimeError(
                    f"Empty manifest: {manifest}"
                )

            # Deterministically use the first task.
            task = tasks[0]

            selected_tasks.append(
                {
                    "task_id": task.task_id,
                    "dataset": task.dataset,
                    "regime": regime,
                    "n_way": task.n_way,
                    "n_shot": task.n_shot,
                    "n_query": task.n_query,
                    "manifest": str(manifest),
                }
            )

    task_ids = [
        task["task_id"]
        for task in selected_tasks
    ]

    if len(task_ids) != 10:
        raise RuntimeError(
            f"Expected 10 tasks, found {len(task_ids)}"
        )

    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError(
            "Duplicate task IDs selected"
        )

    payload = {
        "schema_version": 1,
        "purpose": (
            "oracle_training_budget_stability"
        ),
        "budgets": [100, 200, 300],
        "reference_budget": 300,
        "validation_episodes": 100,
        "model_seeds_screening": [101],
        "tasks": selected_tasks,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Selected tasks:", len(selected_tasks))

    for task in selected_tasks:
        print(
            f"{task['dataset']:12s} | "
            f"{task['regime']:6s} | "
            f"{task['task_id']}"
        )

    print("Output:", output_path)
    print("BUDGET-STABILITY TASK SELECTION: PASS")


if __name__ == "__main__":
    main()
