from __future__ import annotations

from sml_hpo.baselines.adapter_v2 import (
    build_default_adapter,
)


def main():
    adapter = build_default_adapter(
        device="cuda:0"
    )

    test_tasks = adapter.task_ids(
        split="test",
        dataset="omniglot",
    )

    if not test_tasks:
        raise RuntimeError(
            "No Omniglot test tasks found."
        )

    task_id = test_tasks[0]

    # Known member of the frozen 64-anchor space.
    config_id = "anchor_v2_058"

    print(
        "Task:",
        task_id,
    )

    print(
        "Config:",
        config_id,
    )

    print(
        "Budget:",
        25,
        "training episodes",
    )

    result = (
        adapter.train_validate_anchor(
            task_id=task_id,

            config_id=config_id,

            budget=25,

            model_seed=101,
        )
    )

    print()
    print(
        "Validation accuracy:",
        result[
            "validation_accuracy_mean"
        ],
    )

    print(
        "Training episodes:",
        result[
            "training_episodes"
        ],
    )

    print(
        "Validation episodes:",
        result[
            "validation_episodes"
        ],
    )

    print(
        "Model seeds:",
        result[
            "model_seeds"
        ],
    )

    print(
        "Test bank used:",
        result[
            "test_bank_used"
        ],
    )

    assert (
        result[
            "training_episodes"
        ]
        == 25
    )

    assert (
        result[
            "validation_episodes"
        ]
        == 100
    )

    assert (
        result[
            "test_bank_used"
        ]
        is False
    )

    accuracy = float(
        result[
            "validation_accuracy_mean"
        ]
    )

    assert 0.0 <= accuracy <= 1.0

    print()
    print(
        "BASELINE ADAPTER SMOKE TEST: PASS"
    )


if __name__ == "__main__":
    main()
