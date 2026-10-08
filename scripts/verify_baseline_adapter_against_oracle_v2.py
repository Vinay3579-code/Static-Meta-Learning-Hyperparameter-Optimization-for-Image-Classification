from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sml_hpo.baselines.adapter_v2 import (
    build_default_adapter,
)


def main():
    adapter = build_default_adapter(
        device="cuda:0"
    )

    # Use one deterministic held-out task.
    task_id = adapter.task_ids(
        split="test",
        dataset="omniglot",
    )[0]

    task = adapter.get_task(
        task_id
    )

    stage1_path = (
        Path(
            "results/oracles/"
            "final800_v2/test"
        )
        / task.dataset
        / task.task_id
        / "stage1_seed101_full64.csv"
    )

    if not stage1_path.exists():
        raise FileNotFoundError(
            stage1_path
        )

    stage1 = pd.read_csv(
        stage1_path
    )

    if len(stage1) != 64:
        raise RuntimeError(
            "Expected 64 rows in "
            "stage1 oracle table."
        )

    config_id = (
        "anchor_v2_058"
    )

    reference_rows = stage1[
        stage1["config_id"]
        .astype(str)
        == config_id
    ]

    if len(reference_rows) != 1:
        raise RuntimeError(
            "Could not uniquely find "
            f"{config_id} in stage1."
        )

    reference_accuracy = float(
        reference_rows.iloc[0][
            "validation_accuracy_mean"
        ]
    )

    print(
        "Task:",
        task_id,
    )

    print(
        "Config:",
        config_id,
    )

    print(
        "Frozen stage1 accuracy:",
        reference_accuracy,
    )

    print()
    print(
        "Re-evaluating through "
        "baseline adapter..."
    )

    result = (
        adapter.train_validate_anchor(
            task_id=task_id,

            config_id=config_id,

            budget=200,

            model_seed=101,

            validation_episodes=100,
        )
    )

    adapter_accuracy = float(
        result[
            "validation_accuracy_mean"
        ]
    )

    difference = abs(
        adapter_accuracy
        - reference_accuracy
    )

    print()
    print(
        "Adapter accuracy:",
        adapter_accuracy,
    )

    print(
        "Absolute difference:",
        difference,
    )

    print(
        "Test bank used:",
        result[
            "test_bank_used"
        ],
    )

    assert (
        result[
            "test_bank_used"
        ]
        is False
    )

    # The entire pipeline is deterministic.
    # This should normally be effectively exact.
    if not np.isclose(
        adapter_accuracy,
        reference_accuracy,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Adapter result does not "
            "match frozen oracle result."
        )

    print()
    print(
        "BASELINE ADAPTER / ORACLE "
        "CONSISTENCY: PASS"
    )


if __name__ == "__main__":
    main()
