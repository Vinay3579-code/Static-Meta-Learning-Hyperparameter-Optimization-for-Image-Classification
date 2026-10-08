from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INDEX = Path(
    "data/manifests/final800/index.json"
)

ROOT = Path(
    "results/oracles/final800_v2/test"
)

OUTPUT = Path(
    "results/oracles/"
    "final800_v2_merged/"
    "test_labels.csv"
)


def main():
    index = json.loads(
        INDEX.read_text(
            encoding="utf-8"
        )
    )

    tasks = [
        task
        for task in index["tasks"]
        if task["split"] == "test"
    ]

    if len(tasks) != 120:
        raise RuntimeError(
            f"Expected 120 test tasks, "
            f"found {len(tasks)}"
        )

    rows = []

    for task in tasks:
        task_id = str(
            task["task_id"]
        )

        dataset = str(
            task["dataset"]
        )

        path = (
            ROOT
            / dataset
            / task_id
            / "final_label.json"
        )

        label = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if label[
            "test_episode_bank_used"
        ]:
            raise RuntimeError(
                f"Oracle selection leaked "
                f"test bank: {task_id}"
            )

        rows.append({
            "task_id":
                task_id,

            "dataset":
                dataset,

            "meta_split":
                "test",

            "regime":
                str(task["regime"]),

            "n_way":
                int(label["n_way"]),

            "n_shot":
                int(label["n_shot"]),

            "n_query":
                int(label["n_query"]),

            "best_config_id":
                str(
                    label[
                        "best_config_id"
                    ]
                ),

            "best_robust_validation_accuracy":
                float(
                    label[
                        "best_robust_validation_accuracy"
                    ]
                ),

            "best_robust_validation_seed_std":
                float(
                    label[
                        "best_robust_validation_seed_std"
                    ]
                ),

            "candidate_count":
                int(
                    label[
                        "candidate_count"
                    ]
                ),

            "oracle_equivalent_count":
                int(
                    label[
                        "oracle_equivalent_count"
                    ]
                ),

            "oracle_equivalent_config_ids_json":
                json.dumps(
                    label[
                        "oracle_equivalent_config_ids"
                    ]
                ),

            "candidate_ids_json":
                json.dumps(
                    label[
                        "candidate_ids"
                    ]
                ),

            "final_label_path":
                str(path),
        })

    table = pd.DataFrame(
        rows
    )

    if len(table) != 120:
        raise RuntimeError(
            "Wrong output row count"
        )

    if table[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate test task IDs"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.sort_values(
        "task_id"
    ).to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "Test labels:",
        len(table),
    )

    print(
        "Unique hard winners:",
        table[
            "best_config_id"
        ].nunique(),
    )

    print(
        "Mean candidate count:",
        f"{table['candidate_count'].mean():.3f}",
    )

    print(
        "Mean equivalent count:",
        f"{table['oracle_equivalent_count'].mean():.3f}",
    )

    print(
        "Output:",
        OUTPUT,
    )

    print()
    print(
        "TEST ORACLE LABEL MERGE: PASS"
    )


if __name__ == "__main__":
    main()
