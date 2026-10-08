from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


EXPECTED = {
    "train": 560,
    "validation": 120,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--oracle-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_payload = json.loads(
        args.index.read_text(
            encoding="utf-8"
        )
    )

    rows = []

    for task in index_payload["tasks"]:
        split = str(task["split"])

        if split not in EXPECTED:
            continue

        task_id = str(
            task["task_id"]
        )

        dataset = str(
            task["dataset"]
        )

        label_path = (
            args.oracle_root
            / split
            / dataset
            / task_id
            / "final_label.json"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing final label: {label_path}"
            )

        label = json.loads(
            label_path.read_text(
                encoding="utf-8"
            )
        )

        if label["task_id"] != task_id:
            raise RuntimeError(
                f"Task mismatch in {label_path}: "
                f"{label['task_id']} != {task_id}"
            )

        if label["dataset"] != dataset:
            raise RuntimeError(
                f"Dataset mismatch in {label_path}"
            )

        if label["split"] != split:
            raise RuntimeError(
                f"Split mismatch in {label_path}"
            )

        if label.get(
            "test_episode_bank_used",
            False,
        ):
            raise RuntimeError(
                f"Test episode bank used by {task_id}"
            )

        robust_seed_set = label.get(
            "robust_seed_set"
        )

        if robust_seed_set != [
            101,
            202,
            303,
        ]:
            raise RuntimeError(
                f"Unexpected robust seeds for "
                f"{task_id}: {robust_seed_set}"
            )

        candidate_count = int(
            label["candidate_count"]
        )

        oracle_equivalent_ids = (
            label[
                "oracle_equivalent_config_ids"
            ]
        )

        candidate_ids = label[
            "candidate_ids"
        ]

        best_config_id = str(
            label["best_config_id"]
        )

        if candidate_count != len(
            candidate_ids
        ):
            raise RuntimeError(
                f"Candidate count mismatch for "
                f"{task_id}"
            )

        if best_config_id not in (
            candidate_ids
        ):
            raise RuntimeError(
                f"Best configuration missing "
                f"from candidate set for "
                f"{task_id}"
            )

        if best_config_id not in (
            oracle_equivalent_ids
        ):
            raise RuntimeError(
                f"Best configuration not marked "
                f"oracle-equivalent for {task_id}"
            )

        rows.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "meta_split": split,
                "regime": str(
                    task["regime"]
                ),
                "n_way": int(
                    label["n_way"]
                ),
                "n_shot": int(
                    label["n_shot"]
                ),
                "n_query": int(
                    label["n_query"]
                ),
                "best_config_id": (
                    best_config_id
                ),
                "best_robust_validation_accuracy": float(
                    label[
                        "best_robust_validation_accuracy"
                    ]
                ),
                "best_robust_validation_seed_std": float(
                    label[
                        "best_robust_validation_seed_std"
                    ]
                ),
                "candidate_count": (
                    candidate_count
                ),
                "oracle_equivalent_count": int(
                    label[
                        "oracle_equivalent_count"
                    ]
                ),
                "oracle_equivalent_config_ids_json": (
                    json.dumps(
                        oracle_equivalent_ids
                    )
                ),
                "candidate_ids_json": (
                    json.dumps(
                        candidate_ids
                    )
                ),
                "final_label_path": str(
                    label_path
                ),
            }
        )

    table = pd.DataFrame(rows)

    if table["task_id"].duplicated().any():
        duplicates = table.loc[
            table["task_id"].duplicated(
                keep=False
            ),
            "task_id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate oracle labels: "
            f"{duplicates[:10]}"
        )

    if len(table) != 680:
        raise RuntimeError(
            f"Expected 680 labels, "
            f"found {len(table)}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_counts = {}

    for split, expected_count in (
        EXPECTED.items()
    ):
        subset = table[
            table["meta_split"]
            == split
        ].copy()

        split_counts[split] = int(
            len(subset)
        )

        if len(subset) != (
            expected_count
        ):
            raise RuntimeError(
                f"Expected {expected_count} "
                f"{split} labels, found "
                f"{len(subset)}"
            )

        subset = subset.sort_values(
            "task_id"
        ).reset_index(
            drop=True
        )

        subset.to_csv(
            args.output_dir
            / f"{split}_labels.csv",
            index=False,
        )

    table = table.sort_values(
        [
            "meta_split",
            "dataset",
            "regime",
            "task_id",
        ]
    ).reset_index(drop=True)

    table.to_csv(
        args.output_dir
        / "all680_labels.csv",
        index=False,
    )

    winner_counts = Counter(
        table["best_config_id"]
    )

    candidate_counts = (
        table["candidate_count"]
    )

    equivalent_counts = (
        table["oracle_equivalent_count"]
    )

    summary = {
        "schema_version": 1,
        "task_count": int(
            len(table)
        ),
        "split_counts": (
            split_counts
        ),
        "dataset_counts": {
            str(key): int(value)
            for key, value
            in Counter(
                table["dataset"]
            ).items()
        },
        "regime_counts": {
            str(key): int(value)
            for key, value
            in Counter(
                table["regime"]
            ).items()
        },
        "unique_winning_configuration_count": int(
            len(winner_counts)
        ),
        "winning_configuration_counts": {
            str(key): int(value)
            for key, value
            in winner_counts.most_common()
        },
        "mean_candidate_count": float(
            candidate_counts.mean()
        ),
        "minimum_candidate_count": int(
            candidate_counts.min()
        ),
        "maximum_candidate_count": int(
            candidate_counts.max()
        ),
        "mean_oracle_equivalent_count": float(
            equivalent_counts.mean()
        ),
        "maximum_oracle_equivalent_count": int(
            equivalent_counts.max()
        ),
        "test_oracles_included": False,
    }

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Labels:", len(table))
    print(
        "Split counts:",
        summary["split_counts"],
    )
    print(
        "Unique winning configurations:",
        summary[
            "unique_winning_configuration_count"
        ],
    )
    print(
        "Mean candidate count:",
        f"{summary['mean_candidate_count']:.3f}",
    )
    print(
        "Mean oracle-equivalent count:",
        f"{summary['mean_oracle_equivalent_count']:.3f}",
    )
    print(
        "Test oracles included:",
        False,
    )

    print()
    print(
        "Winning configuration counts:"
    )

    for config_id, count in (
        winner_counts.most_common()
    ):
        print(
            f"  {config_id:16s} "
            f"{count:4d} "
            f"({100 * count / len(table):6.2f}%)"
        )

    print()
    print(
        "Outputs:"
    )
    print(
        " ",
        args.output_dir
        / "train_labels.csv",
    )
    print(
        " ",
        args.output_dir
        / "validation_labels.csv",
    )
    print(
        " ",
        args.output_dir
        / "all680_labels.csv",
    )
    print(
        " ",
        summary_path,
    )

    print()
    print(
        "FINAL ORACLE LABEL MERGE: PASS"
    )


if __name__ == "__main__":
    main()
