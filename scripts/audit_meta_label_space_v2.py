from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/oracles/final800_v2_merged"
)


def parse_ids(value: str) -> list[str]:
    result = json.loads(value)

    if not isinstance(result, list):
        raise ValueError(
            "Expected JSON list"
        )

    return [
        str(item)
        for item in result
    ]


def analyze(
    table: pd.DataFrame,
    split_name: str,
) -> set[str]:
    hard_counts = Counter(
        table["best_config_id"]
    )

    equivalent_sets = [
        set(parse_ids(value))
        for value in table[
            "oracle_equivalent_config_ids_json"
        ]
    ]

    equivalent_union = set().union(
        *equivalent_sets
    )

    equivalent_frequency = Counter()

    for ids in equivalent_sets:
        equivalent_frequency.update(ids)

    print()
    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    print("Tasks:", len(table))

    print(
        "Hard-winner vocabulary:",
        len(hard_counts),
    )

    print(
        "Equivalent-label vocabulary:",
        len(equivalent_union),
    )

    print()
    print(
        "Hard winners:",
        sorted(hard_counts),
    )

    print()
    print(
        "Equivalent-label configurations:"
    )

    for config_id, count in (
        equivalent_frequency.most_common()
    ):
        print(
            f"  {config_id:16s} "
            f"{count:4d} "
            f"({100 * count / len(table):6.2f}%)"
        )

    return equivalent_union


def main() -> None:
    train = pd.read_csv(
        ROOT / "train_labels.csv"
    )

    validation = pd.read_csv(
        ROOT / "validation_labels.csv"
    )

    train_vocab = analyze(
        train,
        "train",
    )

    validation_vocab = analyze(
        validation,
        "validation",
    )

    unseen_validation = (
        validation_vocab
        - train_vocab
    )

    print()
    print("=" * 70)
    print("COVERAGE")
    print("=" * 70)

    print(
        "Train equivalent vocabulary:",
        len(train_vocab),
    )

    print(
        "Validation equivalent vocabulary:",
        len(validation_vocab),
    )

    print(
        "Validation configs unseen in training:",
        sorted(unseen_validation),
    )

    print()

    if unseen_validation:
        print(
            "LABEL SPACE COVERAGE: FAIL"
        )
    else:
        print(
            "LABEL SPACE COVERAGE: PASS"
        )


if __name__ == "__main__":
    main()
