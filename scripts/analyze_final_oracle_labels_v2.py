from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_ROOT = Path(
    "results/oracles/final800_v2_merged"
)

OUTPUT_ROOT = Path(
    "results/oracles/final800_v2_analysis"
)


def entropy_from_counts(
    counts: Counter,
) -> float:
    values = np.asarray(
        list(counts.values()),
        dtype=np.float64,
    )

    probabilities = values / values.sum()

    return float(
        -np.sum(
            probabilities
            * np.log(
                probabilities + 1e-15
            )
        )
    )


def analyze_split(
    table: pd.DataFrame,
    split_name: str,
) -> dict:
    winners = Counter(
        table["best_config_id"]
    )

    candidate_counts = (
        table["candidate_count"]
        .to_numpy(dtype=np.int64)
    )

    equivalent_counts = (
        table["oracle_equivalent_count"]
        .to_numpy(dtype=np.int64)
    )

    result = {
        "task_count": int(
            len(table)
        ),
        "unique_winners": int(
            len(winners)
        ),
        "winner_entropy": (
            entropy_from_counts(
                winners
            )
        ),
        "winner_counts": {
            str(key): int(value)
            for key, value
            in winners.most_common()
        },
        "largest_winner_fraction": float(
            max(winners.values())
            / len(table)
        ),
        "mean_candidate_count": float(
            candidate_counts.mean()
        ),
        "maximum_candidate_count": int(
            candidate_counts.max()
        ),
        "mean_equivalent_count": float(
            equivalent_counts.mean()
        ),
        "maximum_equivalent_count": int(
            equivalent_counts.max()
        ),
    }

    print()
    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    print("Tasks:", result["task_count"])
    print(
        "Unique winners:",
        result["unique_winners"],
    )
    print(
        "Largest winner fraction:",
        f"{100 * result['largest_winner_fraction']:.1f}%",
    )
    print(
        "Mean candidate count:",
        f"{result['mean_candidate_count']:.3f}",
    )
    print(
        "Mean oracle-equivalent count:",
        f"{result['mean_equivalent_count']:.3f}",
    )

    print()
    print("Winning configurations:")

    for config_id, count in (
        winners.most_common()
    ):
        print(
            f"  {config_id:16s} "
            f"{count:4d} "
            f"({100 * count / len(table):6.2f}%)"
        )

    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        LABEL_ROOT
        / "train_labels.csv"
    )

    validation = pd.read_csv(
        LABEL_ROOT
        / "validation_labels.csv"
    )

    assert len(train) == 560
    assert len(validation) == 120

    assert set(
        train["task_id"]
    ).isdisjoint(
        set(validation["task_id"])
    )

    train_summary = analyze_split(
        train,
        "train",
    )

    validation_summary = analyze_split(
        validation,
        "validation",
    )

    print()
    print("=" * 70)
    print("DATASET × REGIME WINNERS")
    print("=" * 70)

    combined = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    grouped = (
        combined.groupby(
            [
                "meta_split",
                "dataset",
                "regime",
                "best_config_id",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    print(
        grouped.to_string(
            index=False
        )
    )

    grouped.to_csv(
        OUTPUT_ROOT
        / "winner_distribution.csv",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "train": train_summary,
        "validation": (
            validation_summary
        ),
    }

    (
        OUTPUT_ROOT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "FINAL ORACLE LABEL ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()
