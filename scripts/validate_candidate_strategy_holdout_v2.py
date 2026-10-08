from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TASKS = 20
EXPECTED_CONFIGS = 64
EXPECTED_SEEDS = (101, 202)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--screening-selection",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
    )

    return parser.parse_args()


def parse_vector(
    value: object,
    expected_length: int,
) -> np.ndarray:
    values = np.asarray(
        json.loads(str(value)),
        dtype=np.float64,
    )

    if values.shape != (expected_length,):
        raise ValueError(
            f"Expected vector length {expected_length}, "
            f"received shape {values.shape}"
        )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite seed accuracy detected"
        )

    return values


def rank_configs(
    scores: pd.Series,
) -> list[str]:
    ranking = pd.DataFrame(
        {
            "config_id": scores.index,
            "score": scores.to_numpy(
                dtype=np.float64
            ),
        }
    )

    ranking = ranking.sort_values(
        by=["score", "config_id"],
        ascending=[False, True],
    )

    return ranking["config_id"].astype(
        str
    ).tolist()


def discover_primary_tables(
    root: Path,
) -> list[Path]:
    primary_tables: list[Path] = []

    for path in sorted(root.glob("*.csv")):
        if path.stem.endswith("_ranked"):
            continue

        table = pd.read_csv(path)

        required = {
            "task_id",
            "dataset",
            "config_id",
            "model_seeds_json",
            "validation_accuracies_by_seed_json",
        }

        if not required <= set(table.columns):
            continue

        if (
            len(table) == EXPECTED_CONFIGS
            and table["config_id"].nunique()
            == EXPECTED_CONFIGS
            and table["task_id"].nunique() == 1
        ):
            primary_tables.append(path)

    return primary_tables


def main() -> None:
    args = parse_args()

    if args.epsilon < 0:
        raise ValueError(
            "epsilon must be non-negative"
        )

    k_values = sorted(
        set(args.k_values)
    )

    if not k_values or min(k_values) <= 0:
        raise ValueError(
            "All K values must be positive"
        )

    screening_payload = json.loads(
        args.screening_selection.read_text(
            encoding="utf-8"
        )
    )

    screening_task_ids = {
        str(task["task_id"])
        for task in screening_payload["tasks"]
    }

    table_paths = discover_primary_tables(
        args.pilot_root
    )

    if len(table_paths) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} primary pilot tables, "
            f"found {len(table_paths)}"
        )

    rows: list[dict[str, object]] = []
    holdout_task_ids: set[str] = set()

    for path in table_paths:
        table = pd.read_csv(path)

        task_id = str(
            table.iloc[0]["task_id"]
        )

        dataset = str(
            table.iloc[0]["dataset"]
        )

        if task_id in holdout_task_ids:
            raise RuntimeError(
                f"Duplicate task: {task_id}"
            )

        holdout_task_ids.add(task_id)

        seed_lists = {
            tuple(
                int(seed)
                for seed in json.loads(
                    str(value)
                )
            )
            for value in table[
                "model_seeds_json"
            ]
        }

        if seed_lists != {
            EXPECTED_SEEDS
        }:
            raise RuntimeError(
                f"{task_id} uses unexpected seeds: "
                f"{seed_lists}"
            )

        table = table.set_index(
            "config_id"
        ).sort_index()

        records = []

        for config_id, row in table.iterrows():
            values = parse_vector(
                row[
                    "validation_accuracies_by_seed_json"
                ],
                expected_length=2,
            )

            records.append(
                {
                    "config_id": str(config_id),
                    "seed101": float(values[0]),
                    "seed202": float(values[1]),
                    "robust_two_seed": float(
                        values.mean()
                    ),
                }
            )

        performance = (
            pd.DataFrame(records)
            .set_index("config_id")
            .sort_index()
        )

        seed101_ranking = rank_configs(
            performance["seed101"]
        )

        robust_ranking = rank_configs(
            performance["robust_two_seed"]
        )

        seed101_best_id = (
            seed101_ranking[0]
        )

        robust_best_id = robust_ranking[0]

        seed101_best_score = float(
            performance.loc[
                seed101_best_id,
                "seed101",
            ]
        )

        robust_best_score = float(
            performance.loc[
                robust_best_id,
                "robust_two_seed",
            ]
        )

        tied_ids = {
            str(config_id)
            for config_id in performance.index[
                (
                    seed101_best_score
                    - performance["seed101"]
                )
                <= args.epsilon + 1e-12
            ]
        }

        for candidate_k in k_values:
            candidate_ids = (
                set(
                    seed101_ranking[
                        :candidate_k
                    ]
                )
                | tied_ids
            )

            candidate_scores = performance.loc[
                sorted(candidate_ids),
                "robust_two_seed",
            ]

            candidate_ranking = rank_configs(
                candidate_scores
            )

            candidate_best_id = (
                candidate_ranking[0]
            )

            candidate_best_score = float(
                performance.loc[
                    candidate_best_id,
                    "robust_two_seed",
                ]
            )

            regret = (
                robust_best_score
                - candidate_best_score
            )

            rows.append(
                {
                    "task_id": task_id,
                    "dataset": dataset,
                    "candidate_k": candidate_k,
                    "candidate_count": len(
                        candidate_ids
                    ),
                    "seed101_tie_count": len(
                        tied_ids
                    ),
                    "seed101_best_config_id": (
                        seed101_best_id
                    ),
                    "robust_best_config_id": (
                        robust_best_id
                    ),
                    "candidate_best_config_id": (
                        candidate_best_id
                    ),
                    "exact_robust_best_recalled": bool(
                        robust_best_id
                        in candidate_ids
                    ),
                    "candidate_regret": float(
                        regret
                    ),
                    "candidate_regret_pp": float(
                        100 * regret
                    ),
                    "candidate_equivalent": bool(
                        regret
                        <= args.epsilon + 1e-12
                    ),
                }
            )

    overlap = (
        screening_task_ids
        & holdout_task_ids
    )

    if overlap:
        raise RuntimeError(
            "Screening and holdout tasks overlap: "
            f"{sorted(overlap)}"
        )

    result = pd.DataFrame(rows)

    summaries = []

    for candidate_k in k_values:
        subset = result[
            result["candidate_k"]
            == candidate_k
        ]

        summary = {
            "candidate_k": int(
                candidate_k
            ),
            "task_count": int(
                len(subset)
            ),
            "mean_candidate_count": float(
                subset[
                    "candidate_count"
                ].mean()
            ),
            "maximum_candidate_count": int(
                subset[
                    "candidate_count"
                ].max()
            ),
            "exact_robust_best_recall": float(
                subset[
                    "exact_robust_best_recalled"
                ].mean()
            ),
            "equivalent_coverage": float(
                subset[
                    "candidate_equivalent"
                ].mean()
            ),
            "mean_regret_pp": float(
                subset[
                    "candidate_regret_pp"
                ].mean()
            ),
            "maximum_regret_pp": float(
                subset[
                    "candidate_regret_pp"
                ].max()
            ),
        }

        # Exact winner identity is tie-sensitive, so 90% recall is
        # sufficient when equivalent coverage remains perfect.
        summary["passes_gate"] = bool(
            summary[
                "exact_robust_best_recall"
            ] >= 0.90
            and summary[
                "equivalent_coverage"
            ] == 1.0
            and summary[
                "maximum_regret_pp"
            ] <= 100 * args.epsilon
            + 1e-10
        )

        summaries.append(summary)

    passing_k = [
        item["candidate_k"]
        for item in summaries
        if item["passes_gate"]
    ]

    selected_k = (
        min(passing_k)
        if passing_k
        else None
    )

    summary_payload = {
        "schema_version": 1,
        "validation_type": (
            "independent_two_seed_pilot_holdout"
        ),
        "holdout_task_count": (
            EXPECTED_TASKS
        ),
        "configuration_count": (
            EXPECTED_CONFIGS
        ),
        "screening_holdout_overlap_count": 0,
        "model_seeds": list(
            EXPECTED_SEEDS
        ),
        "equivalence_epsilon": (
            args.epsilon
        ),
        "candidate_summaries": summaries,
        "selected_candidate_k": (
            selected_k
        ),
        "passes_holdout_validation": bool(
            selected_k is not None
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    args.summary_output.write_text(
        json.dumps(
            summary_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Holdout tasks:", len(holdout_task_ids))
    print(
        "Screening/holdout overlap:",
        len(overlap),
    )
    print()

    print(
        pd.DataFrame(
            summaries
        ).to_string(index=False)
    )

    print()
    print(
        "Selected holdout-validated K:",
        selected_k,
    )
    print(
        "Candidate strategy holdout:",
        (
            "PASS"
            if selected_k is not None
            else "FAIL"
        ),
    )

    print()
    print("Worst K=4 cases:")
    print(
        result[
            result["candidate_k"] == 4
        ].sort_values(
            "candidate_regret_pp",
            ascending=False,
        )[
            [
                "dataset",
                "task_id",
                "candidate_count",
                "robust_best_config_id",
                "candidate_best_config_id",
                "exact_robust_best_recalled",
                "candidate_regret_pp",
                "candidate_equivalent",
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print()
    print("Output:", args.output)
    print(
        "Summary:",
        args.summary_output,
    )
    print(
        "CANDIDATE HOLDOUT VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
