from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    kendalltau,
    spearmanr,
)


EXPECTED_TASKS = 10
EXPECTED_CONFIGS = 64

BASE_SEED = 101
EXTRA_SEEDS = (202, 303)

EQUIVALENCE_EPSILON = 0.002
CANDIDATE_K_VALUES = (4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-selection",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed101-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--extra-seed-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def parse_json_vector(
    value: object,
    *,
    expected_length: int,
    field_name: str,
) -> np.ndarray:
    if pd.isna(value):
        raise ValueError(
            f"Missing {field_name}"
        )

    parsed = json.loads(
        str(value)
    )

    vector = np.asarray(
        parsed,
        dtype=np.float64,
    )

    if vector.shape != (
        expected_length,
    ):
        raise ValueError(
            f"{field_name} has shape "
            f"{vector.shape}; expected "
            f"({expected_length},)"
        )

    if not np.isfinite(vector).all():
        raise FloatingPointError(
            f"Non-finite values in "
            f"{field_name}"
        )

    return vector


def load_table(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    table = pd.read_csv(path)

    if len(table) != EXPECTED_CONFIGS:
        raise RuntimeError(
            f"{path} contains {len(table)} rows; "
            f"expected {EXPECTED_CONFIGS}"
        )

    if table["config_id"].nunique() != (
        EXPECTED_CONFIGS
    ):
        raise RuntimeError(
            f"{path} does not contain "
            f"{EXPECTED_CONFIGS} unique configs"
        )

    return table


def rank_ids(
    scores: pd.Series,
) -> list[str]:
    ranking_table = pd.DataFrame(
        {
            "config_id": scores.index,
            "score": scores.to_numpy(
                dtype=np.float64
            ),
        }
    )

    ranking_table = (
        ranking_table.sort_values(
            by=[
                "score",
                "config_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return [
        str(value)
        for value
        in ranking_table["config_id"]
    ]


def finite_correlation(
    value: float,
    *,
    name: str,
    task_id: str,
) -> float:
    if not np.isfinite(value):
        raise FloatingPointError(
            f"Invalid {name} correlation "
            f"for {task_id}"
        )

    return float(value)


def main() -> None:
    args = parse_args()

    selection = json.loads(
        args.task_selection.read_text(
            encoding="utf-8"
        )
    )

    tasks = selection["tasks"]

    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} tasks, "
            f"found {len(tasks)}"
        )

    task_rows: list[
        dict[str, object]
    ] = []

    candidate_rows: list[
        dict[str, object]
    ] = []

    for task in tasks:
        task_id = str(
            task["task_id"]
        )

        seed101_path = (
            args.seed101_root
            / task_id
            / "budget_200.csv"
        )

        extra_path = (
            args.extra_seed_root
            / task_id
            / "seeds_202_303.csv"
        )

        seed101_table = load_table(
            seed101_path
        )

        extra_table = load_table(
            extra_path
        )

        seed101_ids = set(
            seed101_table["config_id"]
        )

        extra_ids = set(
            extra_table["config_id"]
        )

        if seed101_ids != extra_ids:
            raise RuntimeError(
                f"Configuration mismatch for "
                f"{task_id}"
            )

        seed101_table = (
            seed101_table
            .set_index("config_id")
            .sort_index()
        )

        extra_table = (
            extra_table
            .set_index("config_id")
            .sort_index()
        )

        records = []

        for config_id in (
            seed101_table.index
        ):
            seed101_values = (
                parse_json_vector(
                    seed101_table.loc[
                        config_id,
                        "validation_accuracies_by_seed_json",
                    ],
                    expected_length=1,
                    field_name=(
                        "seed101 validation "
                        "accuracies"
                    ),
                )
            )

            extra_values = (
                parse_json_vector(
                    extra_table.loc[
                        config_id,
                        "validation_accuracies_by_seed_json",
                    ],
                    expected_length=2,
                    field_name=(
                        "extra-seed validation "
                        "accuracies"
                    ),
                )
            )

            if not np.isclose(
                float(seed101_values[0]),
                float(
                    seed101_table.loc[
                        config_id,
                        "validation_accuracy_mean",
                    ]
                ),
                atol=1e-10,
                rtol=1e-8,
            ):
                raise RuntimeError(
                    f"Seed-101 mean mismatch "
                    f"for {task_id}/"
                    f"{config_id}"
                )

            records.append(
                {
                    "config_id": str(
                        config_id
                    ),
                    "seed_101": float(
                        seed101_values[0]
                    ),
                    "seed_202": float(
                        extra_values[0]
                    ),
                    "seed_303": float(
                        extra_values[1]
                    ),
                }
            )

        performance = (
            pd.DataFrame(records)
            .set_index("config_id")
            .sort_index()
        )

        performance["mean_101_202"] = (
            performance[
                [
                    "seed_101",
                    "seed_202",
                ]
            ].mean(axis=1)
        )

        performance["mean_3_seeds"] = (
            performance[
                [
                    "seed_101",
                    "seed_202",
                    "seed_303",
                ]
            ].mean(axis=1)
        )

        performance["seed_std"] = (
            performance[
                [
                    "seed_101",
                    "seed_202",
                    "seed_303",
                ]
            ].std(
                axis=1,
                ddof=1,
            )
        )

        seed101_ranking = rank_ids(
            performance["seed_101"]
        )

        two_seed_ranking = rank_ids(
            performance["mean_101_202"]
        )

        robust_ranking = rank_ids(
            performance["mean_3_seeds"]
        )

        seed101_best_id = (
            seed101_ranking[0]
        )

        two_seed_best_id = (
            two_seed_ranking[0]
        )

        robust_best_id = (
            robust_ranking[0]
        )

        robust_best_score = float(
            performance.loc[
                robust_best_id,
                "mean_3_seeds",
            ]
        )

        seed101_winner_score = float(
            performance.loc[
                seed101_best_id,
                "mean_3_seeds",
            ]
        )

        two_seed_winner_score = float(
            performance.loc[
                two_seed_best_id,
                "mean_3_seeds",
            ]
        )

        seed101_regret = (
            robust_best_score
            - seed101_winner_score
        )

        two_seed_regret = (
            robust_best_score
            - two_seed_winner_score
        )

        spearman = finite_correlation(
            spearmanr(
                performance[
                    "seed_101"
                ],
                performance[
                    "mean_3_seeds"
                ],
            ).statistic,
            name="Spearman",
            task_id=task_id,
        )

        kendall = finite_correlation(
            kendalltau(
                performance[
                    "seed_101"
                ],
                performance[
                    "mean_3_seeds"
                ],
            ).statistic,
            name="Kendall",
            task_id=task_id,
        )

        seed101_top5 = set(
            seed101_ranking[:5]
        )

        robust_top5 = set(
            robust_ranking[:5]
        )

        task_rows.append(
            {
                "task_id": task_id,
                "dataset": (
                    task["dataset"]
                ),
                "regime": (
                    task["regime"]
                ),
                "n_way": (
                    task["n_way"]
                ),
                "n_shot": (
                    task["n_shot"]
                ),
                "seed101_best_config_id": (
                    seed101_best_id
                ),
                "two_seed_best_config_id": (
                    two_seed_best_id
                ),
                "robust_best_config_id": (
                    robust_best_id
                ),
                "seed101_top1_agreement": bool(
                    seed101_best_id
                    == robust_best_id
                ),
                "two_seed_top1_agreement": bool(
                    two_seed_best_id
                    == robust_best_id
                ),
                "seed101_spearman": (
                    spearman
                ),
                "seed101_kendall": (
                    kendall
                ),
                "seed101_top5_overlap": (
                    len(
                        seed101_top5
                        & robust_top5
                    )
                    / 5.0
                ),
                "seed101_winner_regret": float(
                    seed101_regret
                ),
                "seed101_winner_regret_pp": float(
                    100 * seed101_regret
                ),
                "seed101_winner_equivalent": bool(
                    seed101_regret
                    <= EQUIVALENCE_EPSILON
                    + 1e-12
                ),
                "two_seed_winner_regret": float(
                    two_seed_regret
                ),
                "two_seed_winner_regret_pp": float(
                    100 * two_seed_regret
                ),
                "two_seed_winner_equivalent": bool(
                    two_seed_regret
                    <= EQUIVALENCE_EPSILON
                    + 1e-12
                ),
                "mean_config_seed_std": float(
                    performance[
                        "seed_std"
                    ].mean()
                ),
                "maximum_config_seed_std": float(
                    performance[
                        "seed_std"
                    ].max()
                ),
            }
        )

        seed101_best_score = float(
            performance.loc[
                seed101_best_id,
                "seed_101",
            ]
        )

        tied_ids = set(
            performance.index[
                (
                    seed101_best_score
                    - performance[
                        "seed_101"
                    ]
                )
                <= EQUIVALENCE_EPSILON
                + 1e-12
            ]
        )

        for candidate_k in (
            CANDIDATE_K_VALUES
        ):
            candidate_ids = (
                set(
                    seed101_ranking[
                        :candidate_k
                    ]
                )
                | tied_ids
            )

            candidate_performance = (
                performance.loc[
                    sorted(candidate_ids)
                ]
            )

            candidate_ranking = rank_ids(
                candidate_performance[
                    "mean_3_seeds"
                ]
            )

            candidate_best_id = (
                candidate_ranking[0]
            )

            candidate_best_score = float(
                performance.loc[
                    candidate_best_id,
                    "mean_3_seeds",
                ]
            )

            candidate_regret = (
                robust_best_score
                - candidate_best_score
            )

            candidate_rows.append(
                {
                    "task_id": task_id,
                    "dataset": (
                        task["dataset"]
                    ),
                    "regime": (
                        task["regime"]
                    ),
                    "candidate_k": (
                        candidate_k
                    ),
                    "candidate_count": len(
                        candidate_ids
                    ),
                    "seed101_tie_count": len(
                        tied_ids
                    ),
                    "robust_best_config_id": (
                        robust_best_id
                    ),
                    "candidate_best_config_id": (
                        candidate_best_id
                    ),
                    "exact_robust_best_in_candidates": bool(
                        robust_best_id
                        in candidate_ids
                    ),
                    "candidate_oracle_regret": float(
                        candidate_regret
                    ),
                    "candidate_oracle_regret_pp": float(
                        100
                        * candidate_regret
                    ),
                    "candidate_oracle_equivalent": bool(
                        candidate_regret
                        <= EQUIVALENCE_EPSILON
                        + 1e-12
                    ),
                }
            )

    task_table = pd.DataFrame(
        task_rows
    )

    candidate_table = pd.DataFrame(
        candidate_rows
    )

    if len(task_table) != EXPECTED_TASKS:
        raise RuntimeError(
            "Unexpected task-analysis row count"
        )

    if len(candidate_table) != (
        EXPECTED_TASKS
        * len(CANDIDATE_K_VALUES)
    ):
        raise RuntimeError(
            "Unexpected candidate-analysis "
            "row count"
        )

    single_seed_summary = {
        "median_spearman": float(
            task_table[
                "seed101_spearman"
            ].median()
        ),
        "minimum_spearman": float(
            task_table[
                "seed101_spearman"
            ].min()
        ),
        "median_kendall": float(
            task_table[
                "seed101_kendall"
            ].median()
        ),
        "median_top5_overlap": float(
            task_table[
                "seed101_top5_overlap"
            ].median()
        ),
        "top1_agreement_fraction": float(
            task_table[
                "seed101_top1_agreement"
            ].mean()
        ),
        "equivalent_winner_fraction": float(
            task_table[
                "seed101_winner_equivalent"
            ].mean()
        ),
        "median_winner_regret_pp": float(
            task_table[
                "seed101_winner_regret_pp"
            ].median()
        ),
        "maximum_winner_regret_pp": float(
            task_table[
                "seed101_winner_regret_pp"
            ].max()
        ),
    }

    single_seed_summary[
        "passes_gate"
    ] = bool(
        single_seed_summary[
            "median_spearman"
        ] >= 0.90
        and single_seed_summary[
            "minimum_spearman"
        ] >= 0.80
        and single_seed_summary[
            "median_top5_overlap"
        ] >= 0.60
        and single_seed_summary[
            "equivalent_winner_fraction"
        ] >= 0.90
        and single_seed_summary[
            "maximum_winner_regret_pp"
        ] <= 0.75
    )

    two_seed_summary = {
        "top1_agreement_fraction": float(
            task_table[
                "two_seed_top1_agreement"
            ].mean()
        ),
        "equivalent_winner_fraction": float(
            task_table[
                "two_seed_winner_equivalent"
            ].mean()
        ),
        "median_winner_regret_pp": float(
            task_table[
                "two_seed_winner_regret_pp"
            ].median()
        ),
        "maximum_winner_regret_pp": float(
            task_table[
                "two_seed_winner_regret_pp"
            ].max()
        ),
    }

    candidate_summaries = []

    for candidate_k in (
        CANDIDATE_K_VALUES
    ):
        subset = candidate_table[
            candidate_table[
                "candidate_k"
            ]
            == candidate_k
        ]

        candidate_summary = {
            "candidate_k": (
                candidate_k
            ),
            "mean_candidate_count": float(
                subset[
                    "candidate_count"
                ].mean()
            ),
            "median_candidate_count": float(
                subset[
                    "candidate_count"
                ].median()
            ),
            "maximum_candidate_count": int(
                subset[
                    "candidate_count"
                ].max()
            ),
            "exact_robust_best_recall": float(
                subset[
                    "exact_robust_best_in_candidates"
                ].mean()
            ),
            "robust_equivalent_coverage": float(
                subset[
                    "candidate_oracle_equivalent"
                ].mean()
            ),
            "mean_candidate_regret_pp": float(
                subset[
                    "candidate_oracle_regret_pp"
                ].mean()
            ),
            "maximum_candidate_regret_pp": float(
                subset[
                    "candidate_oracle_regret_pp"
                ].max()
            ),
        }

        candidate_summary[
            "passes_gate"
        ] = bool(
            candidate_summary[
                "robust_equivalent_coverage"
            ] == 1.0
            and candidate_summary[
                "maximum_candidate_regret_pp"
            ] <= (
                100
                * EQUIVALENCE_EPSILON
                + 1e-10
            )
        )

        candidate_summaries.append(
            candidate_summary
        )

    passing_candidate_k = [
        item["candidate_k"]
        for item in candidate_summaries
        if item["passes_gate"]
    ]

    selected_candidate_k = (
        min(passing_candidate_k)
        if passing_candidate_k
        else None
    )

    if single_seed_summary[
        "passes_gate"
    ]:
        selected_strategy = (
            "single_seed_full64"
        )
    elif selected_candidate_k is not None:
        selected_strategy = (
            "seed101_full64_then_"
            "seeds202_303_candidates"
        )
    else:
        selected_strategy = (
            "three_seeds_full64"
        )

    summary = {
        "schema_version": 1,
        "task_count": EXPECTED_TASKS,
        "configuration_count": (
            EXPECTED_CONFIGS
        ),
        "training_episodes": 200,
        "validation_episodes": 100,
        "base_seed": BASE_SEED,
        "extra_seeds": list(
            EXTRA_SEEDS
        ),
        "equivalence_epsilon": (
            EQUIVALENCE_EPSILON
        ),
        "single_seed_summary": (
            single_seed_summary
        ),
        "two_seed_summary": (
            two_seed_summary
        ),
        "candidate_summaries": (
            candidate_summaries
        ),
        "selected_candidate_k": (
            selected_candidate_k
        ),
        "selected_final_oracle_strategy": (
            selected_strategy
        ),
    }

    for output_path in (
        args.task_output,
        args.candidate_output,
        args.summary_output,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    task_table.to_csv(
        args.task_output,
        index=False,
    )

    candidate_table.to_csv(
        args.candidate_output,
        index=False,
    )

    args.summary_output.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Tasks:", EXPECTED_TASKS)
    print(
        "Configurations:",
        EXPECTED_CONFIGS,
    )

    print()
    print("Single-seed stability:")
    print(
        "  Median Spearman:",
        f"{single_seed_summary['median_spearman']:.4f}",
    )
    print(
        "  Minimum Spearman:",
        f"{single_seed_summary['minimum_spearman']:.4f}",
    )
    print(
        "  Median top-5 overlap:",
        f"{single_seed_summary['median_top5_overlap']:.3f}",
    )
    print(
        "  Top-1 agreement:",
        f"{100 * single_seed_summary['top1_agreement_fraction']:.1f}%",
    )
    print(
        "  Equivalent winner:",
        f"{100 * single_seed_summary['equivalent_winner_fraction']:.1f}%",
    )
    print(
        "  Maximum winner regret:",
        f"{single_seed_summary['maximum_winner_regret_pp']:.3f} pp",
    )
    print(
        "  Gate:",
        (
            "PASS"
            if single_seed_summary[
                "passes_gate"
            ]
            else "FAIL"
        ),
    )

    print()
    print("Two-seed winner stability:")
    print(
        "  Top-1 agreement:",
        f"{100 * two_seed_summary['top1_agreement_fraction']:.1f}%",
    )
    print(
        "  Equivalent winner:",
        f"{100 * two_seed_summary['equivalent_winner_fraction']:.1f}%",
    )
    print(
        "  Maximum winner regret:",
        f"{two_seed_summary['maximum_winner_regret_pp']:.3f} pp",
    )

    print()
    print("Candidate-set coverage:")

    print(
        pd.DataFrame(
            candidate_summaries
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Selected candidate K:",
        selected_candidate_k,
    )
    print(
        "Selected final strategy:",
        selected_strategy,
    )

    print()
    print("Worst one-seed cases:")
    print(
        task_table.sort_values(
            "seed101_winner_regret_pp",
            ascending=False,
        )[
            [
                "dataset",
                "regime",
                "seed101_best_config_id",
                "robust_best_config_id",
                "seed101_spearman",
                "seed101_winner_regret_pp",
                "seed101_winner_equivalent",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Task output:", args.task_output)
    print(
        "Candidate output:",
        args.candidate_output,
    )
    print(
        "Summary output:",
        args.summary_output,
    )
    print("SEED-STABILITY ANALYSIS: PASS")


if __name__ == "__main__":
    main()
