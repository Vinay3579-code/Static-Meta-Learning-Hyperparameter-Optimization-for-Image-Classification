from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


BUDGETS = (100, 200, 300)
REFERENCE_BUDGET = 300
EXPECTED_CONFIGS = 64
EQUIVALENCE_EPSILON = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-selection",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def load_budget_table(
    *,
    input_root: Path,
    task_id: str,
    budget: int,
) -> pd.DataFrame:
    path = (
        input_root
        / task_id
        / f"budget_{budget}.csv"
    )

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

    if not np.isfinite(
        table[
            "validation_accuracy_mean"
        ].to_numpy(dtype=np.float64)
    ).all():
        raise FloatingPointError(
            f"Non-finite accuracies in {path}"
        )

    return table


def ranking(
    table: pd.DataFrame,
) -> pd.DataFrame:
    return table.sort_values(
        by=[
            "validation_accuracy_mean",
            "validation_accuracy_seed_std",
            "total_elapsed_seconds",
            "config_id",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    selection = json.loads(
        args.task_selection.read_text(
            encoding="utf-8"
        )
    )

    tasks = selection["tasks"]

    task_rows = []

    for task in tasks:
        task_id = str(
            task["task_id"]
        )

        tables = {
            budget: load_budget_table(
                input_root=args.input_root,
                task_id=task_id,
                budget=budget,
            )
            for budget in BUDGETS
        }

        config_sets = {
            frozenset(
                table["config_id"]
            )
            for table in tables.values()
        }

        if len(config_sets) != 1:
            raise RuntimeError(
                f"Configuration mismatch for {task_id}"
            )

        reference = (
            tables[REFERENCE_BUDGET]
            .set_index("config_id")
        )

        reference_ranking = ranking(
            tables[REFERENCE_BUDGET]
        )

        reference_best_id = str(
            reference_ranking.iloc[0][
                "config_id"
            ]
        )

        reference_best_accuracy = float(
            reference_ranking.iloc[0][
                "validation_accuracy_mean"
            ]
        )

        reference_top5 = set(
            reference_ranking[
                "config_id"
            ].head(5)
        )

        for budget in BUDGETS:
            candidate = (
                tables[budget]
                .set_index("config_id")
            )

            candidate_ranking = ranking(
                tables[budget]
            )

            candidate_best_id = str(
                candidate_ranking.iloc[0][
                    "config_id"
                ]
            )

            candidate_top5 = set(
                candidate_ranking[
                    "config_id"
                ].head(5)
            )

            ordered_ids = sorted(
                reference.index
            )

            candidate_values = (
                candidate.loc[
                    ordered_ids,
                    "validation_accuracy_mean",
                ].to_numpy(
                    dtype=np.float64
                )
            )

            reference_values = (
                reference.loc[
                    ordered_ids,
                    "validation_accuracy_mean",
                ].to_numpy(
                    dtype=np.float64
                )
            )

            spearman_value = float(
                spearmanr(
                    candidate_values,
                    reference_values,
                ).statistic
            )

            kendall_value = float(
                kendalltau(
                    candidate_values,
                    reference_values,
                ).statistic
            )

            if not np.isfinite(
                spearman_value
            ):
                raise FloatingPointError(
                    f"Invalid Spearman correlation "
                    f"for {task_id}, budget={budget}"
                )

            if not np.isfinite(
                kendall_value
            ):
                raise FloatingPointError(
                    f"Invalid Kendall correlation "
                    f"for {task_id}, budget={budget}"
                )

            final_accuracy_of_budget_winner = float(
                reference.loc[
                    candidate_best_id,
                    "validation_accuracy_mean",
                ]
            )

            winner_regret = (
                reference_best_accuracy
                - final_accuracy_of_budget_winner
            )

            task_rows.append(
                {
                    "task_id": task_id,
                    "dataset": task["dataset"],
                    "regime": task["regime"],
                    "n_way": task["n_way"],
                    "n_shot": task["n_shot"],
                    "budget": budget,
                    "reference_budget": (
                        REFERENCE_BUDGET
                    ),
                    "candidate_best_config_id": (
                        candidate_best_id
                    ),
                    "reference_best_config_id": (
                        reference_best_id
                    ),
                    "top1_agreement": bool(
                        candidate_best_id
                        == reference_best_id
                    ),
                    "top5_overlap_fraction": (
                        len(
                            candidate_top5
                            & reference_top5
                        )
                        / 5.0
                    ),
                    "spearman_correlation": (
                        spearman_value
                    ),
                    "kendall_correlation": (
                        kendall_value
                    ),
                    "reference_best_accuracy": (
                        reference_best_accuracy
                    ),
                    "reference_accuracy_of_candidate_winner": (
                        final_accuracy_of_budget_winner
                    ),
                    "winner_regret": float(
                        winner_regret
                    ),
                    "winner_regret_pp": float(
                        100 * winner_regret
                    ),
                    "winner_final_equivalent": bool(
                        winner_regret
                        <= EQUIVALENCE_EPSILON
                        + 1e-12
                    ),
                }
            )

    task_table = pd.DataFrame(
        task_rows
    )

    if len(task_table) != (
        len(tasks) * len(BUDGETS)
    ):
        raise RuntimeError(
            "Unexpected analysis row count"
        )

    candidate_budgets = (
        100,
        200,
    )

    budget_summaries = []

    for budget in candidate_budgets:
        subset = task_table[
            task_table["budget"] == budget
        ]

        summary = {
            "budget": budget,
            "task_count": int(
                len(subset)
            ),
            "median_spearman": float(
                subset[
                    "spearman_correlation"
                ].median()
            ),
            "minimum_spearman": float(
                subset[
                    "spearman_correlation"
                ].min()
            ),
            "median_kendall": float(
                subset[
                    "kendall_correlation"
                ].median()
            ),
            "median_top5_overlap": float(
                subset[
                    "top5_overlap_fraction"
                ].median()
            ),
            "top1_agreement_fraction": float(
                subset[
                    "top1_agreement"
                ].mean()
            ),
            "final_equivalent_fraction": float(
                subset[
                    "winner_final_equivalent"
                ].mean()
            ),
            "median_winner_regret_pp": float(
                subset[
                    "winner_regret_pp"
                ].median()
            ),
            "maximum_winner_regret_pp": float(
                subset[
                    "winner_regret_pp"
                ].max()
            ),
        }

        summary["passes_gate"] = bool(
            summary["median_spearman"]
            >= 0.90
            and summary["minimum_spearman"]
            >= 0.80
            and summary["median_top5_overlap"]
            >= 0.60
            and summary[
                "final_equivalent_fraction"
            ]
            >= 0.80
            and summary[
                "median_winner_regret_pp"
            ]
            <= 0.20
            and summary[
                "maximum_winner_regret_pp"
            ]
            <= 0.75
        )

        budget_summaries.append(
            summary
        )

    passing_budgets = [
        summary["budget"]
        for summary in budget_summaries
        if summary["passes_gate"]
    ]

    selected_budget = (
        min(passing_budgets)
        if passing_budgets
        else REFERENCE_BUDGET
    )

    final_summary = {
        "schema_version": 1,
        "task_count": len(tasks),
        "configuration_count": (
            EXPECTED_CONFIGS
        ),
        "candidate_budgets": list(
            candidate_budgets
        ),
        "reference_budget": (
            REFERENCE_BUDGET
        ),
        "equivalence_epsilon": (
            EQUIVALENCE_EPSILON
        ),
        "gate": {
            "median_spearman_minimum": 0.90,
            "minimum_spearman_minimum": 0.80,
            "median_top5_overlap_minimum": 0.60,
            "final_equivalent_fraction_minimum": (
                0.80
            ),
            "median_winner_regret_pp_maximum": (
                0.20
            ),
            "maximum_winner_regret_pp_maximum": (
                0.75
            ),
        },
        "budget_summaries": (
            budget_summaries
        ),
        "selected_training_budget": int(
            selected_budget
        ),
        "screening_model_seeds": (
            selection.get(
                "model_seeds_screening",
                [101],
            )
        ),
    }

    args.task_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_table.to_csv(
        args.task_output,
        index=False,
    )

    args.summary_output.write_text(
        json.dumps(
            final_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Tasks:", len(tasks))
    print(
        "Configurations:",
        EXPECTED_CONFIGS,
    )
    print(
        "Reference budget:",
        REFERENCE_BUDGET,
    )

    for summary in budget_summaries:
        print()
        print(
            "Budget:",
            summary["budget"],
        )
        print(
            "  Median Spearman:",
            f"{summary['median_spearman']:.4f}",
        )
        print(
            "  Minimum Spearman:",
            f"{summary['minimum_spearman']:.4f}",
        )
        print(
            "  Median Kendall:",
            f"{summary['median_kendall']:.4f}",
        )
        print(
            "  Median top-5 overlap:",
            f"{summary['median_top5_overlap']:.3f}",
        )
        print(
            "  Top-1 agreement:",
            f"{100 * summary['top1_agreement_fraction']:.1f}%",
        )
        print(
            "  Final-equivalent winner:",
            f"{100 * summary['final_equivalent_fraction']:.1f}%",
        )
        print(
            "  Median winner regret:",
            f"{summary['median_winner_regret_pp']:.3f} pp",
        )
        print(
            "  Maximum winner regret:",
            f"{summary['maximum_winner_regret_pp']:.3f} pp",
        )
        print(
            "  Gate:",
            (
                "PASS"
                if summary["passes_gate"]
                else "FAIL"
            ),
        )

    print()
    print(
        "Selected training budget:",
        selected_budget,
    )

    print()
    print("Worst 100-episode cases:")
    print(
        task_table[
            task_table["budget"] == 100
        ].sort_values(
            "winner_regret_pp",
            ascending=False,
        )[
            [
                "dataset",
                "regime",
                "spearman_correlation",
                "top5_overlap_fraction",
                "candidate_best_config_id",
                "reference_best_config_id",
                "winner_regret_pp",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("Worst 200-episode cases:")
    print(
        task_table[
            task_table["budget"] == 200
        ].sort_values(
            "winner_regret_pp",
            ascending=False,
        )[
            [
                "dataset",
                "regime",
                "spearman_correlation",
                "top5_overlap_fraction",
                "candidate_best_config_id",
                "reference_best_config_id",
                "winner_regret_pp",
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("Task output:", args.task_output)
    print(
        "Summary output:",
        args.summary_output,
    )
    print("BUDGET-STABILITY ANALYSIS: PASS")


if __name__ == "__main__":
    main()
