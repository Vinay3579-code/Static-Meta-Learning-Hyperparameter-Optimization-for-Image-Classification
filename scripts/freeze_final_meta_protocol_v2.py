from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/meta_learning"
)

OUTPUT = Path(
    "configs/protocols/"
    "final_meta_learning_v2.json"
)


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main():
    zero = {}

    probe = {}

    for seed in [
        101,
        202,
        303,
    ]:
        zero[str(seed)] = read_json(
            ROOT
            / "zero_sml_v2"
            / f"seed{seed}"
            / "summary.json"
        )

        probe[str(seed)] = read_json(
            ROOT
            / "mlp_v2"
            / f"seed{seed}"
            / "summary.json"
        )

    baselines = pd.read_csv(
        ROOT
        / "baselines_v2"
        / "validation.csv"
    )

    classical = pd.read_csv(
        ROOT
        / "classical_v2"
        / "validation.csv"
    )

    comparison = read_json(
        ROOT
        / "zero_probe_comparison_v2"
        / "summary.json"
    )

    zero_epochs = {
        seed: int(
            result["best_epoch"]
        )
        for seed, result
        in zero.items()
    }

    probe_epochs = {
        seed: int(
            result["best_epoch"]
        )
        for seed, result
        in probe.items()
    }

    zero_rates = [
        float(
            result[
                "validation_oracle_equivalent_rate"
            ]
        )
        for result in zero.values()
    ]

    probe_rates = [
        float(
            result[
                "validation_oracle_equivalent_rate"
            ]
        )
        for result in probe.values()
    ]

    best_classical = (
        classical.sort_values(
            [
                "equivalent_rate",
                "hard_accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    protocol = {
        "schema_version": 2,

        "status": (
            "FROZEN_BEFORE_TEST_ORACLE"
        ),

        "test_oracle_labels_seen": False,

        "meta_training_tasks": 560,
        "meta_validation_tasks": 120,
        "final_refit_tasks": 680,
        "final_test_tasks": 120,

        "primary_method": {
            "name": (
                "Zero-SML MLP ensemble"
            ),

            "descriptor_dimension": 64,
            "protocol_dimension": 3,
            "input_dimension": 67,

            "protocol_features": [
                "n_way",
                "n_shot",
                "n_query",
            ],

            "dataset_identity_used": False,

            "output_semantics": (
                "oracle-equivalent "
                "configuration classification"
            ),

            "loss": (
                "soft-target cross entropy"
            ),

            "ensemble": (
                "three-seed deterministic "
                "majority vote"
            ),

            "ensemble_tie_break": (
                "lexicographically smallest "
                "config_id"
            ),

            "seeds": [
                101,
                202,
                303,
            ],

            "final_refit_epochs": (
                zero_epochs
            ),

            "architecture": [
                "Linear(67,256)",
                "LayerNorm(256)",
                "GELU",
                "Dropout(0.20)",

                "Linear(256,128)",
                "LayerNorm(128)",
                "GELU",
                "Dropout(0.20)",

                "Linear(128,64)",
                "LayerNorm(64)",
                "GELU",
                "Dropout(0.10)",

                "Linear(64,output_dim)",
            ],

            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
        },

        "probe_ablation": {
            "name": (
                "Probe-SML MLP ensemble"
            ),

            "descriptor_dimension": 84,
            "protocol_dimension": 3,
            "input_dimension": 87,

            "dataset_identity_used": False,

            "seeds": [
                101,
                202,
                303,
            ],

            "final_refit_epochs": (
                probe_epochs
            ),

            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
        },

        "validation_results": {
            "zero_seed_rates": (
                zero_rates
            ),

            "probe_seed_rates": (
                probe_rates
            ),

            "zero_ensemble_equivalent_rate": (
                comparison[
                    "zero_ensemble_equivalent_rate"
                ]
            ),

            "probe_ensemble_equivalent_rate": (
                comparison[
                    "probe_ensemble_equivalent_rate"
                ]
            ),

            "best_classical": {
                "feature_set": str(
                    best_classical[
                        "feature_set"
                    ]
                ),

                "model": str(
                    best_classical[
                        "model"
                    ]
                ),

                "hard_accuracy": float(
                    best_classical[
                        "hard_accuracy"
                    ]
                ),

                "equivalent_rate": float(
                    best_classical[
                        "equivalent_rate"
                    ]
                ),
            },
        },

        "static_validation_baselines": (
            baselines.to_dict(
                orient="records"
            )
        ),

        "classical_validation_models": (
            classical.to_dict(
                orient="records"
            )
        ),

        "final_test_policy": {
            "predictions_must_be_generated_before_test_oracle": True,

            "no_model_selection_after_test_oracle": True,

            "primary_metric": (
                "test regret relative "
                "to robust oracle"
            ),

            "secondary_metrics": [
                "oracle-equivalent recommendation rate",
                "hard oracle-label accuracy",
                "mean regret",
                "median regret",
                "95th-percentile regret",
                "maximum regret",
                "recommendation latency",
            ],
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Frozen protocol:",
        OUTPUT,
    )

    print(
        "Zero final epochs:",
        zero_epochs,
    )

    print(
        "Probe final epochs:",
        probe_epochs,
    )

    print(
        "Best classical:",
        best_classical[
            "feature_set"
        ],
        "/",
        best_classical[
            "model"
        ],
        f"{100*best_classical['equivalent_rate']:.2f}%",
    )

    print()
    print(
        "FINAL META PROTOCOL FREEZE: PASS"
    )


if __name__ == "__main__":
    main()
