from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/meta_learning/datasets_v2"
)

OUTPUT = Path(
    "results/meta_learning/baselines_v2"
)


def equivalents(
    value: str,
) -> set[str]:
    result = json.loads(str(value))

    if not isinstance(result, list):
        raise ValueError(
            "oracle-equivalent field must be a JSON list"
        )

    return {
        str(item)
        for item in result
    }


def majority(
    values: pd.Series,
) -> str:
    counts = Counter(
        values.astype(str)
    )

    # Deterministic tie-breaking.
    return sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[0][0]


def ensure_regime(
    table: pd.DataFrame,
) -> pd.DataFrame:
    table = table.copy()

    if "regime" not in table.columns:
        required = {
            "n_way",
            "n_shot",
        }

        missing = (
            required
            - set(table.columns)
        )

        if missing:
            raise RuntimeError(
                "Cannot reconstruct regime. "
                f"Missing columns: {sorted(missing)}"
            )

        table["regime"] = [
            f"{int(n_way)}w{int(n_shot)}s"
            for n_way, n_shot in zip(
                table["n_way"],
                table["n_shot"],
            )
        ]

    valid_regimes = {
        "5w1s",
        "5w5s",
        "10w1s",
        "10w5s",
    }

    observed = set(
        table["regime"].astype(str)
    )

    unexpected = (
        observed
        - valid_regimes
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected regimes: "
            f"{sorted(unexpected)}"
        )

    return table


def evaluate_predictions(
    table: pd.DataFrame,
    predictions: list[str],
) -> dict[str, float]:
    if len(predictions) != len(table):
        raise RuntimeError(
            "Prediction count mismatch"
        )

    hard_correct = []
    equivalent_correct = []

    for (
        (_, row),
        prediction,
    ) in zip(
        table.iterrows(),
        predictions,
    ):
        prediction = str(
            prediction
        )

        hard_correct.append(
            prediction
            == str(
                row["best_config_id"]
            )
        )

        equivalent_correct.append(
            prediction
            in equivalents(
                row[
                    "oracle_equivalent_config_ids_json"
                ]
            )
        )

    return {
        "hard_accuracy": float(
            sum(hard_correct)
            / len(hard_correct)
        ),
        "oracle_equivalent_rate": float(
            sum(equivalent_correct)
            / len(equivalent_correct)
        ),
    }


def append_result(
    *,
    results: list[dict],
    method: str,
    table: pd.DataFrame,
    predictions: list[str],
    fallback_count: int = 0,
    uses_dataset_identity: bool,
) -> None:
    metrics = evaluate_predictions(
        table,
        predictions,
    )

    results.append(
        {
            "method": method,
            "hard_accuracy": (
                metrics[
                    "hard_accuracy"
                ]
            ),
            "oracle_equivalent_rate": (
                metrics[
                    "oracle_equivalent_rate"
                ]
            ),
            "fallback_count": int(
                fallback_count
            ),
            "uses_dataset_identity": bool(
                uses_dataset_identity
            ),
        }
    )


def main() -> None:
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        ROOT / "train.csv"
    )

    validation = pd.read_csv(
        ROOT / "validation.csv"
    )

    train = ensure_regime(
        train
    )

    validation = ensure_regime(
        validation
    )

    required = {
        "task_id",
        "dataset",
        "regime",
        "best_config_id",
        "oracle_equivalent_config_ids_json",
    }

    for name, table in (
        ("train", train),
        ("validation", validation),
    ):
        missing = (
            required
            - set(table.columns)
        )

        if missing:
            raise RuntimeError(
                f"{name} missing columns: "
                f"{sorted(missing)}"
            )

    print(
        "Train tasks:",
        len(train),
    )

    print(
        "Validation tasks:",
        len(validation),
    )

    print(
        "Train regimes:",
        sorted(
            train[
                "regime"
            ].unique()
        ),
    )

    print(
        "Validation regimes:",
        sorted(
            validation[
                "regime"
            ].unique()
        ),
    )

    results: list[dict] = []

    # =====================================================
    # 1. Global majority
    # =====================================================

    global_anchor = majority(
        train["best_config_id"]
    )

    global_predictions = [
        global_anchor
        for _ in range(
            len(validation)
        )
    ]

    append_result(
        results=results,
        method="global_majority",
        table=validation,
        predictions=global_predictions,
        fallback_count=0,
        uses_dataset_identity=False,
    )

    print()
    print(
        "Global anchor:",
        global_anchor,
    )

    # =====================================================
    # 2. Protocol/regime-only majority
    #
    # Fairer comparator to the learned model because it
    # does NOT receive dataset identity.
    # =====================================================

    regime_map: dict[str, str] = {}

    for regime, group in (
        train.groupby(
            "regime"
        )
    ):
        regime_map[
            str(regime)
        ] = majority(
            group[
                "best_config_id"
            ]
        )

    regime_predictions = []
    regime_fallbacks = 0

    for _, row in (
        validation.iterrows()
    ):
        regime = str(
            row["regime"]
        )

        if regime in regime_map:
            regime_predictions.append(
                regime_map[
                    regime
                ]
            )
        else:
            regime_predictions.append(
                global_anchor
            )
            regime_fallbacks += 1

    append_result(
        results=results,
        method="regime_majority",
        table=validation,
        predictions=(
            regime_predictions
        ),
        fallback_count=(
            regime_fallbacks
        ),
        uses_dataset_identity=False,
    )

    # =====================================================
    # 3. Dataset majority
    #
    # Strong privileged baseline: dataset ID is supplied
    # explicitly here, while our main learned model will
    # not receive it.
    # =====================================================

    dataset_map: dict[str, str] = {}

    for dataset, group in (
        train.groupby(
            "dataset"
        )
    ):
        dataset_map[
            str(dataset)
        ] = majority(
            group[
                "best_config_id"
            ]
        )

    dataset_predictions = []
    dataset_fallbacks = 0

    for _, row in (
        validation.iterrows()
    ):
        dataset = str(
            row["dataset"]
        )

        if dataset in dataset_map:
            dataset_predictions.append(
                dataset_map[
                    dataset
                ]
            )
        else:
            dataset_predictions.append(
                global_anchor
            )
            dataset_fallbacks += 1

    append_result(
        results=results,
        method="dataset_majority",
        table=validation,
        predictions=(
            dataset_predictions
        ),
        fallback_count=(
            dataset_fallbacks
        ),
        uses_dataset_identity=True,
    )

    # =====================================================
    # 4. Dataset × regime majority
    #
    # Strongest privileged static baseline.
    # =====================================================

    dataset_regime_map: dict[
        tuple[str, str],
        str,
    ] = {}

    for (
        dataset,
        regime,
    ), group in train.groupby(
        [
            "dataset",
            "regime",
        ]
    ):
        key = (
            str(dataset),
            str(regime),
        )

        dataset_regime_map[
            key
        ] = majority(
            group[
                "best_config_id"
            ]
        )

    dataset_regime_predictions = []
    dataset_regime_fallbacks = 0

    for _, row in (
        validation.iterrows()
    ):
        key = (
            str(
                row["dataset"]
            ),
            str(
                row["regime"]
            ),
        )

        if key in (
            dataset_regime_map
        ):
            dataset_regime_predictions.append(
                dataset_regime_map[
                    key
                ]
            )

        elif str(
            row["dataset"]
        ) in dataset_map:
            dataset_regime_predictions.append(
                dataset_map[
                    str(
                        row[
                            "dataset"
                        ]
                    )
                ]
            )
            dataset_regime_fallbacks += 1

        else:
            dataset_regime_predictions.append(
                global_anchor
            )
            dataset_regime_fallbacks += 1

    append_result(
        results=results,
        method=(
            "dataset_regime_majority"
        ),
        table=validation,
        predictions=(
            dataset_regime_predictions
        ),
        fallback_count=(
            dataset_regime_fallbacks
        ),
        uses_dataset_identity=True,
    )

    # =====================================================
    # Save results
    # =====================================================

    result_table = pd.DataFrame(
        results
    )

    result_table[
        "hard_accuracy_percent"
    ] = (
        100
        * result_table[
            "hard_accuracy"
        ]
    )

    result_table[
        "oracle_equivalent_rate_percent"
    ] = (
        100
        * result_table[
            "oracle_equivalent_rate"
        ]
    )

    print()
    print("=" * 86)
    print("STATIC META-BASELINE RESULTS")
    print("=" * 86)

    print(
        result_table[
            [
                "method",
                "hard_accuracy_percent",
                "oracle_equivalent_rate_percent",
                "fallback_count",
                "uses_dataset_identity",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    result_table.to_csv(
        OUTPUT
        / "validation.csv",
        index=False,
    )

    metadata = {
        "schema_version": 2,
        "global_anchor": (
            global_anchor
        ),
        "regime_map": (
            regime_map
        ),
        "dataset_map": (
            dataset_map
        ),
        "dataset_regime_map": {
            (
                f"{dataset}|"
                f"{regime}"
            ): anchor
            for (
                dataset,
                regime,
            ), anchor in (
                dataset_regime_map.items()
            )
        },
        "notes": {
            "global_majority": (
                "No task information."
            ),
            "regime_majority": (
                "Uses episodic protocol only."
            ),
            "dataset_majority": (
                "Privileged baseline using "
                "dataset identity."
            ),
            "dataset_regime_majority": (
                "Privileged baseline using "
                "dataset identity and protocol."
            ),
        },
    }

    (
        OUTPUT
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Metadata:",
        OUTPUT
        / "metadata.json",
    )

    print(
        "Results:",
        OUTPUT
        / "validation.csv",
    )

    print()
    print(
        "STATIC META BASELINES: PASS"
    )


if __name__ == "__main__":
    main()
