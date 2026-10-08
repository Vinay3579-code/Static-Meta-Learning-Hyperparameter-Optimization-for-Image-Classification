from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.neighbors import (
    NearestNeighbors,
)
from sklearn.preprocessing import (
    StandardScaler,
)

from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)
from sml_hpo.descriptors.probe import (
    PROBE_FEATURE_NAMES,
)


TRAIN_PATH = Path(
    "results/meta_learning/datasets_v2/train.csv"
)

VALIDATION_PATH = Path(
    "results/meta_learning/datasets_v2/validation.csv"
)

OUTPUT = Path(
    "results/meta_learning/classical_v2"
)


def equivalents(value):
    result = json.loads(str(value))

    if not isinstance(result, list):
        raise ValueError(
            "Expected JSON list"
        )

    return [
        str(x)
        for x in result
    ]


def make_x(
    table,
    feature_names,
):
    descriptor = table[
        feature_names
    ].to_numpy(
        dtype=np.float64
    )

    protocol = table[
        [
            "n_way",
            "n_shot",
            "n_query",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    x = np.concatenate(
        [
            descriptor,
            protocol,
        ],
        axis=1,
    )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Non-finite input"
        )

    return x


def expand_soft_labels(
    x,
    table,
):
    x_rows = []
    labels = []
    weights = []

    for index, value in enumerate(
        table[
            "oracle_equivalent_config_ids_json"
        ]
    ):
        ids = equivalents(value)

        if not ids:
            raise RuntimeError(
                "Empty equivalent set"
            )

        weight = 1.0 / len(ids)

        for config_id in ids:
            x_rows.append(
                x[index]
            )

            labels.append(
                config_id
            )

            weights.append(
                weight
            )

    return (
        np.asarray(
            x_rows,
            dtype=np.float64,
        ),
        np.asarray(
            labels,
            dtype=object,
        ),
        np.asarray(
            weights,
            dtype=np.float64,
        ),
    )


def evaluate(
    table,
    predictions,
):
    hard = []
    equiv = []

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

        hard.append(
            prediction
            == str(
                row[
                    "best_config_id"
                ]
            )
        )

        equiv.append(
            prediction
            in equivalents(
                row[
                    "oracle_equivalent_config_ids_json"
                ]
            )
        )

    return {
        "hard_accuracy":
            float(np.mean(hard)),
        "equivalent_rate":
            float(np.mean(equiv)),
    }


def knn_predict(
    x_train,
    train_table,
    x_validation,
    k=15,
):
    scaler = StandardScaler()

    train_scaled = (
        scaler.fit_transform(
            x_train
        )
    )

    validation_scaled = (
        scaler.transform(
            x_validation
        )
    )

    neighbors = NearestNeighbors(
        n_neighbors=min(
            k,
            len(train_table),
        ),
        metric="euclidean",
    )

    neighbors.fit(
        train_scaled
    )

    distances, indices = (
        neighbors.kneighbors(
            validation_scaled
        )
    )

    predictions = []

    for row_distances, row_indices in zip(
        distances,
        indices,
    ):
        scores = {}

        for distance, train_index in zip(
            row_distances,
            row_indices,
        ):
            ids = equivalents(
                train_table.iloc[
                    int(train_index)
                ][
                    "oracle_equivalent_config_ids_json"
                ]
            )

            neighbor_weight = (
                1.0
                / (float(distance) + 1e-8)
            )

            config_weight = (
                neighbor_weight
                / len(ids)
            )

            for config_id in ids:
                scores[
                    config_id
                ] = (
                    scores.get(
                        config_id,
                        0.0,
                    )
                    + config_weight
                )

        prediction = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[0][0]

        predictions.append(
            prediction
        )

    return predictions


def evaluate_feature_set(
    name,
    features,
    train,
    validation,
):
    print()
    print("=" * 72)
    print(
        f"FEATURE SET: {name}"
    )
    print("=" * 72)

    x_train = make_x(
        train,
        features,
    )

    x_validation = make_x(
        validation,
        features,
    )

    print(
        "Input dimensions:",
        x_train.shape[1],
    )

    scaler = StandardScaler()

    x_train_scaled = (
        scaler.fit_transform(
            x_train
        )
    )

    x_validation_scaled = (
        scaler.transform(
            x_validation
        )
    )

    (
        expanded_x,
        expanded_y,
        expanded_weight,
    ) = expand_soft_labels(
        x_train_scaled,
        train,
    )

    results = []

    # Equivalence-aware multinomial logistic
    # via weighted duplicated observations.
    logistic = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=5000,
        random_state=101,
    )

    logistic.fit(
        expanded_x,
        expanded_y,
        sample_weight=(
            expanded_weight
        ),
    )

    predictions = logistic.predict(
        x_validation_scaled
    )

    metrics = evaluate(
        validation,
        predictions,
    )

    results.append({
        "feature_set": name,
        "model":
            "weighted_logistic",
        **metrics,
    })

    # Random Forest.
    forest = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=101,
        n_jobs=-1,
    )

    forest.fit(
        expanded_x,
        expanded_y,
        sample_weight=(
            expanded_weight
        ),
    )

    predictions = forest.predict(
        x_validation_scaled
    )

    metrics = evaluate(
        validation,
        predictions,
    )

    results.append({
        "feature_set": name,
        "model":
            "weighted_random_forest",
        **metrics,
    })

    # Extra Trees.
    extra = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=101,
        n_jobs=-1,
    )

    extra.fit(
        expanded_x,
        expanded_y,
        sample_weight=(
            expanded_weight
        ),
    )

    predictions = extra.predict(
        x_validation_scaled
    )

    metrics = evaluate(
        validation,
        predictions,
    )

    results.append({
        "feature_set": name,
        "model":
            "weighted_extra_trees",
        **metrics,
    })

    # Equivalence-aware nearest-task
    # warm-start baseline.
    predictions = knn_predict(
        x_train,
        train,
        x_validation,
        k=15,
    )

    metrics = evaluate(
        validation,
        predictions,
    )

    results.append({
        "feature_set": name,
        "model":
            "equivalence_knn_k15",
        **metrics,
    })

    return results


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    assert len(train) == 560
    assert len(validation) == 120

    zero_features = list(
        ZERO_FEATURE_NAMES
    )

    probe_features = [
        *ZERO_FEATURE_NAMES,
        *PROBE_FEATURE_NAMES,
    ]

    assert len(
        zero_features
    ) == 64

    assert len(
        probe_features
    ) == 84

    results = []

    results.extend(
        evaluate_feature_set(
            "Zero-SML",
            zero_features,
            train,
            validation,
        )
    )

    results.extend(
        evaluate_feature_set(
            "Probe-SML",
            probe_features,
            train,
            validation,
        )
    )

    table = pd.DataFrame(
        results
    )

    table[
        "hard_accuracy_percent"
    ] = (
        100
        * table[
            "hard_accuracy"
        ]
    )

    table[
        "equivalent_rate_percent"
    ] = (
        100
        * table[
            "equivalent_rate"
        ]
    )

    table = table.sort_values(
        [
            "equivalent_rate",
            "hard_accuracy",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    table.to_csv(
        OUTPUT
        / "validation.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "CLASSICAL META-MODEL RESULTS"
    )
    print("=" * 90)

    print(
        table[
            [
                "feature_set",
                "model",
                "hard_accuracy_percent",
                "equivalent_rate_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    print()
    print(
        "Current neural reference:"
    )

    print(
        "Zero-SML 3-seed ensemble: "
        "80.00% equivalent"
    )

    print(
        "Probe-SML 3-seed ensemble: "
        "80.00% equivalent"
    )

    print()
    print(
        "CLASSICAL META MODELS: PASS"
    )


if __name__ == "__main__":
    main()
