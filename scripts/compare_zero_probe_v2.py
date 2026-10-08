from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEEDS = [101, 202, 303]

DATASET = Path(
    "results/meta_learning/datasets_v2/validation.csv"
)

ZERO_ROOT = Path(
    "results/meta_learning/zero_sml_v2"
)

PROBE_ROOT = Path(
    "results/meta_learning/mlp_v2"
)

OUTPUT = Path(
    "results/meta_learning/zero_probe_comparison_v2"
)


def load_zero_module():
    path = Path(
        "scripts/train_zero_sml_v2.py"
    )

    spec = importlib.util.spec_from_file_location(
        "zero_sml_module",
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        module
    )

    return module


def parse_equiv(value):
    return set(
        str(x)
        for x in json.loads(str(value))
    )


def deterministic_mode(values):
    counts = Counter(values)

    maximum = max(
        counts.values()
    )

    winners = sorted(
        key
        for key, count in counts.items()
        if count == maximum
    )

    return winners[0]


def export_zero_predictions():
    module = load_zero_module()

    validation = pd.read_csv(
        DATASET
    )

    base_x = module.make_x(
        validation
    )

    for seed in SEEDS:
        run_dir = (
            ZERO_ROOT
            / f"seed{seed}"
        )

        checkpoint = torch.load(
            run_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )

        scaler = np.load(
            run_dir / "scaler.npz",
            allow_pickle=True,
        )

        mean = scaler["mean"]
        scale = scaler["scale"]

        x = (
            base_x - mean
        ) / scale

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32,
        )

        vocabulary = [
            str(x)
            for x in checkpoint[
                "vocabulary"
            ]
        ]

        model = module.MetaMLP(
            len(vocabulary)
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        with torch.no_grad():
            indices = (
                model(x_tensor)
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

        predictions = [
            vocabulary[int(i)]
            for i in indices
        ]

        output = validation[
            [
                "task_id",
                "dataset",
                "best_config_id",
                "oracle_equivalent_config_ids_json",
            ]
        ].copy()

        output[
            "predicted_config_id"
        ] = predictions

        output[
            "hard_correct"
        ] = (
            output[
                "predicted_config_id"
            ]
            == output[
                "best_config_id"
            ]
        )

        output[
            "oracle_equivalent"
        ] = [
            prediction
            in parse_equiv(equivalent)
            for prediction, equivalent
            in zip(
                output[
                    "predicted_config_id"
                ],
                output[
                    "oracle_equivalent_config_ids_json"
                ],
            )
        ]

        output.to_csv(
            run_dir
            / "validation_predictions.csv",
            index=False,
        )


def load_prediction_matrix(
    root: Path,
):
    frames = []

    for seed in SEEDS:
        path = (
            root
            / f"seed{seed}"
            / "validation_predictions.csv"
        )

        frame = pd.read_csv(path)

        frame = frame.sort_values(
            "task_id"
        ).reset_index(
            drop=True
        )

        frames.append(frame)

    reference_ids = (
        frames[0]["task_id"].tolist()
    )

    for frame in frames[1:]:
        if (
            frame["task_id"].tolist()
            != reference_ids
        ):
            raise RuntimeError(
                "Task ordering mismatch"
            )

    return frames


def ensemble_predictions(
    frames,
):
    result = frames[0][
        [
            "task_id",
            "dataset",
            "best_config_id",
            "oracle_equivalent_config_ids_json",
        ]
    ].copy()

    matrix = np.asarray(
        [
            frame[
                "predicted_config_id"
            ].astype(str).tolist()
            for frame in frames
        ],
        dtype=object,
    )

    predictions = []

    for column in range(
        matrix.shape[1]
    ):
        predictions.append(
            deterministic_mode(
                matrix[:, column].tolist()
            )
        )

    result[
        "predicted_config_id"
    ] = predictions

    result[
        "hard_correct"
    ] = (
        result[
            "predicted_config_id"
        ]
        == result[
            "best_config_id"
        ]
    )

    result[
        "oracle_equivalent"
    ] = [
        prediction
        in parse_equiv(equivalent)
        for prediction, equivalent
        in zip(
            result[
                "predicted_config_id"
            ],
            result[
                "oracle_equivalent_config_ids_json"
            ],
        )
    ]

    return result


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_zero_predictions()

    zero_frames = (
        load_prediction_matrix(
            ZERO_ROOT
        )
    )

    probe_frames = (
        load_prediction_matrix(
            PROBE_ROOT
        )
    )

    zero_ensemble = (
        ensemble_predictions(
            zero_frames
        )
    )

    probe_ensemble = (
        ensemble_predictions(
            probe_frames
        )
    )

    zero_ensemble.to_csv(
        OUTPUT
        / "zero_ensemble_predictions.csv",
        index=False,
    )

    probe_ensemble.to_csv(
        OUTPUT
        / "probe_ensemble_predictions.csv",
        index=False,
    )

    zero_hit = zero_ensemble[
        "oracle_equivalent"
    ].astype(int).to_numpy()

    probe_hit = probe_ensemble[
        "oracle_equivalent"
    ].astype(int).to_numpy()

    zero_hard = zero_ensemble[
        "hard_correct"
    ].mean()

    probe_hard = probe_ensemble[
        "hard_correct"
    ].mean()

    zero_equiv = zero_hit.mean()
    probe_equiv = probe_hit.mean()

    paired_difference = (
        probe_hit - zero_hit
    )

    probe_only = int(
        np.sum(
            (probe_hit == 1)
            & (zero_hit == 0)
        )
    )

    zero_only = int(
        np.sum(
            (zero_hit == 1)
            & (probe_hit == 0)
        )
    )

    both_correct = int(
        np.sum(
            (zero_hit == 1)
            & (probe_hit == 1)
        )
    )

    both_wrong = int(
        np.sum(
            (zero_hit == 0)
            & (probe_hit == 0)
        )
    )

    # Paired bootstrap.
    rng = np.random.default_rng(
        12345
    )

    n = len(
        paired_difference
    )

    boot = np.empty(
        20000,
        dtype=np.float64,
    )

    for i in range(
        len(boot)
    ):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = (
            paired_difference[
                indices
            ].mean()
        )

    lower, upper = np.percentile(
        boot,
        [2.5, 97.5],
    )

    print()
    print("=" * 72)
    print(
        "ZERO-SML VS PROBE-SML"
    )
    print("=" * 72)

    print()
    print("Three-seed majority ensemble:")

    print(
        f"Zero hard accuracy: "
        f"{100*zero_hard:.2f}%"
    )

    print(
        f"Probe hard accuracy: "
        f"{100*probe_hard:.2f}%"
    )

    print()

    print(
        f"Zero equivalent rate: "
        f"{100*zero_equiv:.2f}%"
    )

    print(
        f"Probe equivalent rate: "
        f"{100*probe_equiv:.2f}%"
    )

    print(
        f"Probe - Zero: "
        f"{100*(probe_equiv-zero_equiv):+.2f} pp"
    )

    print()
    print("Paired task outcomes:")

    print(
        "Both equivalent:",
        both_correct,
    )

    print(
        "Probe only:",
        probe_only,
    )

    print(
        "Zero only:",
        zero_only,
    )

    print(
        "Neither:",
        both_wrong,
    )

    print()
    print(
        "Paired bootstrap 95% CI "
        "for Probe - Zero:"
    )

    print(
        f"[{100*lower:+.2f}, "
        f"{100*upper:+.2f}] pp"
    )

    summary = {
        "zero_ensemble_hard_accuracy":
            float(zero_hard),
        "probe_ensemble_hard_accuracy":
            float(probe_hard),
        "zero_ensemble_equivalent_rate":
            float(zero_equiv),
        "probe_ensemble_equivalent_rate":
            float(probe_equiv),
        "equivalent_rate_difference":
            float(
                probe_equiv-zero_equiv
            ),
        "bootstrap_95_ci_lower":
            float(lower),
        "bootstrap_95_ci_upper":
            float(upper),
        "both_correct":
            both_correct,
        "probe_only":
            probe_only,
        "zero_only":
            zero_only,
        "both_wrong":
            both_wrong,
    }

    (
        OUTPUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "ZERO/PROBE PAIRED COMPARISON: PASS"
    )


if __name__ == "__main__":
    main()
