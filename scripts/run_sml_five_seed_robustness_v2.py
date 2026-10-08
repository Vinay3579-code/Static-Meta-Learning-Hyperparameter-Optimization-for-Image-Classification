from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import fit_frozen_meta_ensembles_v2 as frozen
import train_equivalence_mlp_v2 as train_mod
import generate_locked_test_recommendations_v2 as rec_mod

from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
)

from sml_hpo.descriptors.probe import (
    PROBE_FEATURE_NAMES,
)


# ============================================================
# Frozen inputs
# ============================================================

DATA_PATH = Path(
    "results/meta_learning/datasets_v2/"
    "all680.csv"
)

TEST_DESCRIPTOR_PATH = Path(
    "results/descriptors/final800/merged/"
    "test_full84.csv"
)

ORIGINAL_PROTOCOL_PATH = Path(
    "configs/protocols/"
    "final_meta_learning_v2.json"
)

OUTPUT_ROOT = Path(
    "results/meta_learning/"
    "five_run_robustness_v2"
)

FROZEN_MODEL_ROOT = (
    OUTPUT_ROOT
    / "frozen"
)

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

OLD_SEEDS = [
    101,
    202,
    303,
]

MAX_EPOCHS = 400
PATIENCE = 50
GRADIENT_CLIP = 5.0


# ============================================================
# Utilities
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def get_meta_splits(
    all680: pd.DataFrame,
):

    split_column = None

    for candidate in [
        "split",
        "meta_split",
    ]:
        if candidate in all680.columns:
            split_column = candidate
            break

    if split_column is None:
        raise RuntimeError(
            "Could not find split/meta_split "
            "column in all680.csv."
        )

    values = (
        all680[
            split_column
        ]
        .astype(str)
        .str.lower()
    )

    train_mask = values.isin(
        [
            "train",
            "meta_train",
        ]
    )

    validation_mask = values.isin(
        [
            "validation",
            "val",
            "meta_validation",
        ]
    )

    train = (
        all680[
            train_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation = (
        all680[
            validation_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(train) != 560:
        raise RuntimeError(
            "Expected 560 meta-train tasks, "
            f"observed {len(train)}."
        )

    if len(validation) != 120:
        raise RuntimeError(
            "Expected 120 meta-validation tasks, "
            f"observed {len(validation)}."
        )

    if not set(
        train[
            "task_id"
        ].astype(str)
    ).isdisjoint(
        set(
            validation[
                "task_id"
            ].astype(str)
        )
    ):
        raise RuntimeError(
            "Train/validation task overlap."
        )

    return (
        train,
        validation,
        split_column,
    )


def build_vocabulary(
    train: pd.DataFrame,
):

    values = set()

    for value in train[
        "oracle_equivalent_config_ids_json"
    ]:
        values.update(
            train_mod.parse_equivalent(
                value
            )
        )

    vocabulary = sorted(
        values
    )

    if len(vocabulary) != 19:
        raise RuntimeError(
            "Expected 19 train-equivalent "
            "anchor classes, observed "
            f"{len(vocabulary)}."
        )

    return vocabulary


# ============================================================
# Validation epoch selection
# ============================================================

def select_epoch(
    *,
    train,
    validation,
    features,
    seed,
    vocabulary,
    batch_size,
    learning_rate,
    weight_decay,
    device,
):
    """
    Same architecture, scaling, soft targets,
    optimizer, DataLoader seeding, gradient
    clipping and validation metrics as the
    existing SML training implementation.

    Selection priority:
      1. oracle-equivalent recommendation rate
      2. hard oracle-label accuracy
      3. earliest epoch
    """

    set_seed(
        seed
    )

    x_train = (
        train_mod.build_input(
            train,
            list(features),
        )
    )

    x_validation = (
        train_mod.build_input(
            validation,
            list(features),
        )
    )

    expected_input_dim = (
        len(features)
        + 3
    )

    if x_train.shape != (
        560,
        expected_input_dim,
    ):
        raise RuntimeError(
            f"Unexpected train shape "
            f"{x_train.shape}"
        )

    if x_validation.shape != (
        120,
        expected_input_dim,
    ):
        raise RuntimeError(
            "Unexpected validation shape "
            f"{x_validation.shape}"
        )

    # Fit only on meta-train, exactly as
    # validation-stage model selection.
    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        x_train
    )

    x_validation = scaler.transform(
        x_validation
    )

    y_train = (
        train_mod.build_soft_targets(
            train,
            vocabulary,
        )
    )

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    x_validation_tensor = torch.tensor(
        x_validation,
        dtype=torch.float32,
        device=device,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )

    loader = DataLoader(
        TensorDataset(
            x_train_tensor,
            y_train_tensor,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    model = (
        train_mod.MetaMLP(
            input_dim=(
                expected_input_dim
            ),
            output_dim=(
                len(vocabulary)
            ),
        )
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_equivalent = -1.0
    best_hard = -1.0
    best_epoch = -1
    best_train_loss = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_examples = 0

        for (
            x_batch,
            y_batch,
        ) in loader:

            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x_batch
            )

            log_probabilities = (
                torch.log_softmax(
                    logits,
                    dim=1,
                )
            )

            loss = -(
                y_batch
                * log_probabilities
            ).sum(
                dim=1
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP,
            )

            optimizer.step()

            current_batch = (
                x_batch.shape[0]
            )

            total_loss += (
                float(
                    loss.item()
                )
                * current_batch
            )

            total_examples += (
                current_batch
            )

        train_loss = (
            total_loss
            / total_examples
        )

        metrics = train_mod.evaluate(
            model,
            x_validation_tensor,
            validation,
            vocabulary,
        )

        equivalent_rate = float(
            metrics[
                "oracle_equivalent_rate"
            ]
        )

        hard_accuracy = float(
            metrics[
                "hard_accuracy"
            ]
        )

        history.append(
            {
                "seed":
                    seed,

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "oracle_equivalent_rate":
                    equivalent_rate,

                "hard_accuracy":
                    hard_accuracy,
            }
        )

        equivalent_improved = (
            equivalent_rate
            > best_equivalent
            + 1e-12
        )

        equivalent_tied = (
            abs(
                equivalent_rate
                - best_equivalent
            )
            <= 1e-12
        )

        hard_improved_on_tie = (
            equivalent_tied
            and
            hard_accuracy
            > best_hard
            + 1e-12
        )

        improved = (
            equivalent_improved
            or
            hard_improved_on_tie
        )

        if improved:
            best_equivalent = (
                equivalent_rate
            )

            best_hard = (
                hard_accuracy
            )

            best_epoch = (
                epoch
            )

            best_train_loss = (
                train_loss
            )

            epochs_without_improvement = 0

            print(
                f"seed={seed:3d} "
                f"epoch={epoch:3d} "
                f"equiv="
                f"{100*equivalent_rate:6.2f}% "
                f"hard="
                f"{100*hard_accuracy:6.2f}%"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            break

    if best_epoch <= 0:
        raise RuntimeError(
            "No validation epoch selected."
        )

    return {
        "seed":
            int(seed),

        "selected_epoch":
            int(best_epoch),

        "validation_equivalent_rate":
            float(
                best_equivalent
            ),

        "validation_hard_accuracy":
            float(
                best_hard
            ),

        "training_loss_at_selected_epoch":
            float(
                best_train_loss
            ),

        "epochs_executed":
            int(
                history[-1][
                    "epoch"
                ]
            ),

        "history":
            pd.DataFrame(
                history
            ),
    }


# ============================================================
# Reproduce old frozen selection before new seeds
# ============================================================

def reproduce_original_selection(
    *,
    protocol,
    train,
    validation,
    vocabulary,
    zero_features,
    probe_features,
    device,
):

    expected = {
        "zero_sml": {
            int(seed):
                int(epoch)
            for seed, epoch
            in protocol[
                "primary_method"
            ][
                "final_refit_epochs"
            ].items()
        },

        "probe_sml": {
            int(seed):
                int(epoch)
            for seed, epoch
            in protocol[
                "probe_ablation"
            ][
                "final_refit_epochs"
            ].items()
        },
    }

    expected_rates = {
        "zero_sml": {
            seed: rate
            for seed, rate
            in zip(
                OLD_SEEDS,
                protocol[
                    "validation_results"
                ][
                    "zero_seed_rates"
                ],
            )
        },

        "probe_sml": {
            seed: rate
            for seed, rate
            in zip(
                OLD_SEEDS,
                protocol[
                    "validation_results"
                ][
                    "probe_seed_rates"
                ],
            )
        },
    }

    variants = [
        (
            "zero_sml",
            zero_features,
            protocol[
                "primary_method"
            ],
        ),
        (
            "probe_sml",
            probe_features,
            protocol[
                "probe_ablation"
            ],
        ),
    ]

    rows = []

    for (
        variant,
        features,
        settings,
    ) in variants:

        for seed in OLD_SEEDS:

            result = select_epoch(
                train=train,
                validation=validation,
                features=features,
                seed=seed,
                vocabulary=vocabulary,
                batch_size=int(
                    settings[
                        "batch_size"
                    ]
                ),
                learning_rate=float(
                    settings[
                        "learning_rate"
                    ]
                ),
                weight_decay=float(
                    settings[
                        "weight_decay"
                    ]
                ),
                device=device,
            )

            expected_epoch = (
                expected[
                    variant
                ][
                    seed
                ]
            )

            expected_rate = float(
                expected_rates[
                    variant
                ][
                    seed
                ]
            )

            epoch_ok = (
                result[
                    "selected_epoch"
                ]
                == expected_epoch
            )

            rate_ok = np.isclose(
                result[
                    "validation_equivalent_rate"
                ],
                expected_rate,
                atol=1e-12,
                rtol=0.0,
            )

            rows.append(
                {
                    "variant":
                        variant,

                    "seed":
                        seed,

                    "observed_epoch":
                        result[
                            "selected_epoch"
                        ],

                    "expected_epoch":
                        expected_epoch,

                    "observed_equivalent_rate":
                        result[
                            "validation_equivalent_rate"
                        ],

                    "expected_equivalent_rate":
                        expected_rate,

                    "epoch_match":
                        epoch_ok,

                    "rate_match":
                        rate_ok,
                }
            )

            if not (
                epoch_ok
                and rate_ok
            ):
                raise RuntimeError(
                    "\nOLD PROTOCOL "
                    "REPRODUCTION FAILED\n"
                    f"variant={variant}\n"
                    f"seed={seed}\n"
                    f"observed epoch="
                    f"{result['selected_epoch']}\n"
                    f"expected epoch="
                    f"{expected_epoch}\n"
                    f"observed equiv="
                    f"{result['validation_equivalent_rate']}\n"
                    f"expected equiv="
                    f"{expected_rate}\n"
                    "\nSTOP. Do not run new "
                    "five-seed experiment."
                )

    audit = pd.DataFrame(
        rows
    )

    audit.to_csv(
        OUTPUT_ROOT
        / "old_protocol_reproduction.csv",
        index=False,
    )

    print()
    print(
        "OLD 3-SEED VALIDATION "
        "SELECTION REPRODUCTION: PASS"
    )

    return audit


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FROZEN_MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    protocol = json.loads(
        ORIGINAL_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    all680 = pd.read_csv(
        DATA_PATH
    )

    if len(all680) != 680:
        raise RuntimeError(
            "Expected all680.csv to "
            "contain 680 rows."
        )

    train, validation, split_column = (
        get_meta_splits(
            all680
        )
    )

    vocabulary = (
        build_vocabulary(
            train
        )
    )

    zero_features = list(
        ZERO_FEATURE_NAMES
    )

    probe_features = [
        *ZERO_FEATURE_NAMES,
        *PROBE_FEATURE_NAMES,
    ]

    if len(
        zero_features
    ) != 64:
        raise RuntimeError(
            "Zero descriptor must "
            "contain 64 features."
        )

    if len(
        probe_features
    ) != 84:
        raise RuntimeError(
            "Probe descriptor must "
            "contain 84 features."
        )

    if set(
        vocabulary
    ) != set(
        protocol[
            "primary_method"
        ][
            "vocabulary"
        ]
        if (
            "vocabulary"
            in protocol[
                "primary_method"
            ]
        )
        else vocabulary
    ):
        raise RuntimeError(
            "Vocabulary differs from "
            "frozen protocol."
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
        "Refit tasks:",
        len(all680),
    )

    print(
        "Output classes:",
        len(vocabulary),
    )

    print(
        "Zero input:",
        len(
            zero_features
        ) + 3,
    )

    print(
        "Probe input:",
        len(
            probe_features
        ) + 3,
    )

    # ========================================================
    # A. Reproduce original 101/202/303 epoch selection.
    # ========================================================

    print()
    print("=" * 100)
    print(
        "REPRODUCING ORIGINAL "
        "VALIDATION SELECTION"
    )
    print("=" * 100)

    reproduce_original_selection(
        protocol=protocol,
        train=train,
        validation=validation,
        vocabulary=vocabulary,
        zero_features=zero_features,
        probe_features=probe_features,
        device=device,
    )

    # ========================================================
    # B. Select epochs for 0..4.
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FIVE-SEED VALIDATION "
        "EPOCH SELECTION"
    )
    print("=" * 100)

    validation_rows = []

    selected_epochs = {
        "zero_sml": {},
        "probe_sml": {},
    }

    variants = [
        (
            "zero_sml",
            zero_features,
            protocol[
                "primary_method"
            ],
        ),
        (
            "probe_sml",
            probe_features,
            protocol[
                "probe_ablation"
            ],
        ),
    ]

    history_root = (
        OUTPUT_ROOT
        / "validation_histories"
    )

    history_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        variant,
        features,
        settings,
    ) in variants:

        for seed in SEEDS:

            print()
            print(
                f"{variant} | seed={seed}"
            )

            result = select_epoch(
                train=train,
                validation=validation,
                features=features,
                seed=seed,
                vocabulary=vocabulary,
                batch_size=int(
                    settings[
                        "batch_size"
                    ]
                ),
                learning_rate=float(
                    settings[
                        "learning_rate"
                    ]
                ),
                weight_decay=float(
                    settings[
                        "weight_decay"
                    ]
                ),
                device=device,
            )

            selected_epochs[
                variant
            ][
                str(seed)
            ] = int(
                result[
                    "selected_epoch"
                ]
            )

            validation_rows.append(
                {
                    "variant":
                        variant,

                    "seed":
                        seed,

                    "selected_epoch":
                        result[
                            "selected_epoch"
                        ],

                    "validation_equivalent_rate":
                        result[
                            "validation_equivalent_rate"
                        ],

                    "validation_hard_accuracy":
                        result[
                            "validation_hard_accuracy"
                        ],

                    "training_loss_at_selected_epoch":
                        result[
                            "training_loss_at_selected_epoch"
                        ],

                    "epochs_executed":
                        result[
                            "epochs_executed"
                        ],
                }
            )

            result[
                "history"
            ].to_csv(
                history_root
                / (
                    f"{variant}_"
                    f"seed{seed}.csv"
                ),
                index=False,
            )

    validation_summary = (
        pd.DataFrame(
            validation_rows
        )
    )

    validation_summary_path = (
        OUTPUT_ROOT
        / "validation_summary.csv"
    )

    validation_summary.to_csv(
        validation_summary_path,
        index=False,
    )

    print()
    print(
        "Selected epochs:"
    )

    print(
        json.dumps(
            selected_epochs,
            indent=2,
        )
    )

    # ========================================================
    # C. Exact 680-task refit using existing fit_variant().
    # ========================================================

    print()
    print("=" * 100)
    print(
        "REFITTING FIVE-SEED "
        "SML MODELS ON ALL 680 TASKS"
    )
    print("=" * 100)

    primary = protocol[
        "primary_method"
    ]

    probe = protocol[
        "probe_ablation"
    ]

    frozen.fit_variant(
        table=all680,

        features=zero_features,

        variant_name="zero_sml",

        epochs_by_seed=(
            selected_epochs[
                "zero_sml"
            ]
        ),

        batch_size=int(
            primary[
                "batch_size"
            ]
        ),

        learning_rate=float(
            primary[
                "learning_rate"
            ]
        ),

        weight_decay=float(
            primary[
                "weight_decay"
            ]
        ),

        device=device,

        output_root=(
            FROZEN_MODEL_ROOT
        ),
    )

    frozen.fit_variant(
        table=all680,

        features=probe_features,

        variant_name="probe_sml",

        epochs_by_seed=(
            selected_epochs[
                "probe_sml"
            ]
        ),

        batch_size=int(
            probe[
                "batch_size"
            ]
        ),

        learning_rate=float(
            probe[
                "learning_rate"
            ]
        ),

        weight_decay=float(
            probe[
                "weight_decay"
            ]
        ),

        device=device,

        output_root=(
            FROZEN_MODEL_ROOT
        ),
    )

    # ========================================================
    # D. Generate test recommendations.
    #
    # IMPORTANT:
    # We deliberately DO NOT call rec_mod.main().
    # That original historical script contains an oracle-file
    # existence guard because it was designed for the original
    # pre-test lock.
    #
    # Here we invoke only the frozen prediction helper.
    # No test labels / final-test results are read.
    # ========================================================

    test = pd.read_csv(
        TEST_DESCRIPTOR_PATH
    )

    if len(test) != 120:
        raise RuntimeError(
            "Expected 120 test descriptors."
        )

    if test[
        "task_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate test task IDs."
        )

    # Point the existing exact prediction helper
    # at our new five-seed models.
    rec_mod.FROZEN_ROOT = (
        FROZEN_MODEL_ROOT
    )

    rec_mod.SEEDS = list(
        SEEDS
    )

    (
        zero_ensemble,
        zero_per_seed,
    ) = (
        rec_mod
        .predict_frozen_ensemble(
            table=test,
            variant="zero_sml",
            features=zero_features,
        )
    )

    (
        probe_ensemble,
        probe_per_seed,
    ) = (
        rec_mod
        .predict_frozen_ensemble(
            table=test,
            variant="probe_sml",
            features=probe_features,
        )
    )

    result = test[
        [
            "task_id",
            "dataset",
            "n_way",
            "n_shot",
            "n_query",
        ]
    ].copy()

    result[
        "regime"
    ] = [
        f"{int(nw)}w{int(ns)}s"
        for nw, ns
        in zip(
            result[
                "n_way"
            ],
            result[
                "n_shot"
            ],
        )
    ]

    for seed in SEEDS:

        result[
            f"zero_sml_seed{seed}"
        ] = zero_per_seed[
            seed
        ]

        result[
            f"probe_sml_seed{seed}"
        ] = probe_per_seed[
            seed
        ]

    result[
        "zero_sml_five_seed_ensemble"
    ] = zero_ensemble

    result[
        "probe_sml_five_seed_ensemble"
    ] = probe_ensemble

    recommendation_path = (
        OUTPUT_ROOT
        / "test120_recommendations.csv"
    )

    result.to_csv(
        recommendation_path,
        index=False,
    )

    # ========================================================
    # E. Freeze experiment protocol.
    # ========================================================

    robustness_protocol = {
        "schema_version":
            1,

        "experiment":
            "five_seed_sml_robustness",

        "purpose":
            (
                "Reviewer-requested five-run "
                "robustness evaluation of "
                "Zero-SML and Probe-SML."
            ),

        "seeds":
            SEEDS,

        "meta_train_tasks":
            560,

        "meta_validation_tasks":
            120,

        "final_refit_tasks":
            680,

        "test_descriptor_tasks":
            120,

        "zero_descriptor_dimension":
            64,

        "probe_descriptor_dimension":
            84,

        "protocol_dimension":
            3,

        "zero_input_dimension":
            67,

        "probe_input_dimension":
            87,

        "output_classes":
            len(
                vocabulary
            ),

        "vocabulary":
            vocabulary,

        "selected_epochs":
            selected_epochs,

        "selection_metric":
            (
                "oracle-equivalent "
                "recommendation rate"
            ),

        "selection_tiebreak":
            (
                "hard oracle-label "
                "accuracy, then earliest epoch"
            ),

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "gradient_clip_norm":
            GRADIENT_CLIP,

        "optimizer":
            "AdamW",

        "learning_rate":
            float(
                primary[
                    "learning_rate"
                ]
            ),

        "weight_decay":
            float(
                primary[
                    "weight_decay"
                ]
            ),

        "batch_size":
            int(
                primary[
                    "batch_size"
                ]
            ),

        "scaler_selection_stage":
            (
                "fit on 560 meta-train "
                "tasks only"
            ),

        "scaler_refit_stage":
            (
                "existing fit_variant "
                "implementation on all "
                "680 refit tasks"
            ),

        "test_stage1_labels_read":
            False,

        "test_oracle_labels_read":
            False,

        "test_bank_read":
            False,

        "test_evaluation_performed":
            False,

        "chronology_note":
            (
                "This is a reviewer-requested "
                "post-main-analysis robustness "
                "experiment. Model selection "
                "uses only the frozen "
                "meta-train/meta-validation "
                "data. The script does not "
                "read test oracle labels or "
                "held-out test accuracies "
                "before recommendations are "
                "locked."
            ),

        "source_protocol":
            str(
                ORIGINAL_PROTOCOL_PATH
            ),

        "source_protocol_sha256":
            sha256_file(
                ORIGINAL_PROTOCOL_PATH
            ),
    }

    protocol_path = (
        OUTPUT_ROOT
        / "protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            robustness_protocol,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # F. Recommendation lock.
    # ========================================================

    files_to_lock = [
        DATA_PATH,
        TEST_DESCRIPTOR_PATH,
        ORIGINAL_PROTOCOL_PATH,
        validation_summary_path,
        recommendation_path,
        protocol_path,
        FROZEN_MODEL_ROOT
        / "zero_sml"
        / "metadata.json",
        FROZEN_MODEL_ROOT
        / "zero_sml"
        / "scaler.npz",
        FROZEN_MODEL_ROOT
        / "probe_sml"
        / "metadata.json",
        FROZEN_MODEL_ROOT
        / "probe_sml"
        / "scaler.npz",
    ]

    for variant in [
        "zero_sml",
        "probe_sml",
    ]:
        for seed in SEEDS:
            files_to_lock.append(
                FROZEN_MODEL_ROOT
                / variant
                / f"model_seed{seed}.pt"
            )

    for path in files_to_lock:
        if not path.exists():
            raise RuntimeError(
                f"Missing lock file: "
                f"{path}"
            )

    locked_files = {
        str(path): {
            "sha256":
                sha256_file(
                    path
                ),

            "size_bytes":
                int(
                    path.stat().st_size
                ),
        }

        for path
        in files_to_lock
    }

    lock = {
        "schema_version":
            1,

        "experiment":
            "five_seed_sml_robustness",

        "selection_frozen":
            True,

        "recommendation_rows":
            120,

        "zero_seed_recommendations":
            120 * 5,

        "probe_seed_recommendations":
            120 * 5,

        "test_oracle_labels_read":
            False,

        "test_bank_used":
            False,

        "test_evaluation_performed":
            False,

        "files":
            locked_files,
    }

    lock_path = (
        OUTPUT_ROOT
        / "recommendation_lock_manifest.json"
    )

    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    checksum_paths = [
        *files_to_lock,
        lock_path,
    ]

    checksum_path = (
        OUTPUT_ROOT
        / "recommendation_files.sha256"
    )

    checksum_path.write_text(
        "".join(
            (
                f"{sha256_file(path)}  "
                f"{path}\n"
            )
            for path
            in checksum_paths
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Final audit
    # ========================================================

    expected_seed_columns = [
        f"{variant}_seed{seed}"
        for variant in [
            "zero_sml",
            "probe_sml",
        ]
        for seed in SEEDS
    ]

    for column in (
        expected_seed_columns
    ):
        if column not in result.columns:
            raise RuntimeError(
                f"Missing recommendation "
                f"column {column}"
            )

        if result[
            column
        ].isna().any():
            raise RuntimeError(
                f"NaN recommendation "
                f"in {column}"
            )

    if len(result) != 120:
        raise RuntimeError(
            "Expected 120 recommendation rows."
        )

    print()
    print("=" * 100)
    print(
        "FIVE-SEED SML ROBUSTNESS SUMMARY"
    )
    print("=" * 100)

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Zero selected epochs:",
        selected_epochs[
            "zero_sml"
        ],
    )

    print(
        "Probe selected epochs:",
        selected_epochs[
            "probe_sml"
        ],
    )

    zero_val = (
        validation_summary[
            validation_summary[
                "variant"
            ]
            == "zero_sml"
        ]
    )

    probe_val = (
        validation_summary[
            validation_summary[
                "variant"
            ]
            == "probe_sml"
        ]
    )

    print(
        "Zero mean validation "
        "equivalent rate:",
        f"{100 * zero_val['validation_equivalent_rate'].mean():.2f}%"
    )

    print(
        "Probe mean validation "
        "equivalent rate:",
        f"{100 * probe_val['validation_equivalent_rate'].mean():.2f}%"
    )

    print(
        "Zero unique recommended "
        "anchors:",
        len(
            set(
                result[
                    [
                        f"zero_sml_seed{s}"
                        for s in SEEDS
                    ]
                ]
                .to_numpy()
                .ravel()
            )
        ),
    )

    print(
        "Probe unique recommended "
        "anchors:",
        len(
            set(
                result[
                    [
                        f"probe_sml_seed{s}"
                        for s in SEEDS
                    ]
                ]
                .to_numpy()
                .ravel()
            )
        ),
    )

    print(
        "Frozen seed-specific "
        "recommendations:",
        120 * 5 * 2,
    )

    print(
        "Test oracle labels read: False"
    )

    print(
        "Test bank used: False"
    )

    print(
        "Test evaluation performed: False"
    )

    print()
    print(
        "Recommendation lock:",
        lock_path,
    )

    print()
    print(
        "FIVE-SEED SML "
        "RECOMMENDATION LOCK: PASS"
    )


if __name__ == "__main__":
    main()
