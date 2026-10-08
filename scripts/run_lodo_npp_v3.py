from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# Make src/ and scripts/ importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

for path in [
    str(SRC_ROOT),
    str(SCRIPTS_ROOT),
]:
    if path not in sys.path:
        sys.path.insert(0, path)


# Reuse the EXACT finalized NPP implementation.
import run_npp_baseline_v2 as npp


# ============================================================
# Frozen LODO protocol
# ============================================================

DATASETS = [
    "omniglot",
    "cifar100",
    "miniimagenet",
    "dtd",
    "flowers102",
]

SEEDS = [0, 1, 2, 3, 4]

DESCRIPTOR_ROOT = Path(
    "results/descriptors/final800/merged"
)

TRAIN_DESCRIPTOR = (
    DESCRIPTOR_ROOT
    / "train_full84.csv"
)

VALIDATION_DESCRIPTOR = (
    DESCRIPTOR_ROOT
    / "validation_full84.csv"
)

TEST_DESCRIPTOR = (
    DESCRIPTOR_ROOT
    / "test_full84.csv"
)

OUTPUT_ROOT = Path(
    "results/lodo_generalization_v3/probe_npp"
)


# ============================================================
# Utilities
# ============================================================

def normalize_dataset(value: str) -> str:
    value = str(value).strip().lower()

    aliases = {
        "cifar-100":
            "cifar100",

        "cifar_100":
            "cifar100",

        "mini-imagenet":
            "miniimagenet",

        "mini_imagenet":
            "miniimagenet",

        "flowers-102":
            "flowers102",

        "flowers_102":
            "flowers102",
    }

    return aliases.get(
        value,
        value,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def source_stage1_files(
    *,
    split: str,
    source_datasets: list[str],
) -> list[Path]:
    """
    Return ONLY historical stage-1 files belonging
    to the four source datasets.

    The held-out dataset's stage-1 file is never
    opened by this function.
    """

    if split not in {
        "train",
        "validation",
    }:
        raise RuntimeError(
            "Historical NPP labels may only "
            "come from train/validation."
        )

    expected_per_dataset = (
        112
        if split == "train"
        else 24
    )

    files = []

    for dataset in source_datasets:

        dataset_root = (
            npp.ORACLE_ROOT
            / split
            / dataset
        )

        dataset_files = sorted(
            dataset_root.glob(
                "*/*stage1_seed101_full64.csv"
            )
        )

        # The actual existing layout used by the
        # original experiment is normally:
        #
        # split/dataset/task_id/stage1_seed101_full64.csv
        #
        # So try the exact one-level task layout if
        # the broader pattern did not match.
        if not dataset_files:
            dataset_files = sorted(
                dataset_root.glob(
                    "*/stage1_seed101_full64.csv"
                )
            )

        if (
            len(dataset_files)
            != expected_per_dataset
        ):
            raise RuntimeError(
                f"{split}/{dataset}: expected "
                f"{expected_per_dataset} stage-1 "
                f"files, observed "
                f"{len(dataset_files)}.\n"
                f"Root: {dataset_root}"
            )

        files.extend(
            dataset_files
        )

    expected_total = (
        448
        if split == "train"
        else 96
    )

    if len(files) != expected_total:
        raise RuntimeError(
            f"{split}: expected "
            f"{expected_total} source files, "
            f"observed {len(files)}."
        )

    return sorted(files)


# ============================================================
# Exact NPP historical-pair construction,
# restricted to source datasets
# ============================================================

def build_source_historical_pairs(
    *,
    split: str,
    descriptor_df: pd.DataFrame,
    descriptor_columns: list[str],
    anchor_encoder,
    source_datasets: list[str],
):
    """
    This reproduces run_npp_baseline_v2.py's
    build_historical_pairs(), except that the
    stage-1 files are restricted to the four
    source datasets of the current LODO fold.
    """

    descriptor_lookup = (
        descriptor_df
        .copy()
        .assign(
            task_id=lambda x:
                x[
                    "task_id"
                ].astype(str)
        )
        .set_index(
            "task_id",
            drop=False,
        )
    )

    anchor_ids = list(
        anchor_encoder.anchor_ids
    )

    if len(anchor_ids) != 64:
        raise RuntimeError(
            "Expected exactly 64 anchors."
        )

    X_rows = []
    y_rows = []
    task_rows = []
    config_rows = []

    files = source_stage1_files(
        split=split,
        source_datasets=source_datasets,
    )

    for path in files:

        # Important:
        # this path belongs only to a source dataset.
        path_dataset = normalize_dataset(
            path.parent.parent.name
        )

        if (
            path_dataset
            not in source_datasets
        ):
            raise RuntimeError(
                "LODO leakage: stage-1 path "
                "belongs to held-out dataset:\n"
                f"{path}"
            )

        table = pd.read_csv(
            path
        )

        if len(table) != 64:
            raise RuntimeError(
                f"{path}: expected "
                "64 anchors."
            )

        table[
            "config_id"
        ] = (
            table[
                "config_id"
            ].astype(str)
        )

        if (
            table[
                "config_id"
            ].nunique()
            != 64
        ):
            raise RuntimeError(
                f"{path}: duplicate anchors."
            )

        if set(
            table[
                "config_id"
            ]
        ) != set(
            anchor_ids
        ):
            raise RuntimeError(
                f"{path}: anchor portfolio "
                "mismatch."
            )

        task_ids = (
            table[
                "task_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(task_ids) != 1:
            raise RuntimeError(
                f"{path}: ambiguous task ID."
            )

        task_id = task_ids[0]

        if (
            task_id
            not in descriptor_lookup.index
        ):
            raise RuntimeError(
                f"{task_id}: missing "
                "source descriptor row."
            )

        descriptor_row = (
            descriptor_lookup.loc[
                task_id
            ]
        )

        row_dataset = normalize_dataset(
            descriptor_row[
                "dataset"
            ]
        )

        if (
            row_dataset
            not in source_datasets
        ):
            raise RuntimeError(
                "LODO leakage detected in "
                "historical descriptor data."
            )

        base = npp.task_base_vector(
            descriptor_row,
            descriptor_columns,
        )

        performance = dict(
            zip(
                table[
                    "config_id"
                ].astype(str),

                table[
                    "validation_accuracy_mean"
                ].astype(float),
            )
        )

        for config_id in anchor_ids:

            anchor_vector = (
                anchor_encoder.encode(
                    config_id
                )
            )

            X_rows.append(
                np.concatenate(
                    [
                        base,
                        anchor_vector,
                    ]
                )
            )

            y_rows.append(
                float(
                    performance[
                        config_id
                    ]
                )
            )

            task_rows.append(
                task_id
            )

            config_rows.append(
                config_id
            )

    X = np.asarray(
        X_rows,
        dtype=np.float32,
    )

    y = np.asarray(
        y_rows,
        dtype=np.float32,
    )

    result = {
        "X":
            X,

        "y":
            y,

        "task_ids":
            np.asarray(
                task_rows,
                dtype=object,
            ),

        "config_ids":
            np.asarray(
                config_rows,
                dtype=object,
            ),
    }

    expected_tasks = (
        448
        if split == "train"
        else 96
    )

    expected_pairs = (
        expected_tasks * 64
    )

    if (
        len(
            np.unique(
                result[
                    "task_ids"
                ]
            )
        )
        != expected_tasks
    ):
        raise RuntimeError(
            f"{split}: unexpected source "
            "task count."
        )

    if len(
        result[
            "y"
        ]
    ) != expected_pairs:
        raise RuntimeError(
            f"{split}: expected "
            f"{expected_pairs} "
            "task-anchor pairs, found "
            f"{len(result['y'])}."
        )

    return result


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print(
        "STRICT FIVE-DATASET "
        "LEAVE-ONE-DATASET-OUT PROBE-NPP"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Exact finalized NPP hyperparameters
    # --------------------------------------------------------

    if tuple(
        npp.MODEL_SEEDS
    ) != tuple(SEEDS):
        raise RuntimeError(
            "Unexpected finalized NPP seeds."
        )

    if npp.EXPECTED_DESCRIPTOR_DIM != 84:
        raise RuntimeError(
            "Expected 84-D Probe-NPP descriptor."
        )

    if npp.MAX_EPOCHS != 200:
        raise RuntimeError(
            "Unexpected NPP max epochs."
        )

    if npp.PATIENCE != 30:
        raise RuntimeError(
            "Unexpected NPP patience."
        )

    if npp.BATCH_SIZE != 256:
        raise RuntimeError(
            "Unexpected NPP batch size."
        )

    if not np.isclose(
        npp.LEARNING_RATE,
        1e-3,
    ):
        raise RuntimeError(
            "Unexpected NPP learning rate."
        )

    print(
        "NPP seeds:",
        SEEDS,
    )

    print(
        "Max epochs:",
        npp.MAX_EPOCHS,
    )

    print(
        "Patience:",
        npp.PATIENCE,
    )

    print(
        "Batch size:",
        npp.BATCH_SIZE,
    )

    print(
        "Learning rate:",
        npp.LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Load descriptors
    # --------------------------------------------------------

    train_all = pd.read_csv(
        TRAIN_DESCRIPTOR
    )

    validation_all = pd.read_csv(
        VALIDATION_DESCRIPTOR
    )

    test_all = pd.read_csv(
        TEST_DESCRIPTOR
    )

    for frame in [
        train_all,
        validation_all,
        test_all,
    ]:

        if "dataset" not in frame.columns:
            raise RuntimeError(
                "Descriptor table missing "
                "dataset column."
            )

        frame[
            "dataset"
        ] = (
            frame[
                "dataset"
            ].map(
                normalize_dataset
            )
        )

        frame[
            "task_id"
        ] = (
            frame[
                "task_id"
            ].astype(str)
        )

    if len(train_all) != 560:
        raise RuntimeError(
            "Expected 560 training descriptors."
        )

    if len(
        validation_all
    ) != 120:
        raise RuntimeError(
            "Expected 120 validation descriptors."
        )

    if len(test_all) != 120:
        raise RuntimeError(
            "Expected 120 test descriptors."
        )

    # --------------------------------------------------------
    # Frozen anchor encoder: same as finalized NPP
    # --------------------------------------------------------

    adapter = npp.BaselineAdapterV2(
        anchor_path=(
            npp.DEFAULT_ANCHORS
        ),

        manifest_root=(
            npp.DEFAULT_MANIFEST_ROOT
        ),

        device="cpu",

        image_size=84,

        split_seed=42,
    )

    anchor_encoder = (
        npp.AnchorFeatureEncoder(
            adapter
        )
    )

    if len(
        anchor_encoder.anchor_ids
    ) != 64:
        raise RuntimeError(
            "Expected frozen "
            "64-anchor portfolio."
        )

    print(
        "Anchor count:",
        len(
            anchor_encoder.anchor_ids
        ),
    )

    print(
        "Anchor encoding dimension:",
        anchor_encoder.matrix.shape[1],
    )

    device = torch.device(
        "cpu"
    )

    all_test_recommendations = []
    all_candidate_scores = []
    all_validation_summaries = []
    all_validation_recommendations = []

    fold_metadata = {}

    # ========================================================
    # Five strict LODO folds
    # ========================================================

    for heldout_dataset in DATASETS:

        source_datasets = [
            dataset
            for dataset in DATASETS
            if dataset != heldout_dataset
        ]

        print()
        print("=" * 100)
        print(
            "HELD-OUT DATASET:",
            heldout_dataset,
        )
        print("=" * 100)

        source_train = (
            train_all[
                train_all[
                    "dataset"
                ].isin(
                    source_datasets
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        source_validation = (
            validation_all[
                validation_all[
                    "dataset"
                ].isin(
                    source_datasets
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        target_test = (
            test_all[
                test_all[
                    "dataset"
                ]
                == heldout_dataset
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(
            source_train
        ) != 448:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 448 source "
                "training tasks."
            )

        if len(
            source_validation
        ) != 96:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 96 source "
                "validation tasks."
            )

        if len(
            target_test
        ) != 24:
            raise RuntimeError(
                f"{heldout_dataset}: "
                "expected 24 target "
                "test tasks."
            )

        if (
            heldout_dataset
            in set(
                source_train[
                    "dataset"
                ]
            )
        ):
            raise RuntimeError(
                "Held-out dataset leaked "
                "into source train."
            )

        if (
            heldout_dataset
            in set(
                source_validation[
                    "dataset"
                ]
            )
        ):
            raise RuntimeError(
                "Held-out dataset leaked "
                "into source validation."
            )

        # ----------------------------------------------------
        # Frozen exact 84-D descriptor schema
        # ----------------------------------------------------

        descriptor_columns = (
            npp.infer_descriptor_columns(
                source_train
            )
        )

        if len(
            descriptor_columns
        ) != 84:
            raise RuntimeError(
                "Expected 84 descriptor columns."
            )

        validation_columns = (
            npp.infer_descriptor_columns(
                source_validation
            )
        )

        target_columns = (
            npp.infer_descriptor_columns(
                target_test
            )
        )

        if (
            validation_columns
            != descriptor_columns
        ):
            raise RuntimeError(
                "Source-validation descriptor "
                "schema mismatch."
            )

        if (
            target_columns
            != descriptor_columns
        ):
            raise RuntimeError(
                "Target descriptor schema mismatch."
            )

        print(
            "Source train tasks:",
            len(source_train),
        )

        print(
            "Source validation tasks:",
            len(
                source_validation
            ),
        )

        print(
            "Source refit tasks:",
            len(source_train)
            + len(source_validation),
        )

        print(
            "Target test tasks:",
            len(target_test),
        )

        print(
            "Descriptor dimension:",
            len(descriptor_columns),
        )

        print(
            "Protocol dimension:",
            3,
        )

        print(
            "Task-side dimension:",
            87,
        )

        fold_dir = (
            OUTPUT_ROOT
            / (
                "holdout_"
                + heldout_dataset
            )
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        selection_dir = (
            fold_dir
            / "model_selection"
        )

        selection_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_dir = (
            fold_dir
            / "models"
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Build historical observations from SOURCE ONLY
        # ----------------------------------------------------

        print(
            "Building source "
            "meta-train pairs..."
        )

        train_data = (
            build_source_historical_pairs(
                split="train",

                descriptor_df=(
                    source_train
                ),

                descriptor_columns=(
                    descriptor_columns
                ),

                anchor_encoder=(
                    anchor_encoder
                ),

                source_datasets=(
                    source_datasets
                ),
            )
        )

        print(
            "Building source "
            "meta-validation pairs..."
        )

        validation_data = (
            build_source_historical_pairs(
                split="validation",

                descriptor_df=(
                    source_validation
                ),

                descriptor_columns=(
                    descriptor_columns
                ),

                anchor_encoder=(
                    anchor_encoder
                ),

                source_datasets=(
                    source_datasets
                ),
            )
        )

        print(
            "Historical train pairs:",
            len(
                train_data[
                    "y"
                ]
            ),
        )

        print(
            "Historical validation pairs:",
            len(
                validation_data[
                    "y"
                ]
            ),
        )

        if (
            len(
                train_data[
                    "y"
                ]
            )
            != 448 * 64
        ):
            raise RuntimeError(
                "Expected 28,672 "
                "source-train pairs."
            )

        if (
            len(
                validation_data[
                    "y"
                ]
            )
            != 96 * 64
        ):
            raise RuntimeError(
                "Expected 6,144 "
                "source-validation pairs."
            )

        if (
            train_data[
                "X"
            ].shape[1]
            != validation_data[
                "X"
            ].shape[1]
        ):
            raise RuntimeError(
                "NPP input dimension mismatch."
            )

        # ----------------------------------------------------
        # Build target candidate features.
        #
        # IMPORTANT:
        # no target stage-1 labels or test-bank scores.
        # ----------------------------------------------------

        target_candidates = (
            npp.build_test_candidates(
                descriptor_df=(
                    target_test
                ),

                descriptor_columns=(
                    descriptor_columns
                ),

                anchor_encoder=(
                    anchor_encoder
                ),
            )
        )

        if len(
            target_candidates
        ) != 24 * 64:
            raise RuntimeError(
                "Expected 1,536 target "
                "candidate rows."
            )

        selected_epochs = {}
        seed_metadata = {}

        # ====================================================
        # Five independent predictor seeds
        # ====================================================

        for seed in SEEDS:

            print()
            print(
                f"[{heldout_dataset}] "
                f"NPP seed={seed}: "
                "source-domain model selection"
            )

            result = (
                npp.train_with_validation(
                    seed=seed,

                    train_data=(
                        train_data
                    ),

                    validation_data=(
                        validation_data
                    ),

                    device=device,

                    output_root=(
                        selection_dir
                    ),
                )
            )

            epoch = int(
                result[
                    "best_epoch"
                ]
            )

            selected_epochs[
                str(seed)
            ] = epoch

            metrics = (
                result[
                    "metrics"
                ]
            )

            print(
                f"[{heldout_dataset}] "
                f"seed={seed}: "
                f"epoch={epoch}, "
                "val_regret="
                f"{metrics['mean_regret_pp']:.4f}pp, "
                "rmse="
                f"{metrics['rmse']:.5f}, "
                "hard="
                f"{100 * metrics['hard_hit_rate']:.2f}%, "
                "value="
                f"{100 * metrics['best_value_hit_rate']:.2f}%"
            )

            summary_row = {
                "heldout_dataset":
                    heldout_dataset,

                "run_seed":
                    seed,

                "selected_epoch":
                    epoch,

                "validation_rmse":
                    metrics[
                        "rmse"
                    ],

                "validation_mean_regret_pp":
                    metrics[
                        "mean_regret_pp"
                    ],

                "validation_median_regret_pp":
                    metrics[
                        "median_regret_pp"
                    ],

                "validation_hard_hit_rate":
                    metrics[
                        "hard_hit_rate"
                    ],

                "validation_best_value_hit_rate":
                    metrics[
                        "best_value_hit_rate"
                    ],
            }

            all_validation_summaries.append(
                summary_row
            )

            val_rec = (
                result[
                    "validation_recommendations"
                ]
                .copy()
            )

            val_rec[
                "heldout_dataset"
            ] = heldout_dataset

            val_rec[
                "run_seed"
            ] = seed

            all_validation_recommendations.append(
                val_rec
            )

            # ------------------------------------------------
            # Refit on source train + source validation
            # ------------------------------------------------

            X_full = np.concatenate(
                [
                    train_data[
                        "X"
                    ],
                    validation_data[
                        "X"
                    ],
                ],
                axis=0,
            )

            y_full = np.concatenate(
                [
                    train_data[
                        "y"
                    ],
                    validation_data[
                        "y"
                    ],
                ],
                axis=0,
            )

            if len(
                X_full
            ) != 544 * 64:
                raise RuntimeError(
                    "Expected 34,816 "
                    "source-refit observations."
                )

            (
                model,
                scaler_mean,
                scaler_std,
            ) = npp.refit_model(
                seed=seed,

                epochs=epoch,

                X=X_full,

                y=y_full,

                device=device,
            )

            checkpoint_path = (
                model_dir
                / f"npp_seed{seed}.pt"
            )

            torch.save(
                {
                    "heldout_dataset":
                        heldout_dataset,

                    "source_datasets":
                        source_datasets,

                    "seed":
                        seed,

                    "selected_epoch":
                        epoch,

                    "descriptor_dimension":
                        84,

                    "protocol_dimension":
                        3,

                    "input_dim":
                        int(
                            X_full.shape[1]
                        ),

                    "state_dict":
                        {
                            key:
                                value
                                .detach()
                                .cpu()

                            for key, value
                            in model
                            .state_dict()
                            .items()
                        },

                    "scaler_mean":
                        scaler_mean,

                    "scaler_std":
                        scaler_std,

                    "descriptor_columns":
                        descriptor_columns,

                    "heldout_dataset_used_for_training":
                        False,

                    "heldout_dataset_used_for_validation":
                        False,

                    "target_stage1_labels_used":
                        False,

                    "test_bank_used":
                        False,
                },
                checkpoint_path,
            )

            # ------------------------------------------------
            # Generate target recommendations
            # ------------------------------------------------

            (
                recommendations,
                candidate_scores,
            ) = (
                npp.recommend_test_tasks(
                    model=model,

                    scaler_mean=(
                        scaler_mean
                    ),

                    scaler_std=(
                        scaler_std
                    ),

                    seed=seed,

                    test_candidates=(
                        target_candidates
                    ),

                    device=device,
                )
            )

            recommendations = (
                recommendations.copy()
            )

            candidate_scores = (
                candidate_scores.copy()
            )

            recommendations[
                "heldout_dataset"
            ] = heldout_dataset

            recommendations[
                "run_seed"
            ] = seed

            recommendations[
                "method"
            ] = "probe_npp_lodo"

            candidate_scores[
                "heldout_dataset"
            ] = heldout_dataset

            candidate_scores[
                "run_seed"
            ] = seed

            candidate_scores[
                "method"
            ] = "probe_npp_lodo"

            if (
                "selected_config_id"
                not in recommendations.columns
            ):
                raise RuntimeError(
                    "NPP recommendation output "
                    "does not contain "
                    "selected_config_id."
                )

            if len(
                recommendations
            ) != 24:
                raise RuntimeError(
                    "Expected 24 NPP "
                    "recommendations per seed."
                )

            if len(
                candidate_scores
            ) != 24 * 64:
                raise RuntimeError(
                    "Expected 1,536 candidate "
                    "scores per seed."
                )

            all_test_recommendations.append(
                recommendations
            )

            all_candidate_scores.append(
                candidate_scores
            )

            seed_metadata[
                str(seed)
            ] = {
                "selected_epoch":
                    epoch,

                "validation_mean_regret_pp":
                    float(
                        metrics[
                            "mean_regret_pp"
                        ]
                    ),

                "validation_rmse":
                    float(
                        metrics[
                            "rmse"
                        ]
                    ),

                "validation_hard_hit_rate":
                    float(
                        metrics[
                            "hard_hit_rate"
                        ]
                    ),

                "validation_best_value_hit_rate":
                    float(
                        metrics[
                            "best_value_hit_rate"
                        ]
                    ),
            }

        fold_metadata[
            heldout_dataset
        ] = {
            "source_datasets":
                source_datasets,

            "source_train_tasks":
                448,

            "source_validation_tasks":
                96,

            "source_refit_tasks":
                544,

            "target_test_tasks":
                24,

            "historical_train_pairs":
                448 * 64,

            "historical_validation_pairs":
                96 * 64,

            "refit_pairs":
                544 * 64,

            "target_candidate_rows_per_seed":
                24 * 64,

            "selected_epochs":
                selected_epochs,

            "runs":
                seed_metadata,
        }

    # ========================================================
    # Combine + audit
    # ========================================================

    recommendations = (
        pd.concat(
            all_test_recommendations,
            ignore_index=True,
        )
    )

    candidate_scores = (
        pd.concat(
            all_candidate_scores,
            ignore_index=True,
        )
    )

    validation_summary = (
        pd.DataFrame(
            all_validation_summaries
        )
    )

    validation_recommendations = (
        pd.concat(
            all_validation_recommendations,
            ignore_index=True,
        )
    )

    # --------------------------------------------------------
    # Strict expected counts
    # --------------------------------------------------------

    if len(
        recommendations
    ) != (
        5 * 5 * 24
    ):
        raise RuntimeError(
            "Expected exactly 600 "
            "LODO-NPP recommendations, "
            f"found {len(recommendations)}."
        )

    if len(
        candidate_scores
    ) != (
        5 * 5 * 24 * 64
    ):
        raise RuntimeError(
            "Expected exactly 38,400 "
            "candidate-score rows, "
            f"found {len(candidate_scores)}."
        )

    if len(
        validation_summary
    ) != 25:
        raise RuntimeError(
            "Expected 25 validation "
            "summary rows."
        )

    if len(
        validation_recommendations
    ) != (
        5 * 5 * 96
    ):
        raise RuntimeError(
            "Expected 2,400 source-validation "
            "recommendation rows."
        )

    # --------------------------------------------------------
    # Check one target recommendation per fold/seed/task
    # --------------------------------------------------------

    duplicate_key = [
        "heldout_dataset",
        "run_seed",
        "task_id",
    ]

    if recommendations[
        duplicate_key
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate LODO-NPP "
            "recommendation."
        )

    tasks_per_fold_run = (
        recommendations.groupby(
            [
                "heldout_dataset",
                "run_seed",
            ]
        )[
            "task_id"
        ]
        .nunique()
    )

    if not (
        tasks_per_fold_run
        == 24
    ).all():
        raise RuntimeError(
            "Every fold/seed must contain "
            "24 target recommendations."
        )

    # --------------------------------------------------------
    # Explicit leakage flags from finalized function
    # --------------------------------------------------------

    for column in [
        "test_stage1_labels_used",
        "test_bank_used",
    ]:
        if column in (
            recommendations.columns
        ):
            if (
                recommendations[
                    column
                ]
                .astype(bool)
                .any()
            ):
                raise RuntimeError(
                    f"Leakage flag set: "
                    f"{column}"
                )

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------

    recommendations = (
        recommendations
        .sort_values(
            [
                "heldout_dataset",
                "run_seed",
                "task_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    candidate_sort = [
        column
        for column in [
            "heldout_dataset",
            "run_seed",
            "task_id",
            "config_id",
        ]
        if column
        in candidate_scores.columns
    ]

    candidate_scores = (
        candidate_scores
        .sort_values(
            candidate_sort,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    validation_summary = (
        validation_summary
        .sort_values(
            [
                "heldout_dataset",
                "run_seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Freeze recommendation files BEFORE evaluation
    # ========================================================

    recommendations_path = (
        OUTPUT_ROOT
        / "locked_recommendations.csv"
    )

    scores_path = (
        OUTPUT_ROOT
        / "locked_candidate_predictions.csv"
    )

    validation_path = (
        OUTPUT_ROOT
        / "validation_run_summary.csv"
    )

    validation_rec_path = (
        OUTPUT_ROOT
        / "source_validation_recommendations.csv"
    )

    recommendations.to_csv(
        recommendations_path,
        index=False,
    )

    candidate_scores.to_csv(
        scores_path,
        index=False,
    )

    validation_summary.to_csv(
        validation_path,
        index=False,
    )

    validation_recommendations.to_csv(
        validation_rec_path,
        index=False,
    )

    # --------------------------------------------------------
    # Protocol metadata
    # --------------------------------------------------------

    protocol = {
        "schema_version":
            3,

        "experiment":
            "strict_leave_one_dataset_out",

        "method":
            "probe_npp",

        "description":
            (
                "Neural Performance Predictor "
                "trained on four source datasets "
                "and deployed to a completely "
                "held-out target dataset."
            ),

        "datasets":
            DATASETS,

        "model_seeds":
            SEEDS,

        "descriptor_dimension":
            84,

        "protocol_dimension":
            3,

        "task_side_dimension":
            87,

        "anchor_count":
            64,

        "anchor_encoding_dimension":
            int(
                anchor_encoder
                .matrix
                .shape[1]
            ),

        "model":
            (
                "MLP: input -> 128 ReLU "
                "Dropout(0.1) -> "
                "64 ReLU -> scalar"
            ),

        "loss":
            "MSE",

        "optimizer":
            "Adam",

        "learning_rate":
            float(
                npp.LEARNING_RATE
            ),

        "batch_size":
            int(
                npp.BATCH_SIZE
            ),

        "maximum_epochs":
            int(
                npp.MAX_EPOCHS
            ),

        "early_stopping_patience":
            int(
                npp.PATIENCE
            ),

        "model_selection":
            (
                "minimum mean source-validation "
                "recommendation regret; RMSE "
                "then earlier epoch as tie-breaks"
            ),

        "search_space":
            (
                "same frozen V2 "
                "64-anchor portfolio"
            ),

        "target_task_validation_evaluations":
            0,

        "heldout_dataset_used_for_training":
            False,

        "heldout_dataset_used_for_validation":
            False,

        "heldout_dataset_used_for_scaler_fit":
            False,

        "target_stage1_labels_used":
            False,

        "test_bank_used":
            False,

        "test_evaluation_performed":
            False,

        "recommendation_rows":
            int(
                len(
                    recommendations
                )
            ),

        "candidate_prediction_rows":
            int(
                len(
                    candidate_scores
                )
            ),

        "folds":
            fold_metadata,
    }

    protocol_path = (
        OUTPUT_ROOT
        / "protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Cryptographic lock
    # --------------------------------------------------------

    hashes = {
        "locked_recommendations.csv":
            sha256_file(
                recommendations_path
            ),

        "locked_candidate_predictions.csv":
            sha256_file(
                scores_path
            ),

        "validation_run_summary.csv":
            sha256_file(
                validation_path
            ),

        "source_validation_recommendations.csv":
            sha256_file(
                validation_rec_path
            ),

        "protocol.json":
            sha256_file(
                protocol_path
            ),
    }

    lock_path = (
        OUTPUT_ROOT
        / "RECOMMENDATION_LOCK.sha256"
    )

    with lock_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for filename, digest in (
            hashes.items()
        ):
            handle.write(
                f"{digest}  "
                f"{OUTPUT_ROOT / filename}\n"
            )

    # ========================================================
    # Final audit
    # ========================================================

    print()
    print("=" * 100)
    print(
        "LODO PROBE-NPP "
        "RECOMMENDATION LOCK"
    )
    print("=" * 100)

    print(
        "Recommendation rows:",
        len(
            recommendations
        ),
    )

    print(
        "Candidate prediction rows:",
        len(
            candidate_scores
        ),
    )

    print(
        "Validation summary rows:",
        len(
            validation_summary
        ),
    )

    print(
        "Held-out folds:",
        recommendations[
            "heldout_dataset"
        ].nunique(),
    )

    print(
        "Runs/fold:",
        recommendations.groupby(
            "heldout_dataset"
        )[
            "run_seed"
        ]
        .nunique()
        .to_dict(),
    )

    print(
        "Tasks/fold/run:",
        sorted(
            tasks_per_fold_run
            .unique()
            .tolist()
        ),
    )

    print()

    print(
        "Recommendation SHA256:",
        hashes[
            "locked_recommendations.csv"
        ],
    )

    print()

    print(
        "Held-out dataset used "
        "for training: False"
    )

    print(
        "Held-out dataset used "
        "for validation: False"
    )

    print(
        "Target stage-1 labels used: False"
    )

    print(
        "Test bank used: False"
    )

    print(
        "Test evaluation performed: False"
    )

    print()

    print(
        "STRICT LODO PROBE-NPP "
        "RECOMMENDATION LOCK: PASS"
    )


if __name__ == "__main__":
    main()
