from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from sml_hpo.data.registry import (
    load_dataset_splits,
)
from sml_hpo.episodes.sampler import (
    EpisodeSampler,
)
from sml_hpo.models.protonet import (
    ProtoNet,
)
from sml_hpo.oracle.evaluator_v2 import (
    build_optimizer_v2,
    build_scheduler_v2,
    evaluate_anchor_config_v2,
    evaluate_episode_bank,
    seed_everything,
    train_episode_v2,
)
from sml_hpo.oracle.search_space_v2 import (
    load_anchor_manifest_v2,
)
from sml_hpo.tasks.spec import (
    TaskSpec,
    load_task_manifest,
)


DEFAULT_ROOTS = {
    "omniglot":
        Path("data/raw/omniglot"),

    "cifar100":
        Path("data/raw/cifar100"),

    "miniimagenet":
        Path("data/raw/miniimagenet"),

    "dtd":
        Path("data/raw/dtd"),

    "flowers102":
        Path("data/raw/flowers102"),
}


DEFAULT_ANCHORS = Path(
    "configs/search_spaces/"
    "oracle_anchors_v2_64.json"
)

DEFAULT_MANIFEST_ROOT = Path(
    "data/manifests/final800"
)


# ------------------------------------------------------------
# Frozen protocol values from the final manuscript experiment.
# ------------------------------------------------------------

FULL_TRAIN_EPISODES = 200
VALIDATION_EPISODES = 100
TEST_EPISODES = 600

# Stage-1/reference screening used seed 101.
DEFAULT_SEARCH_MODEL_SEED = 101

# Final held-out evaluation ensemble.
FINAL_MODEL_SEEDS = (
    101,
    202,
    303,
)


class BaselineAdapterV2:
    """
    Project-specific adapter connecting HPO baselines to the
    EXACT ProtoNet / task / episode pipeline used by the final
    SML experiment.

    Important
    ---------
    Validation evaluations:
        - train on task.training_seeds()
        - select using task.validation_seeds()
        - NEVER access task.testing_seeds()

    Final test evaluations:
        - only after a baseline has frozen its selected anchor
        - train from scratch
        - evaluate on task.testing_seeds()

    Hyperparameter search space:
        the same frozen 64 AnchorConfigV2 configurations used by
        Zero-SML / Probe-SML and the validation reference.
    """

    def __init__(
        self,
        *,
        anchor_path: Path = DEFAULT_ANCHORS,
        manifest_root: Path = DEFAULT_MANIFEST_ROOT,
        device: str = "cuda:0",
        image_size: int = 84,
        split_seed: int = 42,
    ):
        self.anchor_path = Path(
            anchor_path
        )

        self.manifest_root = Path(
            manifest_root
        )

        self.device = torch.device(
            device
        )

        self.image_size = int(
            image_size
        )

        self.split_seed = int(
            split_seed
        )

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested, "
                    "but CUDA is unavailable."
                )

            torch.cuda.set_device(
                self.device.index or 0
            )

        # ----------------------------------------------------
        # Load EXACT frozen 64-anchor portfolio.
        # ----------------------------------------------------

        anchors = load_anchor_manifest_v2(
            self.anchor_path
        )

        self.config_map = {
            str(config.config_id):
                config
            for config in anchors
        }

        if len(self.config_map) != 64:
            raise RuntimeError(
                "Expected exactly 64 frozen "
                "V2 anchors, found "
                f"{len(self.config_map)}."
            )

        # ----------------------------------------------------
        # Load all deterministic final800 task manifests.
        # ----------------------------------------------------

        self.task_map: dict[
            str,
            TaskSpec,
        ] = {}

        for split in (
            "train",
            "validation",
            "test",
        ):
            split_root = (
                self.manifest_root
                / split
            )

            if not split_root.exists():
                raise FileNotFoundError(
                    split_root
                )

            manifest_paths = sorted(
                split_root.glob(
                    "*.json"
                )
            )

            if not manifest_paths:
                raise RuntimeError(
                    "No task manifests found "
                    f"under {split_root}"
                )

            for manifest_path in (
                manifest_paths
            ):
                tasks = load_task_manifest(
                    manifest_path
                )

                for task in tasks:
                    task_id = str(
                        task.task_id
                    )

                    if (
                        task_id
                        in self.task_map
                    ):
                        raise RuntimeError(
                            "Duplicate task ID: "
                            f"{task_id}"
                        )

                    self.task_map[
                        task_id
                    ] = task

        if len(self.task_map) != 800:
            raise RuntimeError(
                "Expected exactly 800 "
                "final tasks, found "
                f"{len(self.task_map)}."
            )

        # Dataset objects are expensive to load.
        # Load lazily and cache.
        self._dataset_split_cache = {}

    # ========================================================
    # Public inspection helpers
    # ========================================================

    def anchor_ids(
        self,
    ) -> list[str]:
        return sorted(
            self.config_map
        )

    def get_config(
        self,
        config_id: str,
    ):
        config_id = str(
            config_id
        )

        if (
            config_id
            not in self.config_map
        ):
            raise KeyError(
                f"Unknown config_id: "
                f"{config_id}"
            )

        return self.config_map[
            config_id
        ]

    def get_task(
        self,
        task_id: str,
    ) -> TaskSpec:
        task_id = str(
            task_id
        )

        if task_id not in self.task_map:
            raise KeyError(
                f"Unknown task_id: "
                f"{task_id}"
            )

        return self.task_map[
            task_id
        ]

    def task_ids(
        self,
        *,
        split: str | None = None,
        dataset: str | None = None,
    ) -> list[str]:
        result = []

        for task_id, task in (
            self.task_map.items()
        ):
            if (
                split is not None
                and task.split != split
            ):
                continue

            if (
                dataset is not None
                and task.dataset
                != dataset
            ):
                continue

            result.append(
                task_id
            )

        return sorted(
            result
        )

    def protocol_summary(
        self,
    ) -> dict[str, Any]:
        split_counts = {
            split:
                sum(
                    1
                    for task
                    in self.task_map.values()
                    if task.split
                    == split
                )
            for split in (
                "train",
                "validation",
                "test",
            )
        }

        dataset_counts = {}

        regime_counts = {}

        for task in (
            self.task_map.values()
        ):
            dataset_counts[
                task.dataset
            ] = (
                dataset_counts.get(
                    task.dataset,
                    0,
                )
                + 1
            )

            regime = (
                f"{task.n_way}w"
                f"{task.n_shot}s"
            )

            regime_counts[
                regime
            ] = (
                regime_counts.get(
                    regime,
                    0,
                )
                + 1
            )

        return {
            "task_count":
                len(
                    self.task_map
                ),

            "anchor_count":
                len(
                    self.config_map
                ),

            "split_counts":
                split_counts,

            "dataset_counts":
                dataset_counts,

            "regime_counts":
                regime_counts,

            "full_train_episodes":
                FULL_TRAIN_EPISODES,

            "validation_episodes":
                VALIDATION_EPISODES,

            "test_episodes":
                TEST_EPISODES,

            "default_search_model_seed":
                DEFAULT_SEARCH_MODEL_SEED,

            "final_model_seeds":
                list(
                    FINAL_MODEL_SEEDS
                ),
        }

    # ========================================================
    # Internal dataset / sampler helpers
    # ========================================================

    def _load_dataset_splits(
        self,
        dataset_name: str,
    ):
        dataset_name = str(
            dataset_name
        )

        if (
            dataset_name
            not in DEFAULT_ROOTS
        ):
            raise KeyError(
                "Unsupported dataset: "
                f"{dataset_name}"
            )

        if (
            dataset_name
            not in self._dataset_split_cache
        ):
            self._dataset_split_cache[
                dataset_name
            ] = (
                load_dataset_splits(
                    dataset_name=(
                        dataset_name
                    ),

                    root=(
                        DEFAULT_ROOTS[
                            dataset_name
                        ]
                    ),

                    image_size=(
                        self.image_size
                    ),

                    split_seed=(
                        self.split_seed
                    ),

                    download=False,
                )
            )

        return (
            self._dataset_split_cache[
                dataset_name
            ]
        )

    def _dataset_for_task(
        self,
        task: TaskSpec,
    ):
        splits = (
            self._load_dataset_splits(
                task.dataset
            )
        )

        attribute = {
            "train":
                "train_dataset",

            "validation":
                "validation_dataset",

            "test":
                "test_dataset",
        }[
            task.split
        ]

        if not hasattr(
            splits,
            attribute,
        ):
            raise RuntimeError(
                f"Dataset loader for "
                f"{task.dataset} does not "
                f"provide {attribute}."
            )

        return getattr(
            splits,
            attribute,
        )

    def make_sampler(
        self,
        task_id: str,
    ) -> EpisodeSampler:
        task = self.get_task(
            task_id
        )

        dataset = (
            self._dataset_for_task(
                task
            )
        )

        return EpisodeSampler(
            dataset=dataset,

            allowed_classes=(
                task.class_ids
            ),

            n_way=(
                task.n_way
            ),

            n_shot=(
                task.n_shot
            ),

            n_query=(
                task.n_query
            ),
        )

    # ========================================================
    # VALIDATION ADAPTER
    #
    # This is the key function used by Random Search, BO,
    # Hyperband and BOHB.
    #
    # It NEVER accesses the test episode bank.
    # ========================================================

    def train_validate_anchor(
        self,
        *,
        task_id: str,
        config_id: str,
        budget: int,
        model_seed: int = (
            DEFAULT_SEARCH_MODEL_SEED
        ),
        validation_episodes: int = (
            VALIDATION_EPISODES
        ),
    ) -> dict[str, Any]:

        task = self.get_task(
            task_id
        )

        config = self.get_config(
            config_id
        )

        budget = int(
            budget
        )

        validation_episodes = int(
            validation_episodes
        )

        if budget <= 0:
            raise ValueError(
                "budget must be positive."
            )

        if (
            budget
            > FULL_TRAIN_EPISODES
        ):
            raise ValueError(
                "budget exceeds frozen "
                f"maximum of "
                f"{FULL_TRAIN_EPISODES} "
                "training episodes."
            )

        if (
            budget
            > task.train_episodes
        ):
            raise ValueError(
                f"Task {task_id} contains "
                f"only {task.train_episodes} "
                "training episode seeds."
            )

        if (
            validation_episodes
            > task.validation_episodes
        ):
            raise ValueError(
                f"Task {task_id} contains "
                f"only "
                f"{task.validation_episodes} "
                "validation episode seeds."
            )

        sampler = self.make_sampler(
            task_id
        )

        # IMPORTANT:
        # evaluate_anchor_config_v2 uses ONLY:
        #     task.training_seeds()
        #     task.validation_seeds()
        #
        # It cannot touch task.testing_seeds().
        result = (
            evaluate_anchor_config_v2(
                task=task,

                sampler=sampler,

                config=config,

                device=self.device,

                model_seeds=[
                    int(
                        model_seed
                    )
                ],

                train_episodes=(
                    budget
                ),

                validation_episodes=(
                    validation_episodes
                ),
            )
        )

        output = dict(
            result
        )

        output.update(
            {
                "task_id":
                    task.task_id,

                "dataset":
                    task.dataset,

                "meta_split":
                    task.split,

                "config_id":
                    config.config_id,

                # Normalized adapter-level names.
                # evaluator_v2 itself uses
                # "train_episodes".
                "training_episodes":
                    budget,

                "train_episodes":
                    budget,

                "validation_episodes":
                    validation_episodes,

                "resource_type":
                    "training_episodes",

                "resource_budget":
                    budget,

                "model_seed":
                    int(
                        model_seed
                    ),

                "model_seeds":
                    [
                        int(
                            model_seed
                        )
                    ],

                "model_seed_count":
                    1,

                "test_bank_used":
                    False,
            }
        )

        return output

    def validation_accuracy(
        self,
        *,
        task_id: str,
        config_id: str,
        budget: int,
        model_seed: int = (
            DEFAULT_SEARCH_MODEL_SEED
        ),
    ) -> float:

        result = (
            self.train_validate_anchor(
                task_id=task_id,

                config_id=config_id,

                budget=budget,

                model_seed=model_seed,
            )
        )

        return float(
            result[
                "validation_accuracy_mean"
            ]
        )

    # ========================================================
    # FINAL TEST ADAPTER
    #
    # NEVER call this inside any HPO objective.
    #
    # It is only for the anchor selected after validation
    # search has finished and been frozen.
    # ========================================================

    def evaluate_selected_anchor_on_test(
        self,
        *,
        task_id: str,
        config_id: str,
        train_episodes: int = (
            FULL_TRAIN_EPISODES
        ),
        model_seeds: Sequence[int] = (
            FINAL_MODEL_SEEDS
        ),
    ) -> dict[str, Any]:

        task = self.get_task(
            task_id
        )

        if task.split != "test":
            raise ValueError(
                "Final held-out evaluation "
                "is only defined for "
                "meta-test tasks."
            )

        config = self.get_config(
            config_id
        )

        sampler = self.make_sampler(
            task_id
        )

        train_episodes = int(
            train_episodes
        )

        train_seed_bank = (
            task.training_seeds()[
                :train_episodes
            ]
        )

        test_seed_bank = (
            task.testing_seeds()
        )

        if (
            len(train_seed_bank)
            != train_episodes
        ):
            raise RuntimeError(
                "Insufficient training "
                "episode seeds."
            )

        if (
            len(test_seed_bank)
            != TEST_EPISODES
        ):
            raise RuntimeError(
                "Expected exactly "
                f"{TEST_EPISODES} test "
                "episodes, found "
                f"{len(test_seed_bank)}."
            )

        if not model_seeds:
            raise ValueError(
                "model_seeds must not "
                "be empty."
            )

        per_seed = []

        config_start = (
            time.perf_counter()
        )

        for model_seed in (
            model_seeds
        ):
            model_seed = int(
                model_seed
            )

            seed_everything(
                model_seed,
                deterministic=True,
            )

            model = ProtoNet(
                input_channels=3,

                hidden_channels=(
                    config.hidden_channels
                ),

                embedding_dim=(
                    config.embedding_dim
                ),

                dropout=(
                    config.dropout
                ),

                distance_metric=(
                    config.distance_metric
                ),

                temperature=(
                    config.temperature
                ),
            ).to(
                self.device
            )

            optimizer = (
                build_optimizer_v2(
                    model,
                    config,
                )
            )

            scheduler = (
                build_scheduler_v2(
                    optimizer,
                    config,
                    train_episodes,
                )
            )

            seed_start = (
                time.perf_counter()
            )

            train_losses = []
            train_accuracies = []

            for episode_seed in (
                train_seed_bank
            ):
                (
                    loss,
                    accuracy,
                ) = train_episode_v2(
                    model=model,

                    optimizer=optimizer,

                    sampler=sampler,

                    episode_seed=(
                        episode_seed
                    ),

                    task=task,

                    config=config,

                    device=self.device,
                )

                train_losses.append(
                    loss
                )

                train_accuracies.append(
                    accuracy
                )

                if (
                    scheduler
                    is not None
                ):
                    scheduler.step()

            test_result = (
                evaluate_episode_bank(
                    model=model,

                    sampler=sampler,

                    episode_seeds=(
                        test_seed_bank
                    ),

                    n_way=(
                        task.n_way
                    ),

                    device=self.device,
                )
            )

            if (
                self.device.type
                == "cuda"
            ):
                torch.cuda.synchronize(
                    self.device
                )

            per_seed.append(
                {
                    "model_seed":
                        model_seed,

                    "test_accuracy":
                        float(
                            test_result[
                                "mean_accuracy"
                            ]
                        ),

                    "test_episode_std":
                        float(
                            test_result[
                                "std_accuracy"
                            ]
                        ),

                    "final_train_loss":
                        float(
                            train_losses[-1]
                        ),

                    "mean_train_accuracy":
                        float(
                            np.mean(
                                train_accuracies
                            )
                        ),

                    "elapsed_seconds":
                        float(
                            time.perf_counter()
                            - seed_start
                        ),
                }
            )

            del model
            del optimizer
            del scheduler

            if (
                self.device.type
                == "cuda"
            ):
                torch.cuda.empty_cache()

        accuracies = np.asarray(
            [
                row[
                    "test_accuracy"
                ]
                for row
                in per_seed
            ],
            dtype=np.float64,
        )

        seed_times = np.asarray(
            [
                row[
                    "elapsed_seconds"
                ]
                for row
                in per_seed
            ],
            dtype=np.float64,
        )

        return {
            "task_id":
                task.task_id,

            "dataset":
                task.dataset,

            "meta_split":
                task.split,

            "config_id":
                config.config_id,

            "training_episodes":
                train_episodes,

            "test_episodes":
                len(
                    test_seed_bank
                ),

            "model_seeds":
                [
                    int(seed)
                    for seed
                    in model_seeds
                ],

            "test_accuracies_by_seed":
                accuracies.tolist(),

            "test_accuracy_mean":
                float(
                    accuracies.mean()
                ),

            "test_accuracy_seed_std":
                float(
                    accuracies.std(
                        ddof=1
                    )
                    if len(
                        accuracies
                    ) > 1
                    else 0.0
                ),

            "mean_seed_elapsed_seconds":
                float(
                    seed_times.mean()
                ),

            "total_elapsed_seconds":
                float(
                    time.perf_counter()
                    - config_start
                ),

            "test_bank_used":
                True,
        }


def build_default_adapter(
    *,
    device: str = "cuda:0",
) -> BaselineAdapterV2:

    return BaselineAdapterV2(
        anchor_path=(
            DEFAULT_ANCHORS
        ),

        manifest_root=(
            DEFAULT_MANIFEST_ROOT
        ),

        device=device,

        image_size=84,

        split_seed=42,
    )
