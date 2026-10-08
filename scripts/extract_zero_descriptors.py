from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sml_hpo.data.registry import load_dataset_splits
from sml_hpo.descriptors.zero import (
    ZERO_FEATURE_NAMES,
    FrozenDescriptorEncoder,
    descriptor_encoder_sha256,
    extract_zero_descriptor,
)
from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.tasks.spec import load_task_manifest
from sml_hpo.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--descriptor-episodes",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--encoder-seed",
        type=int,
        default=20260801,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    return parser.parse_args()


def select_dataset(splits, split_name: str):
    if split_name == "train":
        return splits.train_dataset

    if split_name == "validation":
        return splits.validation_dataset

    if split_name == "test":
        return splits.test_dataset

    raise ValueError(split_name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    if args.descriptor_episodes <= 0:
        raise ValueError(
            "descriptor-episodes must be positive"
        )

    tasks = load_task_manifest(args.manifest)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")

        tasks = tasks[: args.limit]

    if not tasks:
        raise RuntimeError("No tasks selected")

    datasets = {task.dataset for task in tasks}
    splits_in_manifest = {task.split for task in tasks}

    if len(datasets) != 1:
        raise RuntimeError(
            "A descriptor run must contain one dataset"
        )

    if len(splits_in_manifest) != 1:
        raise RuntimeError(
            "A descriptor run must contain one split"
        )

    dataset_name = tasks[0].dataset
    split_name = tasks[0].split

    seed_everything(
        args.encoder_seed,
        deterministic=True,
    )

    device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        torch.cuda.set_device(device.index or 0)

    dataset_splits = load_dataset_splits(
        dataset_name=dataset_name,
        root=args.data_root,
        image_size=84,
        split_seed=args.split_seed,
        download=False,
    )

    dataset = select_dataset(
        dataset_splits,
        split_name,
    )

    encoder = FrozenDescriptorEncoder(
        seed=args.encoder_seed,
    ).to(device)

    encoder.eval()

    encoder_hash = descriptor_encoder_sha256(
        encoder
    )

    print("Manifest:", args.manifest)
    print("Dataset:", dataset_name)
    print("Split:", split_name)
    print("Tasks:", len(tasks))
    print(
        "Descriptor episodes per task:",
        args.descriptor_episodes,
    )
    print("Device:", device)
    print("Feature dimensions:", len(ZERO_FEATURE_NAMES))
    print("Encoder SHA-256:", encoder_hash)

    rows: list[dict[str, object]] = []

    start_time = time.perf_counter()

    for task_index, task in enumerate(tasks):
        available_seeds = task.training_seeds()

        if args.descriptor_episodes > len(available_seeds):
            raise ValueError(
                f"Task {task.task_id} has only "
                f"{len(available_seeds)} training seeds, "
                f"but {args.descriptor_episodes} were requested"
            )

        sampler = EpisodeSampler(
            dataset=dataset,
            allowed_classes=task.class_ids,
            n_way=task.n_way,
            n_shot=task.n_shot,
            n_query=task.n_query,
        )

        episode_descriptors: list[np.ndarray] = []

        descriptor_seeds = available_seeds[
            : args.descriptor_episodes
        ]

        for episode_seed in descriptor_seeds:
            episode = sampler.sample(episode_seed)

            descriptor = extract_zero_descriptor(
                support_images=episode.support_images,
                support_labels=episode.support_labels,
                dataset_name=dataset_name,
                encoder=encoder,
                device=device,
            )

            episode_descriptors.append(descriptor)

        descriptor_matrix = np.stack(
            episode_descriptors,
            axis=0,
        )

        task_descriptor = descriptor_matrix.mean(
            axis=0,
        )

        descriptor_std = descriptor_matrix.std(
            axis=0,
        )

        if task_descriptor.shape != (64,):
            raise RuntimeError(
                f"Invalid task descriptor shape: "
                f"{task_descriptor.shape}"
            )

        row: dict[str, object] = {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "split": task.split,
            "task_seed": task.task_seed,
            "class_ids": json.dumps(
                list(task.class_ids)
            ),
            "n_way": task.n_way,
            "n_shot": task.n_shot,
            "n_query": task.n_query,
            "descriptor_episodes": (
                args.descriptor_episodes
            ),
            "descriptor_mean_std": float(
                descriptor_std.mean()
            ),
        }

        for name, value in zip(
            ZERO_FEATURE_NAMES,
            task_descriptor,
            strict=True,
        ):
            row[name] = float(value)

        rows.append(row)

        print(
            f"[{task_index + 1:4d}/{len(tasks):4d}] "
            f"{task.task_id} | "
            f"mean={task_descriptor.mean():.6f} | "
            f"std={task_descriptor.std():.6f}"
        )

    result_table = pd.DataFrame(rows)

    feature_values = result_table[
        list(ZERO_FEATURE_NAMES)
    ].to_numpy(dtype=np.float64)

    if feature_values.shape != (
        len(tasks),
        64,
    ):
        raise RuntimeError(
            f"Descriptor table has invalid shape: "
            f"{feature_values.shape}"
        )

    if not np.all(np.isfinite(feature_values)):
        raise FloatingPointError(
            "Descriptor table contains non-finite values"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_table.to_csv(
        args.output,
        index=False,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    metadata = {
        "schema_version": 1,
        "descriptor_name": "zero_sml_64d",
        "operational_definition": (
            "14 pixel + 22 texture + "
            "18 frozen-encoder geometry + "
            "10 spectral features"
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(
            args.manifest
        ),
        "output": str(args.output),
        "output_sha256": file_sha256(
            args.output
        ),
        "dataset": dataset_name,
        "split": split_name,
        "task_count": len(tasks),
        "descriptor_episodes_per_task": (
            args.descriptor_episodes
        ),
        "feature_count": len(
            ZERO_FEATURE_NAMES
        ),
        "feature_names": list(
            ZERO_FEATURE_NAMES
        ),
        "encoder_seed": args.encoder_seed,
        "encoder_sha256": encoder_hash,
        "dataset_split_seed": args.split_seed,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
    }

    metadata_path = args.output.with_suffix(
        ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Descriptor table:", args.output)
    print("Metadata:", metadata_path)
    print("Rows:", len(result_table))
    print("Feature columns:", 64)
    print(f"Elapsed: {elapsed_seconds:.2f} seconds")
    print("ZERO DESCRIPTOR EXTRACTION: PASS")


if __name__ == "__main__":
    main()
