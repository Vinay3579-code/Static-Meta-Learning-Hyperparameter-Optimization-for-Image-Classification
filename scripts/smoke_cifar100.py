from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sml_hpo.data.cifar100 import load_cifar100_splits
from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.models.protonet import ProtoNet
from sml_hpo.training.engine import (
    evaluate_episode_bank,
    train_episode,
)
from sml_hpo.utils.seed import seed_everything

def safe_peak_cuda_memory_gb() -> float | None:
    try:
        return float(
            torch.cuda.max_memory_allocated() / 1024**3
        )
    except RuntimeError as exc:
        print(
            "WARNING: CUDA peak-memory query unavailable:",
            repr(exc),
        )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/cifar100"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--test-episodes", type=int, default=50)

    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=15)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/smoke_cifar100.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/checkpoints/smoke_cifar100.pt"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    seed_everything(args.seed, deterministic=True)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # Initialize the CUDA context before reading allocator statistics.
    torch.empty(1, device=device)

    try:
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError as exc:
        # Memory tracking is supplementary and must not block training.
        print(
            "WARNING: Could not reset CUDA peak-memory statistics:",
            repr(exc),
        )

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    splits = load_cifar100_splits(
        root=args.data_root,
        image_size=84,
        split_seed=args.seed,
        train_class_count=64,
        validation_class_count=16,
        download=True,
    )

    print("Total images:", len(splits.train_dataset))
    print("Train classes:", len(splits.train_classes))
    print("Validation classes:", len(splits.validation_classes))
    print("Test classes:", len(splits.test_classes))

    train_classes = set(splits.train_classes)
    validation_classes = set(splits.validation_classes)
    test_classes = set(splits.test_classes)

    assert train_classes.isdisjoint(validation_classes)
    assert train_classes.isdisjoint(test_classes)
    assert validation_classes.isdisjoint(test_classes)

    train_sampler = EpisodeSampler(
        dataset=splits.train_dataset,
        allowed_classes=splits.train_classes,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
    )

    validation_sampler = EpisodeSampler(
        dataset=splits.validation_dataset,
        allowed_classes=splits.validation_classes,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
    )

    test_sampler = EpisodeSampler(
        dataset=splits.test_dataset,
        allowed_classes=splits.test_classes,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
    )

    model = ProtoNet(
        input_channels=3,
        hidden_channels=64,
        embedding_dim=64,
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Trainable parameters:", f"{parameter_count:,}")

    losses: list[float] = []
    accuracies: list[float] = []

    start_time = time.perf_counter()

    for episode_index in range(args.train_episodes):
        episode_seed = args.seed * 100_000 + episode_index

        loss, accuracy = train_episode(
            model=model,
            optimizer=optimizer,
            episode=train_sampler.sample(episode_seed),
            n_way=args.n_way,
            device=device,
            gradient_clip_norm=5.0,
        )

        losses.append(loss)
        accuracies.append(accuracy)

        if (
            episode_index == 0
            or (episode_index + 1) % 10 == 0
            or episode_index + 1 == args.train_episodes
        ):
            window = min(10, len(losses))

            print(
                f"Episode {episode_index + 1:4d}/"
                f"{args.train_episodes} | "
                f"loss={np.mean(losses[-window:]):.4f} | "
                f"accuracy={100 * np.mean(accuracies[-window:]):.2f}%"
            )

    validation_seeds = [
        args.seed * 200_000 + index
        for index in range(args.validation_episodes)
    ]

    test_seeds = [
        args.seed * 300_000 + index
        for index in range(args.test_episodes)
    ]

    validation_result = evaluate_episode_bank(
        model=model,
        sampler=validation_sampler,
        episode_seeds=validation_seeds,
        n_way=args.n_way,
        device=device,
    )

    test_result = evaluate_episode_bank(
        model=model,
        sampler=test_sampler,
        episode_seeds=test_seeds,
        n_way=args.n_way,
        device=device,
    )

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    result = {
        "status": "PASS",
        "dataset": "CIFAR-100",
        "seed": args.seed,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_images": len(splits.train_dataset),
        "train_classes": len(splits.train_classes),
        "validation_classes": len(splits.validation_classes),
        "test_classes": len(splits.test_classes),
        "train_class_names": [
            splits.class_names[index]
            for index in splits.train_classes
        ],
        "validation_class_names": [
            splits.class_names[index]
            for index in splits.validation_classes
        ],
        "test_class_names": [
            splits.class_names[index]
            for index in splits.test_classes
        ],
        "n_way": args.n_way,
        "n_shot": args.n_shot,
        "n_query": args.n_query,
        "train_episodes": args.train_episodes,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation": validation_result,
        "test": test_result,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_gb": safe_peak_cuda_memory_gb(),
        "parameter_count": parameter_count,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "result": result,
        },
        args.checkpoint,
    )

    print("\nValidation classes:")
    print(
        f"  mean={100 * validation_result['mean_accuracy']:.2f}%"
        f" ± {100 * validation_result['std_accuracy']:.2f}%"
    )

    print("Held-out CIFAR-100 test classes:")
    print(
        f"  mean={100 * test_result['mean_accuracy']:.2f}%"
        f" ± {100 * test_result['std_accuracy']:.2f}%"
    )

    print(f"Elapsed: {elapsed_seconds:.2f} seconds")
    peak_memory = result["peak_cuda_memory_gb"]

    if peak_memory is None:
        print("Peak CUDA memory: unavailable")
    else:
        print(f"Peak CUDA memory: {peak_memory:.3f} GB")
        
    print("Saved result:", args.output)
    print("Saved checkpoint:", args.checkpoint)
    print("\nSMOKE TEST: PASS")


if __name__ == "__main__":
    main()
