from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sml_hpo.data.omniglot import load_omniglot_splits
from sml_hpo.episodes.sampler import EpisodeSampler
from sml_hpo.models.protonet import ProtoNet
from sml_hpo.training.engine import (
    evaluate_episode_bank,
    train_episode,
)
from sml_hpo.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/omniglot"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-episodes", type=int, default=60)
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
        default=Path("results/tables/smoke_omniglot.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/checkpoints/smoke_omniglot.pt"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This smoke test is expected to use GPU."
        )

    seed_everything(args.seed, deterministic=True)

    # CUDA_VISIBLE_DEVICES maps the selected physical GPU to cuda:0.
    device = torch.device("cuda:0")

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    free_memory, total_memory = torch.cuda.mem_get_info(0)
    print(
        "CUDA memory before run: "
        f"{free_memory / 1024**3:.2f} GB free / "
        f"{total_memory / 1024**3:.2f} GB total"
    )

    splits = load_omniglot_splits(
        root=args.data_root,
        image_size=84,
        validation_fraction=0.20,
        split_seed=args.seed,
        download=True,
    )

    print("Train classes:", len(splits.train_classes))
    print("Validation classes:", len(splits.validation_classes))
    print("Test classes:", len(splits.test_classes))

    train_class_set = set(splits.train_classes)
    validation_class_set = set(splits.validation_classes)

    assert train_class_set.isdisjoint(validation_class_set)

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

    training_losses: list[float] = []
    training_accuracies: list[float] = []

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

        training_losses.append(loss)
        training_accuracies.append(accuracy)

        if (
            episode_index == 0
            or (episode_index + 1) % 10 == 0
            or episode_index + 1 == args.train_episodes
        ):
            recent_window = min(10, len(training_losses))

            print(
                f"Episode {episode_index + 1:4d}/"
                f"{args.train_episodes} | "
                f"loss={np.mean(training_losses[-recent_window:]):.4f} | "
                f"accuracy="
                f"{100 * np.mean(training_accuracies[-recent_window:]):.2f}%"
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

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )

    result = {
        "status": "PASS",
        "dataset": "Omniglot",
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "train_classes": len(splits.train_classes),
        "validation_classes": len(splits.validation_classes),
        "test_classes": len(splits.test_classes),
        "n_way": args.n_way,
        "n_shot": args.n_shot,
        "n_query": args.n_query,
        "train_episodes": args.train_episodes,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "final_training_loss": training_losses[-1],
        "final_training_accuracy": training_accuracies[-1],
        "validation": validation_result,
        "test": test_result,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_gb": peak_memory_gb,
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

    print("\nValidation:")
    print(
        f"  mean={100 * validation_result['mean_accuracy']:.2f}%"
        f" ± {100 * validation_result['std_accuracy']:.2f}%"
    )

    print("Held-out Omniglot evaluation classes:")
    print(
        f"  mean={100 * test_result['mean_accuracy']:.2f}%"
        f" ± {100 * test_result['std_accuracy']:.2f}%"
    )

    print(f"Elapsed: {elapsed_seconds:.2f} seconds")
    print(f"Peak CUDA memory: {peak_memory_gb:.3f} GB")
    print("Saved result:", args.output)
    print("Saved checkpoint:", args.checkpoint)
    print("\nSMOKE TEST: PASS")


if __name__ == "__main__":
    main()
