from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sml_hpo.oracle.search_space import (
    generate_anchor_configs,
    save_anchor_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260803,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configs = generate_anchor_configs(
        count=args.count,
        seed=args.seed,
    )

    save_anchor_manifest(
        configs,
        args.output,
        generation_seed=args.seed,
    )

    print("Output:", args.output)
    print("Configuration count:", len(configs))
    print(
        "Optimizer counts:",
        dict(
            Counter(
                config.optimizer
                for config in configs
            )
        ),
    )
    print(
        "Embedding counts:",
        dict(
            Counter(
                config.embedding_dim
                for config in configs
            )
        ),
    )
    print(
        "Dropout counts:",
        dict(
            Counter(
                config.dropout
                for config in configs
            )
        ),
    )
    print(
        "Scheduler counts:",
        dict(
            Counter(
                config.scheduler
                for config in configs
            )
        ),
    )

    print()
    print("First configuration:")
    print(configs[0])
    print()
    print("ANCHOR CONFIGURATION GENERATION: PASS")


if __name__ == "__main__":
    main()
