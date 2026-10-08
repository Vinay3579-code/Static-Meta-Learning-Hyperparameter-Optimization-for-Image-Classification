from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sml_hpo.oracle.search_space_v2 import (
    generate_anchor_configs_v2,
    save_anchor_manifest_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    configs = generate_anchor_configs_v2(
        count=args.count,
        seed=args.seed,
    )

    save_anchor_manifest_v2(
        configs,
        args.output,
        generation_seed=args.seed,
    )

    print("Output:", args.output)
    print("Configurations:", len(configs))

    print(
        "Optimizers:",
        Counter(
            config.optimizer
            for config in configs
        ),
    )

    print(
        "Widths:",
        Counter(
            config.hidden_channels
            for config in configs
        ),
    )

    print(
        "Metrics:",
        Counter(
            config.distance_metric
            for config in configs
        ),
    )

    print(
        "Label smoothing:",
        Counter(
            config.label_smoothing
            for config in configs
        ),
    )

    print("SEARCH SPACE V2 GENERATION: PASS")


if __name__ == "__main__":
    main()
