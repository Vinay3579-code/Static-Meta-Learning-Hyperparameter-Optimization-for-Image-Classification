from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1-results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--full-anchors",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.002,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def atomic_write_json(
    value: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def main() -> None:
    args = parse_args()

    if args.candidate_k <= 0:
        raise ValueError(
            "candidate-k must be positive"
        )

    if args.epsilon < 0:
        raise ValueError(
            "epsilon must be non-negative"
        )

    stage1 = pd.read_csv(
        args.stage1_results
    )

    required = {
        "config_id",
        "validation_accuracy_mean",
        "model_seeds_json",
    }

    missing = required - set(
        stage1.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing stage-one columns: "
            f"{sorted(missing)}"
        )

    if len(stage1) != 64:
        raise RuntimeError(
            f"Expected 64 screening results, "
            f"found {len(stage1)}"
        )

    if stage1[
        "config_id"
    ].nunique() != 64:
        raise RuntimeError(
            "Screening table does not contain "
            "64 unique configurations"
        )

    seed_sets = {
        tuple(
            int(seed)
            for seed in json.loads(
                str(value)
            )
        )
        for value in stage1[
            "model_seeds_json"
        ]
    }

    if seed_sets != {(101,)}:
        raise RuntimeError(
            f"Stage one must use seed 101; "
            f"found {seed_sets}"
        )

    scores = stage1[
        "validation_accuracy_mean"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(scores).all():
        raise FloatingPointError(
            "Non-finite screening scores"
        )

    ranking = stage1.sort_values(
        by=[
            "validation_accuracy_mean",
            "config_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    best_score = float(
        ranking.iloc[0][
            "validation_accuracy_mean"
        ]
    )

    top_k_ids = set(
        ranking[
            "config_id"
        ].head(
            args.candidate_k
        )
    )

    near_tie_ids = set(
        ranking.loc[
            (
                best_score
                - ranking[
                    "validation_accuracy_mean"
                ]
            )
            <= args.epsilon + 1e-12,
            "config_id",
        ]
    )

    candidate_set = (
        top_k_ids
        | near_tie_ids
    )

    candidate_ids = [
        str(config_id)
        for config_id
        in ranking["config_id"]
        if config_id in candidate_set
    ]

    full_payload = json.loads(
        args.full_anchors.read_text(
            encoding="utf-8"
        )
    )

    if full_payload.get(
        "schema_version"
    ) != 2:
        raise ValueError(
            "Expected V2 anchor manifest"
        )

    configurations = full_payload[
        "configurations"
    ]

    configuration_map = {
        str(config["config_id"]): config
        for config in configurations
    }

    if len(configuration_map) != 64:
        raise RuntimeError(
            "Full anchor manifest does not "
            "contain 64 configurations"
        )

    missing_candidates = (
        set(candidate_ids)
        - set(configuration_map)
    )

    if missing_candidates:
        raise RuntimeError(
            f"Candidate configurations missing "
            f"from anchor manifest: "
            f"{sorted(missing_candidates)}"
        )

    candidate_payload = dict(
        full_payload
    )

    candidate_payload.update(
        {
            "method": (
                "seed101_topk_plus_"
                "epsilon_ties"
            ),
            "configuration_count": len(
                candidate_ids
            ),
            "configurations": [
                configuration_map[
                    config_id
                ]
                for config_id
                in candidate_ids
            ],
            "parent_anchor_manifest": str(
                args.full_anchors
            ),
            "parent_anchor_manifest_sha256": (
                sha256(
                    args.full_anchors
                )
            ),
            "selection": {
                "screening_results": str(
                    args.stage1_results
                ),
                "screening_results_sha256": (
                    sha256(
                        args.stage1_results
                    )
                ),
                "screening_seed": 101,
                "candidate_k": (
                    args.candidate_k
                ),
                "epsilon": (
                    args.epsilon
                ),
                "screening_best_score": (
                    best_score
                ),
                "top_k_ids": [
                    str(value)
                    for value
                    in ranking[
                        "config_id"
                    ].head(
                        args.candidate_k
                    )
                ],
                "epsilon_tie_ids": sorted(
                    str(value)
                    for value
                    in near_tie_ids
                ),
                "candidate_ids_in_"
                "screening_rank_order": (
                    candidate_ids
                ),
            },
        }
    )

    atomic_write_json(
        candidate_payload,
        args.output,
    )

    print(
        "Screening best:",
        ranking.iloc[0]["config_id"],
    )
    print(
        "Screening best accuracy:",
        best_score,
    )
    print(
        "Top-K count:",
        len(top_k_ids),
    )
    print(
        "Near-tie count:",
        len(near_tie_ids),
    )
    print(
        "Final candidate count:",
        len(candidate_ids),
    )
    print(
        "Candidate IDs:",
        candidate_ids,
    )
    print("Output:", args.output)
    print("CANDIDATE MANIFEST: PASS")


if __name__ == "__main__":
    main()
