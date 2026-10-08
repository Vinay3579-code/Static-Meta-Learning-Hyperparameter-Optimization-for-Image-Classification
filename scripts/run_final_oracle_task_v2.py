from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--overwrite-stage1",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-stage2",
        action="store_true",
    )

    return parser.parse_args()


def run(command: list[str]) -> None:
    print()
    print("$", " ".join(command))
    print()

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    args = parse_args()

    protocol = json.loads(
        args.protocol.read_text(
            encoding="utf-8"
        )
    )

    assert (
        protocol[
            "training_episodes"
        ]
        == 200
    )

    assert (
        protocol[
            "validation_episodes"
        ]
        == 100
    )

    assert (
        protocol["candidate_k"]
        == 4
    )

    assert protocol[
        "screening_seed"
    ] == 101

    assert protocol[
        "confirmation_seeds"
    ] == [202, 303]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage1 = (
        args.output_dir
        / "stage1_seed101_full64.csv"
    )

    stage1_best = (
        args.output_dir
        / "stage1_seed101_best.json"
    )

    candidate_manifest = (
        args.output_dir
        / "candidate_anchors.json"
    )

    stage2 = (
        args.output_dir
        / "stage2_seeds202_303.csv"
    )

    stage2_best = (
        args.output_dir
        / "stage2_seeds202_303_best.json"
    )

    final_scores = (
        args.output_dir
        / "final_candidate_scores.csv"
    )

    final_label = (
        args.output_dir
        / "final_label.json"
    )

    stage1_command = [
        sys.executable,
        "-u",
        "scripts/run_oracle_sweep_v2.py",
        "--task-manifest",
        str(args.task_manifest),
        "--task-id",
        args.task_id,
        "--anchors",
        str(args.anchors),
        "--output",
        str(stage1),
        "--best-output",
        str(stage1_best),
        "--train-episodes",
        "200",
        "--validation-episodes",
        "100",
        "--model-seeds",
        "101",
        "--device",
        args.device,
    ]

    if args.overwrite_stage1:
        stage1_command.append(
            "--overwrite"
        )

    run(stage1_command)

    stage1_table = pd.read_csv(
        stage1
    )

    if (
        len(stage1_table) != 64
        or stage1_table[
            "config_id"
        ].nunique()
        != 64
    ):
        raise RuntimeError(
            "Stage-one screening is incomplete"
        )

    run(
        [
            sys.executable,
            "scripts/"
            "build_candidate_manifest_v2.py",
            "--stage1-results",
            str(stage1),
            "--full-anchors",
            str(args.anchors),
            "--output",
            str(candidate_manifest),
            "--candidate-k",
            "4",
            "--epsilon",
            "0.002",
        ]
    )

    candidate_payload = json.loads(
        candidate_manifest.read_text(
            encoding="utf-8"
        )
    )

    candidate_ids = {
        str(config["config_id"])
        for config
        in candidate_payload[
            "configurations"
        ]
    }

    if (
        stage2.exists()
        and not args.overwrite_stage2
    ):
        existing_stage2 = pd.read_csv(
            stage2
        )

        existing_ids = set(
            existing_stage2[
                "config_id"
            ].astype(str)
        )

        if not existing_ids.issubset(
            candidate_ids
        ):
            raise RuntimeError(
                "Existing stage-two results "
                "do not match the current "
                "candidate manifest. Rerun "
                "with --overwrite-stage2."
            )

    stage2_command = [
        sys.executable,
        "-u",
        "scripts/run_oracle_sweep_v2.py",
        "--task-manifest",
        str(args.task_manifest),
        "--task-id",
        args.task_id,
        "--anchors",
        str(candidate_manifest),
        "--output",
        str(stage2),
        "--best-output",
        str(stage2_best),
        "--train-episodes",
        "200",
        "--validation-episodes",
        "100",
        "--model-seeds",
        "202",
        "303",
        "--device",
        args.device,
    ]

    if args.overwrite_stage2:
        stage2_command.append(
            "--overwrite"
        )

    run(stage2_command)

    run(
        [
            sys.executable,
            "scripts/"
            "finalize_oracle_task_v2.py",
            "--task-manifest",
            str(args.task_manifest),
            "--task-id",
            args.task_id,
            "--stage1-results",
            str(stage1),
            "--stage2-results",
            str(stage2),
            "--candidate-manifest",
            str(candidate_manifest),
            "--scores-output",
            str(final_scores),
            "--label-output",
            str(final_label),
            "--epsilon",
            "0.002",
        ]
    )

    print()
    print("Task:", args.task_id)
    print("Final label:", final_label)
    print("FINAL ORACLE TASK: PASS")


if __name__ == "__main__":
    main()
