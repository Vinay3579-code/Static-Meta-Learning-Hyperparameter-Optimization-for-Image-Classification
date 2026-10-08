from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sml_hpo.tasks.spec import load_task_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--equivalence-epsilon",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def run_and_tee(
    command: list[str],
    log_path: Path,
    mode: str,
) -> None:
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        mode,
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )


def main() -> None:
    args = parse_args()

    tasks = []

    for manifest in args.manifests:
        tasks.extend(
            (
                manifest,
                task,
            )
            for task in load_task_manifest(
                manifest
            )
        )

    task_ids = [
        task.task_id
        for _, task in tasks
    ]

    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError(
            "Duplicate task IDs across manifests"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Tasks:", len(tasks))
    print("Model seeds:", args.model_seeds)
    print("Device:", args.device)
    print()

    for task_index, (
        manifest,
        task,
    ) in enumerate(tasks):
        print()
        print("=" * 72)
        print(
            f"[{task_index + 1}/{len(tasks)}] "
            f"{task.task_id}"
        )
        print("=" * 72)

        output = (
            args.output_dir
            / f"{task.task_id}.csv"
        )
        best_output = (
            args.output_dir
            / f"{task.task_id}_best.json"
        )
        ranked_output = (
            args.output_dir
            / f"{task.task_id}_ranked.csv"
        )
        summary_output = (
            args.output_dir
            / f"{task.task_id}_summary.json"
        )
        log_path = (
            args.log_dir
            / f"{task.task_id}.log"
        )

        sweep_command = [
            sys.executable,
            "-u",
            "scripts/run_oracle_sweep_v2.py",
            "--task-manifest",
            str(manifest),
            "--task-id",
            task.task_id,
            "--anchors",
            str(args.anchors),
            "--output",
            str(output),
            "--best-output",
            str(best_output),
            "--train-episodes",
            str(args.train_episodes),
            "--validation-episodes",
            str(args.validation_episodes),
            "--model-seeds",
            *[
                str(seed)
                for seed in args.model_seeds
            ],
            "--device",
            args.device,
        ]

        if args.overwrite:
            sweep_command.append(
                "--overwrite"
            )

        run_and_tee(
            sweep_command,
            log_path,
            mode="w",
        )

        analysis_command = [
            sys.executable,
            "scripts/analyze_oracle_sweep.py",
            "--input",
            str(output),
            "--ranked-output",
            str(ranked_output),
            "--summary-output",
            str(summary_output),
            "--equivalence-epsilon",
            str(args.equivalence_epsilon),
            "--softmax-temperature",
            str(args.softmax_temperature),
        ]

        run_and_tee(
            analysis_command,
            log_path,
            mode="a",
        )

    print()
    print("ORACLE COLLECTION: PASS")


if __name__ == "__main__":
    main()
