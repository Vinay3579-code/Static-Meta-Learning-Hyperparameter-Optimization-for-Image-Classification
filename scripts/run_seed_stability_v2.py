from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_TASK_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-selection",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--anchors",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--log-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--extra-model-seeds",
        type=int,
        nargs="+",
        default=[202, 303],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def run_and_tee(
    command: list[str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("$", " ".join(command))
    print()

    with log_path.open(
        "w",
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
            print(
                line,
                end="",
                flush=True,
            )

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

    payload = json.loads(
        args.task_selection.read_text(
            encoding="utf-8"
        )
    )

    tasks = payload["tasks"]

    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks, "
            f"found {len(tasks)}"
        )

    if args.train_episodes != 200:
        raise ValueError(
            "Seed-stability protocol requires "
            "200 training episodes"
        )

    if args.extra_model_seeds != [202, 303]:
        raise ValueError(
            "Expected extra seeds 202 and 303"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for task_index, task in enumerate(
        tasks
    ):
        task_id = str(
            task["task_id"]
        )

        manifest = Path(
            task["manifest"]
        )

        task_output_dir = (
            args.output_root
            / task_id
        )

        task_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            task_output_dir
            / "seeds_202_303.csv"
        )

        best_output = (
            task_output_dir
            / "seeds_202_303_best.json"
        )

        log_path = (
            args.log_root
            / task_id
            / "seeds_202_303.log"
        )

        print()
        print("=" * 78)
        print(
            f"[{task_index + 1}/"
            f"{len(tasks)}] "
            f"{task_id}"
        )
        print("=" * 78)

        command = [
            sys.executable,
            "-u",
            "scripts/run_oracle_sweep_v2.py",
            "--task-manifest",
            str(manifest),
            "--task-id",
            task_id,
            "--anchors",
            str(args.anchors),
            "--output",
            str(output),
            "--best-output",
            str(best_output),
            "--train-episodes",
            str(args.train_episodes),
            "--validation-episodes",
            str(
                args.validation_episodes
            ),
            "--model-seeds",
            *[
                str(seed)
                for seed
                in args.extra_model_seeds
            ],
            "--device",
            args.device,
        ]

        if args.overwrite:
            command.append(
                "--overwrite"
            )

        run_and_tee(
            command,
            log_path,
        )

    print()
    print(
        "Completed seed-stability tasks:",
        len(tasks),
    )
    print("EXTRA-SEED ORACLE RUNS: PASS")


if __name__ == "__main__":
    main()
