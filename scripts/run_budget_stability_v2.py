from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
        "--budgets",
        type=int,
        nargs="+",
        default=[100, 200, 300],
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
        default=[101],
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

    if len(tasks) != 10:
        raise RuntimeError(
            f"Expected 10 selected tasks, "
            f"found {len(tasks)}"
        )

    budgets = sorted(
        set(args.budgets)
    )

    if budgets != [100, 200, 300]:
        raise ValueError(
            "This protocol expects budgets "
            "100, 200 and 300"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_runs = (
        len(tasks) * len(budgets)
    )

    run_index = 0

    for task in tasks:
        task_id = str(
            task["task_id"]
        )

        manifest = Path(
            task["manifest"]
        )

        for budget in budgets:
            run_index += 1

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
                / f"budget_{budget}.csv"
            )

            best_output = (
                task_output_dir
                / f"budget_{budget}_best.json"
            )

            log_path = (
                args.log_root
                / task_id
                / f"budget_{budget}.log"
            )

            print()
            print("=" * 78)
            print(
                f"[{run_index}/{total_runs}] "
                f"{task_id} | budget={budget}"
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
                str(budget),
                "--validation-episodes",
                str(
                    args.validation_episodes
                ),
                "--model-seeds",
                *[
                    str(seed)
                    for seed
                    in args.model_seeds
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
    print("Completed runs:", total_runs)
    print("ORACLE BUDGET-STABILITY RUNS: PASS")


if __name__ == "__main__":
    main()
