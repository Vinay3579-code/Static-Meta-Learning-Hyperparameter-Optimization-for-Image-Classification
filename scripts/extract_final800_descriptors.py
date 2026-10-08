from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATA_ROOTS = {
    "omniglot": Path(
        "data/raw/omniglot"
    ),
    "cifar100": Path(
        "data/raw/cifar100"
    ),
    "miniimagenet": Path(
        "data/raw/miniimagenet"
    ),
    "dtd": Path(
        "data/raw/dtd"
    ),
    "flowers102": Path(
        "data/raw/flowers102"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path(
            "data/manifests/final800"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/descriptors/final800"
        ),
    )

    parser.add_argument(
        "--descriptor-episodes",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--probe-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--probe-validation-episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--skip-existing",
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


def dataset_from_manifest(
    manifest: Path,
) -> str:
    filename = manifest.stem

    for dataset_name in DATA_ROOTS:
        if filename.startswith(
            dataset_name + "_"
        ):
            return dataset_name

    raise ValueError(
        f"Cannot determine dataset from {manifest}"
    )


def main() -> None:
    args = parse_args()

    manifests = sorted(
        path
        for path in args.manifest_root.rglob(
            "*.json"
        )
        if path.name != "index.json"
    )

    if len(manifests) != 60:
        raise RuntimeError(
            f"Expected 60 manifests, found "
            f"{len(manifests)}"
        )

    for manifest_index, manifest in (
        enumerate(manifests)
    ):
        dataset_name = (
            dataset_from_manifest(
                manifest
            )
        )

        split_name = (
            manifest.parent.name
        )

        stem = manifest.stem

        zero_output = (
            args.output_root
            / "zero64"
            / split_name
            / f"{stem}.csv"
        )

        probe_output = (
            args.output_root
            / "probe20"
            / split_name
            / f"{stem}.csv"
        )

        full_output = (
            args.output_root
            / "full84"
            / split_name
            / f"{stem}.csv"
        )

        print()
        print("=" * 76)
        print(
            f"[{manifest_index + 1}/"
            f"{len(manifests)}] "
            f"{manifest}"
        )
        print("=" * 76)

        if (
            args.skip_existing
            and full_output.exists()
            and probe_output.exists()
            and zero_output.exists()
        ):
            print("Complete descriptor set exists: SKIPPED")
            continue

        zero_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        probe_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        full_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not (
            args.skip_existing
            and zero_output.exists()
        ):
            run(
                [
                    sys.executable,
                    "-u",
                    "scripts/extract_zero_descriptors.py",
                    "--manifest",
                    str(manifest),
                    "--data-root",
                    str(
                        DATA_ROOTS[
                            dataset_name
                        ]
                    ),
                    "--output",
                    str(zero_output),
                    "--descriptor-episodes",
                    str(
                        args.descriptor_episodes
                    ),
                    "--device",
                    args.device,
                ]
            )

        run(
            [
                sys.executable,
                "-u",
                "scripts/extract_probe_descriptors.py",
                "--manifest",
                str(manifest),
                "--data-root",
                str(
                    DATA_ROOTS[
                        dataset_name
                    ]
                ),
                "--zero-descriptors",
                str(zero_output),
                "--probe-output",
                str(probe_output),
                "--full-output",
                str(full_output),
                "--probe-steps",
                str(args.probe_steps),
                "--validation-episodes",
                str(
                    args.probe_validation_episodes
                ),
                "--device",
                args.device,
            ]
        )

    print()
    print("FINAL 800 DESCRIPTOR EXTRACTION: PASS")


if __name__ == "__main__":
    main()
