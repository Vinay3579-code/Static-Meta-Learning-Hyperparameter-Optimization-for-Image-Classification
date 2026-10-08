from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


BASE_URL = (
    "https://www.robots.ox.ac.uk/"
    "~vgg/data/flowers/102"
)

OUTPUT_DIRECTORY = Path(
    "data/raw/flowers102/flowers-102"
)

FILES = {
    "imagelabels.mat": (
        "e0620be6f572b9609742df49c70aed4d"
    ),
    "setid.mat": (
        "a5357ecc9cb78c4bef273ce3793fc85c"
    ),
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def download_with_retry(
    *,
    filename: str,
    expected_md5: str,
    attempts: int = 30,
    retry_seconds: int = 20,
) -> None:
    destination = OUTPUT_DIRECTORY / filename
    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    if destination.exists():
        actual_md5 = md5sum(destination)

        if actual_md5 == expected_md5:
            print(
                f"{filename}: already present and valid"
            )
            return

        print(
            f"{filename}: existing file has incorrect MD5 "
            f"{actual_md5}; replacing it"
        )

        destination.unlink()

    url = f"{BASE_URL}/{filename}"

    for attempt in range(1, attempts + 1):
        print(
            f"{filename}: download attempt "
            f"{attempt}/{attempts}"
        )

        try:
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "Flowers102 dataset downloader"
                    )
                },
            )

            with urlopen(
                request,
                timeout=90,
            ) as response:
                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        output.write(chunk)

            actual_md5 = md5sum(temporary)

            if actual_md5 != expected_md5:
                temporary.unlink(
                    missing_ok=True
                )

                raise RuntimeError(
                    f"MD5 mismatch for {filename}: "
                    f"expected {expected_md5}, "
                    f"received {actual_md5}"
                )

            os.replace(
                temporary,
                destination,
            )

            print(
                f"{filename}: downloaded successfully"
            )
            return

        except (
            URLError,
            TimeoutError,
            OSError,
        ) as exc:
            temporary.unlink(
                missing_ok=True
            )

            print(
                f"{filename}: attempt failed: "
                f"{exc!r}"
            )

            if attempt == attempts:
                raise RuntimeError(
                    f"Could not download {filename} "
                    f"after {attempts} attempts"
                ) from exc

            print(
                f"Retrying in {retry_seconds} seconds..."
            )

            time.sleep(retry_seconds)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, expected_md5 in FILES.items():
        download_with_retry(
            filename=filename,
            expected_md5=expected_md5,
        )

    print()
    print("Flowers102 metadata:")
    
    for filename, expected_md5 in FILES.items():
        path = OUTPUT_DIRECTORY / filename
        actual_md5 = md5sum(path)

        print(
            f"{actual_md5}  {path}"
        )

        assert actual_md5 == expected_md5

    print()
    print(
        "FLOWERS102 METADATA DOWNLOAD: PASS"
    )


if __name__ == "__main__":
    main()
