"""Download and verify every dataset required by the documentation gallery."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

from mne.datasets import eegbci, sample, somato


def _fetch_with_retries(
    name: str, fetch: Callable[[], object], *, attempts: int = 3
) -> object:
    """Run one MNE dataset fetch with bounded retries."""
    delays = (5, 15)
    for attempt in range(1, attempts + 1):
        try:
            result = fetch()
        except Exception as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"Could not prepare the {name} documentation dataset after "
                    f"{attempts} attempts."
                ) from error
            delay = delays[attempt - 1]
            print(f"{name} fetch attempt {attempt} failed; retrying in {delay}s")
            time.sleep(delay)
        else:
            print(f"Prepared {name}: {result}")
            return result
    raise AssertionError("unreachable")


def prefetch_docs_data(data_dir: Path) -> None:
    """Populate ``data_dir`` with all real datasets used by gallery examples."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _fetch_with_retries(
        "MNE Sample",
        lambda: sample.data_path(path=data_dir, update_path=False, verbose=True),
    )
    _fetch_with_retries(
        "MNE Somato",
        lambda: somato.data_path(path=data_dir, update_path=False, verbose=True),
    )
    _fetch_with_retries(
        "EEGBCI subject 1 run 1",
        lambda: eegbci.load_data(
            subjects=[1],
            runs=[1],
            path=data_dir,
            update_path=False,
            verbose=True,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MNE_DATA", "~/mne_data")).expanduser(),
        help="Shared MNE dataset directory (default: MNE_DATA or ~/mne_data).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prefetch_docs_data(_parse_args().data_dir.resolve())
