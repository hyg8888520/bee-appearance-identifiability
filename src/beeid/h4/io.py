"""Small atomic CSV helpers shared by H4 stages."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Sequence

from ..utils import atomic_write_text


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    selected = tuple(fields or (tuple(rows[0]) if rows else ()))
    buffer = io.StringIO(newline="")
    if selected:
        writer = csv.DictWriter(
            buffer, fieldnames=selected, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def read_csv(path: Path, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required {label} does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
