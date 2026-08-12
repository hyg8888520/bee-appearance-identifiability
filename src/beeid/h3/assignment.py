"""Dependency-free exact linear assignment used by H3 and its metric audit."""

from __future__ import annotations

import numpy as np


def _hungarian_min(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return a minimum-cost assignment for a finite square matrix."""
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Hungarian input must be a square matrix")
    if not np.isfinite(values).all():
        raise ValueError("Hungarian input must contain only finite costs")
    size = values.shape[0]
    if size == 0:
        return []
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        min_value = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = int(p[column])
            # This is the same Hungarian update as the former scalar inner
            # loops.  Vectorising candidate scans removes the Python hot path
            # while ``argmin`` preserves the prior first-column tie rule.
            available = ~used[1:]
            current = values[active_row - 1] - u[active_row] - v[1:]
            better = available & (current < min_value[1:])
            min_value[1:][better] = current[better]
            way[1:][better] = column
            candidates = np.where(available, min_value[1:], np.inf)
            next_offset = int(np.argmin(candidates))
            delta = float(candidates[next_offset])
            next_column = next_offset + 1
            used_rows = p[used]
            u[used_rows] += delta
            v[used] -= delta
            min_value[1:][available] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = int(way[column])
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    return [(int(p[column]) - 1, column - 1) for column in range(1, size + 1)]


def maximum_weight_matching(
    scores: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    minimum_score: float | None = None,
) -> list[tuple[int, int]]:
    """Maximize pair scores while permitting every row and column to remain unmatched."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Assignment scores must be two-dimensional")
    rows, columns = values.shape
    if rows == 0 or columns == 0:
        return []
    if not np.isfinite(values).all():
        raise ValueError("Assignment scores must be finite")
    allowed = np.ones_like(values, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if allowed.shape != values.shape:
        raise ValueError("Assignment validity mask must match scores")
    if minimum_score is not None:
        allowed &= values >= float(minimum_score)

    # Extra dummy rows and columns make unmatched choices explicit even when the
    # real matrix is square. Valid scores are expected in [0, 1], while invalid
    # real-real edges are dominated by zero-valued dummy assignments.
    size = rows + columns
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:rows, :columns] = np.where(allowed, values, -1_000_000.0)
    maximum = float(np.max(padded))
    assignment = _hungarian_min(maximum - padded)
    result = [
        (row, column)
        for row, column in assignment
        if row < rows and column < columns and allowed[row, column]
    ]
    result.sort()
    return result
