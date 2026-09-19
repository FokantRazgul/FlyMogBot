"""Input-cell ladder probe (brief section 5.4).

Asks the connectome which of three entry points actually exists, in order of
biological preference:

1. Photoreceptors R1-R8, tied to ommatidia.
2. Lamina monopolar cells L1/L2/L3, one set per cartridge.
3. Columnar medulla cells (Mi1, Tm3) at column positions.

With flyvis as the visual front end this is no longer the blocking question it
was in the original plan, but the answer still decides how the bridge is built
and what the fallback looks like. Everything reported here is counted from the
data; nothing is assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from flymog.data.connectome import Connectome

# Heuristic acceptance band for "about one cell per column per eye". The fly has
# roughly 700-900 ommatidia per eye; the band is widened below that because the
# optic lobe reconstruction need not be complete, and a type at 670 per eye is
# still plainly columnar. This is a rule of thumb chosen here, not a published
# figure. TODO(verify) against the release's own per-type counts.
EXPECTED_COLUMNS_PER_EYE = (600, 1000)

RUNGS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "photoreceptors",
        ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"),
        "Photoreceptors R1-R8 with ommatidial identity. Most biological entry point.",
    ),
    (
        "lamina_monopolar",
        ("L1", "L2", "L3"),
        "Lamina monopolar cells, one set per cartridge (~one per ommatidium).",
    ),
    (
        "medulla_columnar",
        ("Mi1", "Tm3"),
        "Columnar medulla cells, usable as a per-column proxy.",
    ),
)


@dataclass
class RungReport:
    name: str
    description: str
    patterns: tuple[str, ...]
    total_cells: int
    per_type_counts: dict[str, int]
    has_positions: bool
    estimated_columns: int | None
    observed_type_names: list[str]
    cells_per_subtype_per_eye: float
    plausible_one_per_column: bool
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "patterns": list(self.patterns),
            "total_cells": self.total_cells,
            "per_type_counts": self.per_type_counts,
            "has_positions": self.has_positions,
            "estimated_columns": self.estimated_columns,
            "observed_type_names": self.observed_type_names,
            "cells_per_subtype_per_eye": round(self.cells_per_subtype_per_eye, 1),
            "plausible_one_per_column": self.plausible_one_per_column,
            "verdict": self.verdict,
        }


def _matching_indices(connectome: Connectome, patterns: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Indices per pattern, matching a cell type exactly or with a side suffix.

    A FlyWire cell type is often written ``L1`` but sometimes carries a subtype
    or side marker, so ``L1``, ``L1_L`` and ``L1a`` all count, while ``L10``
    does not. Getting this wrong would silently inflate the counts.
    """
    types = np.asarray([str(t) for t in connectome.cell_types])
    lowered = np.char.lower(types)
    out: dict[str, np.ndarray] = {}
    for pattern in patterns:
        needle = pattern.lower()
        matches = np.zeros(len(lowered), dtype=bool)
        for idx, name in enumerate(lowered):
            if _type_matches(name, needle):
                matches[idx] = True
        out[pattern] = np.flatnonzero(matches)
    return out


def _parse_range(name: str) -> tuple[str, int, int] | None:
    """Parse a merged annotation such as ``R1-6`` into ``("r", 1, 6)``.

    FlyWire does not annotate R2 through R6 individually: the six outer
    photoreceptors share the single cell type ``R1-6``. Treating that as the
    literal type "R1-6" would report R2..R6 as absent, which is how this probe
    first mis-read the real data.
    """
    match = re.fullmatch(r"([a-z]+)(\d+)-(\d+)", name)
    if not match:
        return None
    prefix, low, high = match.group(1), int(match.group(2)), int(match.group(3))
    if low > high:
        return None
    return prefix, low, high


# A permitted suffix is at most one separator and at most one subtype letter:
# "L1_L", "L1-R", "L1a" are all L1. Anything longer is a different cell type.
# Without this bound, "l2LN19" (a lateral neuron) is counted as L2, which is
# exactly what happened when this probe was first run against the real data.
_ALLOWED_SUFFIX = re.compile(r"^[_\- ]?[a-z]?$")


def _type_matches(name: str, needle: str) -> bool:
    """Does a FlyWire cell type correspond to the canonical type ``needle``?

    Accepts the exact name, a side or subtype suffix (``L1_L``, ``L1a``), and a
    merged numeric range that covers it (``R1-6`` covers ``R3``). Rejects a
    longer number, so ``L10`` never counts as ``L1``, and rejects an arbitrary
    continuation, so ``l2LN19`` never counts as ``L2``.
    """
    if name == needle:
        return True

    covered = _parse_range(name)
    if covered is not None:
        prefix, low, high = covered
        target = re.fullmatch(r"([a-z]+)(\d+)", needle)
        return bool(target and target.group(1) == prefix and low <= int(target.group(2)) <= high)

    if name.startswith(needle) and len(name) > len(needle):
        return bool(_ALLOWED_SUFFIX.fullmatch(name[len(needle) :]))
    return False


def estimate_columns(positions: np.ndarray, max_clusters: int = 1000) -> int | None:
    """Estimate how many distinct retinotopic columns a set of cells spans.

    Uses nearest-neighbour spacing: cells belonging to one column sit much
    closer together than neighbouring columns do. Returns ``None`` when there is
    not enough data to say anything, or when the grouping runs past
    ``max_clusters`` and so has not found column structure at all.
    """
    if positions is None or positions.shape[0] < 10:
        return None
    finite = positions[np.isfinite(positions).all(axis=1)]
    if finite.shape[0] < 10:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover
        return None
    tree = cKDTree(finite)
    # Distance to the nearest other cell, per cell.
    dists, _ = tree.query(finite, k=2)
    nn = dists[:, 1]
    threshold = float(np.median(nn)) * 1.5
    if threshold <= 0:
        return None
    # Single-linkage grouping at that threshold approximates column membership.
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    parent = np.arange(finite.shape[0])

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    groups = len({find(i) for i in range(finite.shape[0])})
    if groups >= max_clusters:
        # Hitting the cap means the grouping did not converge on anything
        # column-like. Returning the cap would look like a measurement.
        return None
    return groups


def probe_input_cells(connectome: Connectome) -> dict[str, Any]:
    """Walk the ladder and report what is available."""
    reports: list[RungReport] = []
    for name, patterns, description in RUNGS:
        matches = _matching_indices(connectome, patterns)
        per_type = {k: len(v) for k, v in matches.items()}
        all_idx = (
            np.unique(np.concatenate([v for v in matches.values() if len(v)]))
            if any(len(v) for v in matches.values())
            else np.asarray([], dtype=np.int64)
        )
        total = len(all_idx)
        positions = connectome.positions
        has_pos = positions is not None and total > 0
        estimated = (
            estimate_columns(positions[all_idx]) if positions is not None and total else None
        )

        observed = sorted(
            {str(connectome.cell_types[i]) for i in all_idx.tolist()},
            key=str.lower,
        )

        # Density check: one cell per column per eye. The subtypes covered are
        # counted from the canonical patterns that actually matched, not from the
        # annotation strings, because a merged type like R1-6 is one string
        # standing for six subtypes and would otherwise make the density look
        # six times too high.
        lo, hi = EXPECTED_COLUMNS_PER_EYE
        n_subtypes_present = sum(1 for v in matches.values() if len(v) > 0)
        per_subtype_per_eye = total / max(1, n_subtypes_present) / 2
        plausible = bool(lo <= per_subtype_per_eye <= hi) if n_subtypes_present else False

        if total == 0:
            verdict = "absent: no cells of these types in the annotations"
        elif not has_pos:
            verdict = "present but without positions: columns cannot be assigned spatially"
        elif plausible:
            verdict = "usable: counts are consistent with one cell per column per eye"
        else:
            verdict = (
                "present but counts do not look one-per-column; "
                "check annotation completeness before relying on it"
            )

        reports.append(
            RungReport(
                name=name,
                description=description,
                patterns=patterns,
                total_cells=total,
                per_type_counts=per_type,
                has_positions=bool(has_pos),
                estimated_columns=estimated,
                observed_type_names=observed,
                cells_per_subtype_per_eye=per_subtype_per_eye,
                plausible_one_per_column=plausible,
                verdict=verdict,
            )
        )

    chosen = next(
        (r.name for r in reports if r.verdict.startswith("usable")),
        None,
    )
    return {
        "connectome": connectome.summary(),
        "is_surrogate": connectome.is_surrogate,
        "surrogate_warning": (
            "These counts come from the SURROGATE connectome and say nothing about "
            "real fly anatomy. Re-run against the real data."
            if connectome.is_surrogate
            else None
        ),
        "rungs": [r.as_dict() for r in reports],
        "recommended_rung": chosen,
        "recommendation": (
            f"Use the '{chosen}' rung as the connectome entry point."
            if chosen
            else "No rung is usable as-is. Fall back to flyvis for the visual front end."
        ),
    }
