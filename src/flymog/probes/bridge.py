"""Bridge probe: can flyvis outputs be wired into the FlyWire central brain?

flyvis and FlyWire are different datasets with different identifiers. The only
practical join is by cell-type name, and that join is an approximation. This
probe measures how good it is instead of assuming it works, and the number it
produces is what the M0 go/no-go decision rests on.

Go/no-go threshold from the approved plan: if fewer than a third of the flyvis
cell types find a FlyWire counterpart, the bridge is not viable and the readout
should sit directly on flyvis activity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from flymog.data.connectome import Connectome

BRIDGE_GO_THRESHOLD = 1.0 / 3.0

# Side markers and subtype suffixes that should not block a match.
_SIDE_SUFFIX = re.compile(r"[_\-\s]?(l|r|left|right|lhs|rhs)$", re.IGNORECASE)
_SUBTYPE_SUFFIX = re.compile(r"(?<=[A-Za-z0-9])[a-d]$")


def normalise_type(name: str) -> str:
    """Normalise a cell-type name for matching.

    Lowercases, drops a trailing side marker and collapses separators. Subtype
    letters (T4a..T4d) are handled separately, since collapsing them is lossy and
    we want to count exact and relaxed matches apart.
    """
    cleaned = str(name).strip()
    cleaned = _SIDE_SUFFIX.sub("", cleaned)
    cleaned = re.sub(r"[_\-\s]+", "", cleaned)
    return cleaned.lower()


def drop_subtype(name: str) -> str:
    """Collapse T4a/T4b/T4c/T4d to T4 and similar."""
    return _SUBTYPE_SUFFIX.sub("", name)


@dataclass
class MatchResult:
    """One flyvis type and every FlyWire type it maps onto.

    A flyvis type routinely corresponds to several FlyWire entries (the two
    hemispheres, or subtypes), so the match is one-to-many. Keeping only the
    first would undercount the neurons the bridge actually reaches.
    """

    flyvis_type: str
    matched_flywire_types: list[str]
    match_kind: str  # exact | normalised | subtype | none
    n_flywire_neurons: int

    @property
    def matched_flywire_type(self) -> str | None:
        """Representative match, for compact reporting."""
        return self.matched_flywire_types[0] if self.matched_flywire_types else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "flyvis_type": self.flyvis_type,
            "matched_flywire_types": self.matched_flywire_types,
            "match_kind": self.match_kind,
            "n_flywire_neurons": self.n_flywire_neurons,
        }


def match_cell_types(
    flyvis_types: list[str], flywire_types: list[str]
) -> tuple[list[MatchResult], dict[str, Any]]:
    """Match two cell-type vocabularies, exactly then progressively more loosely."""
    counts: dict[str, int] = {}
    for name in flywire_types:
        key = str(name)
        counts[key] = counts.get(key, 0) + 1

    by_exact: dict[str, list[str]] = {}
    by_norm: dict[str, list[str]] = {}
    by_sub: dict[str, list[str]] = {}
    for original in counts:
        by_exact.setdefault(original, []).append(original)
        by_norm.setdefault(normalise_type(original), []).append(original)
        by_sub.setdefault(drop_subtype(normalise_type(original)), []).append(original)

    results: list[MatchResult] = []
    for flyvis_type in flyvis_types:
        targets: list[str] = []
        kind = "none"
        if flyvis_type in by_exact:
            targets, kind = by_exact[flyvis_type], "exact"
        elif normalise_type(flyvis_type) in by_norm:
            targets, kind = by_norm[normalise_type(flyvis_type)], "normalised"
        elif drop_subtype(normalise_type(flyvis_type)) in by_sub:
            targets, kind = by_sub[drop_subtype(normalise_type(flyvis_type))], "subtype"
        results.append(
            MatchResult(
                flyvis_type=flyvis_type,
                matched_flywire_types=sorted(targets),
                match_kind=kind,
                n_flywire_neurons=int(sum(counts[t] for t in targets)),
            )
        )

    matched = [r for r in results if r.match_kind != "none"]
    by_kind: dict[str, int] = {}
    for r in results:
        by_kind[r.match_kind] = by_kind.get(r.match_kind, 0) + 1

    coverage = len(matched) / len(flyvis_types) if flyvis_types else 0.0
    stats = {
        "n_flyvis_types": len(flyvis_types),
        "n_flywire_types": len(counts),
        "n_matched": len(matched),
        "coverage": round(coverage, 4),
        "matches_by_kind": by_kind,
        "n_flywire_neurons_reached": int(sum(r.n_flywire_neurons for r in matched)),
        "unmatched_flyvis_types": [r.flyvis_type for r in results if r.match_kind == "none"],
    }
    return results, stats


def probe_bridge(
    connectome: Connectome,
    flyvis_types: list[str] | None = None,
    *,
    only_output_types: bool = True,
) -> dict[str, Any]:
    """Run the bridge probe, or explain why it cannot run.

    By default only the types flyvis marks as leaving the optic lobe are
    matched, because those are the ones that have to reach the central brain.
    Matching all 65 types would inflate coverage with cells that never project
    anywhere near FlyWire's central neuropils.

    ``flyvis_types`` may be supplied explicitly (for example dumped from an
    install on another machine). When it is absent we read the list from flyvis
    itself and refuse to guess if that fails.
    """
    source = "provided by caller"
    if flyvis_types is None:
        from flymog.vision.flyvis_frontend import (
            cell_types,
            flyvis_availability,
            output_cell_types,
        )

        availability = flyvis_availability()
        if not availability.installed:
            return {
                "ran": False,
                "reason": "flyvis is not installed, so its cell-type list is unknown",
                "flyvis": availability.as_dict(),
                "next_step": (
                    "Install the flyvis extra and re-run, or pass a dumped type list "
                    "with --flyvis-types-file."
                ),
            }
        try:
            flyvis_types = output_cell_types() if only_output_types else cell_types()
            scope = "output units" if only_output_types else "all node types"
            source = f"flyvis {availability.version} ({scope})"
        except RuntimeError as exc:
            return {
                "ran": False,
                "reason": str(exc),
                "flyvis": availability.as_dict(),
            }

    flywire_types = [str(t) for t in connectome.cell_types if str(t)]
    results, stats = match_cell_types(flyvis_types, flywire_types)

    # Where do the matched neurons live? A bridge that only reaches optic-lobe
    # neurons is not a bridge into the central brain.
    matched_types = {t for r in results for t in r.matched_flywire_types}
    type_array = np.asarray([str(t) for t in connectome.cell_types])
    mask = (
        np.isin(type_array, list(matched_types))
        if matched_types
        else np.zeros(len(type_array), bool)
    )
    classes, class_counts = np.unique(connectome.super_classes[mask], return_counts=True)

    # A bridge landing only on optic-lobe neurons is not a bridge into the
    # central brain, so projection classes are counted separately.
    projection_classes = {"visual_projection", "visual_centrifugal", "central"}
    n_projection = int(
        sum(
            int(count)
            for name, count in zip(classes, class_counts, strict=True)
            if str(name) in projection_classes
        )
    )

    n_matched_neurons = stats["n_flywire_neurons_reached"]
    projection_share = n_projection / n_matched_neurons if n_matched_neurons else 0.0

    # How the bridge should be wired depends on WHERE the matched neurons sit,
    # not only on how many type names line up. flyvis's "output units" are
    # mostly intrinsic optic-lobe types (Tm, TmY, T4/T5), which FlyWire files
    # under `optic`, not `visual_projection`. Treating them as projection
    # neurons would skip the optic lobe's own wiring; injecting into the
    # matching FlyWire optic neurons and letting FlyWire route onward does not.
    if projection_share >= 0.5:
        wiring = "as_projection_neurons"
        wiring_note = (
            "Most matched neurons are projection neurons, so flyvis outputs can be "
            "treated as the projection stage directly."
        )
    else:
        wiring = "inject_into_matching_optic_neurons"
        wiring_note = (
            f"Only {n_projection} of {n_matched_neurons} matched neurons "
            f"({projection_share:.1%}) are projection or central neurons; the rest are "
            "intrinsic optic-lobe cells. Do not treat flyvis outputs as visual "
            "projection neurons. Inject flyvis activity into the FlyWire neurons of "
            "the matching types and let FlyWire's own connectivity carry it to the "
            "projection neurons and the central brain."
        )

    go = stats["coverage"] >= BRIDGE_GO_THRESHOLD
    return {
        "ran": True,
        "only_output_types": only_output_types,
        "n_matched_neurons_in_projection_classes": n_projection,
        "projection_share_of_matched_neurons": round(projection_share, 4),
        "wiring": wiring,
        "wiring_note": wiring_note,
        "flyvis_type_source": source,
        "is_surrogate": connectome.is_surrogate,
        "surrogate_warning": (
            "Matched against the SURROGATE connectome, whose cell-type names are "
            "deliberately fake. Coverage here is meaningless; re-run on real data."
            if connectome.is_surrogate
            else None
        ),
        "stats": stats,
        "matched_neuron_super_classes": dict(
            zip((str(c) for c in classes), (int(v) for v in class_counts), strict=True)
        ),
        "go_threshold": BRIDGE_GO_THRESHOLD,
        "verdict": "go" if go else "no-go",
        "recommendation": (
            "Bridge is viable: wire the matched types into the central brain."
            if go
            else "Bridge coverage is below threshold. Put the readout directly on "
            "flyvis activity and record this in docs/SCIENCE.md."
        ),
        "matches": [r.as_dict() for r in results],
    }
