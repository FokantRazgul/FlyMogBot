"""Connectome representation, loading and the synthetic surrogate.

One internal format is used regardless of where the data came from: a neuron
table plus a signed, thresholded edge list. Everything downstream (LIF, pruning,
probes) talks to :class:`Connectome` only.

The surrogate exists because this development environment cannot reach the hosts
that serve the real connectome. It is structurally similar but scientifically
meaningless, so it carries ``is_surrogate=True`` and every consumer is expected
to say so out loud rather than quietly report surrogate numbers as findings.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from flymog.config import ConnectomeConfig
from flymog.sim.backends import BackendKind, SparseMatmul, build_backend

# Column names we accept for each field, lowercased. FlyWire Codex exports and
# the Zenodo dumps do not agree on spelling, so we match a small set instead of
# hardcoding one. Anything unmatched is an error, never a silent default.
NEURON_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("root_id", "id", "neuron_id", "root id"),
    "cell_type": ("cell_type", "type", "hemibrain_type", "cell type"),
    "super_class": ("super_class", "superclass", "class", "super class"),
    "transmitter": ("nt_type", "neurotransmitter", "transmitter", "nt"),
    "side": ("side", "hemisphere"),
    "x": ("pos_x", "x", "position_x"),
    "y": ("pos_y", "y", "position_y"),
    "z": ("pos_z", "z", "position_z"),
}

EDGE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pre": ("pre_root_id", "pre_pt_root_id", "pre", "source", "pre_id"),
    "post": ("post_root_id", "post_pt_root_id", "post", "target", "post_id"),
    "syn_count": ("syn_count", "synapses", "weight", "count", "n_syn"),
    "transmitter": ("nt_type", "neurotransmitter", "transmitter", "nt"),
}


class ConnectomeDataMissing(RuntimeError):
    """Raised when required connectome files are absent.

    Carries the manual-download instructions, because in restricted network
    environments the fix is always a human action, not a retry.
    """


@dataclass
class Connectome:
    """Neurons and signed synaptic edges, indexed positionally."""

    neuron_ids: np.ndarray
    cell_types: np.ndarray
    super_classes: np.ndarray
    transmitters: np.ndarray
    positions: np.ndarray | None
    pre_idx: np.ndarray
    post_idx: np.ndarray
    syn_count: np.ndarray
    sign: np.ndarray
    source: str = "unknown"
    version: str = "unknown"
    is_surrogate: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.neuron_ids)
        for name in ("cell_types", "super_classes", "transmitters"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")
        if self.positions is not None and self.positions.shape[0] != n:
            raise ValueError("positions must have one row per neuron")
        n_edges = len(self.pre_idx)
        for name in ("post_idx", "syn_count", "sign"):
            if len(getattr(self, name)) != n_edges:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n_edges}")
        if n_edges and (self.pre_idx.max() >= n or self.post_idx.max() >= n):
            raise ValueError("edge endpoints point outside the neuron table")

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre_idx)

    def summary(self) -> dict:
        types, type_counts = np.unique(self.cell_types, return_counts=True)
        classes, class_counts = np.unique(self.super_classes, return_counts=True)
        return {
            "source": self.source,
            "version": self.version,
            "is_surrogate": self.is_surrogate,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "n_cell_types": int(len(types)),
            "n_super_classes": int(len(classes)),
            "has_positions": self.positions is not None,
            "excitatory_edges": int((self.sign > 0).sum()),
            "inhibitory_edges": int((self.sign < 0).sum()),
            "largest_super_classes": dict(
                sorted(
                    zip((str(c) for c in classes), (int(v) for v in class_counts), strict=True),
                    key=lambda kv: -kv[1],
                )[:10]
            ),
            "most_common_cell_types": dict(
                sorted(
                    zip((str(t) for t in types), (int(v) for v in type_counts), strict=True),
                    key=lambda kv: -kv[1],
                )[:10]
            ),
        }

    def indices_of_cell_types(self, patterns: list[str], *, exact: bool = False) -> np.ndarray:
        """Indices of neurons whose cell type matches any of ``patterns``."""
        types = np.asarray([str(t) for t in self.cell_types])
        mask = np.zeros(len(types), dtype=bool)
        lowered = np.char.lower(types.astype(str))
        for pattern in patterns:
            needle = pattern.lower()
            mask |= lowered == needle if exact else np.char.startswith(lowered, needle)
        return np.flatnonzero(mask)

    def weights(self, *, weight_scale: float = 1.0) -> np.ndarray:
        """Signed edge weights: synapse count times sign, times a global scale."""
        return self.sign * self.syn_count.astype(np.float64) * weight_scale

    def to_backend(
        self,
        *,
        kind: BackendKind = "auto",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        weight_scale: float = 1.0,
    ) -> SparseMatmul:
        """Build the ``W @ spikes`` operator, mapping presynaptic to postsynaptic."""
        return build_backend(
            torch.from_numpy(self.post_idx.astype(np.int64)),
            torch.from_numpy(self.pre_idx.astype(np.int64)),
            torch.from_numpy(self.weights(weight_scale=weight_scale).astype(np.float32)),
            (self.n_neurons, self.n_neurons),
            kind=kind,
            device=device,
            dtype=dtype,
        )

    def k_hop_subgraph(self, seeds: np.ndarray, k: int) -> tuple[Connectome, np.ndarray]:
        """Neurons reachable from ``seeds`` within ``k`` synaptic hops.

        Returns the induced subgraph and the mapping from new index to old index,
        so features can be compared against the full graph.
        """
        reached = np.zeros(self.n_neurons, dtype=bool)
        reached[seeds] = True
        frontier = np.asarray(seeds, dtype=np.int64)
        # Sorting by presynaptic index lets each hop use a single searchsorted.
        order = np.argsort(self.pre_idx, kind="stable")
        pre_sorted = self.pre_idx[order]
        post_sorted = self.post_idx[order]
        for _ in range(k):
            if frontier.size == 0:
                break
            starts = np.searchsorted(pre_sorted, frontier, side="left")
            stops = np.searchsorted(pre_sorted, frontier, side="right")
            spans = [post_sorted[a:b] for a, b in zip(starts, stops, strict=True) if b > a]
            if not spans:
                break
            candidates = np.unique(np.concatenate(spans))
            new = candidates[~reached[candidates]]
            reached[new] = True
            frontier = new
        keep = np.flatnonzero(reached)
        return self.subgraph(keep), keep

    def subgraph(self, keep: np.ndarray) -> Connectome:
        """Induced subgraph on ``keep`` (an array of neuron indices)."""
        keep = np.asarray(keep, dtype=np.int64)
        remap = np.full(self.n_neurons, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        edge_mask = (remap[self.pre_idx] >= 0) & (remap[self.post_idx] >= 0)
        return Connectome(
            neuron_ids=self.neuron_ids[keep],
            cell_types=self.cell_types[keep],
            super_classes=self.super_classes[keep],
            transmitters=self.transmitters[keep],
            positions=None if self.positions is None else self.positions[keep],
            pre_idx=remap[self.pre_idx[edge_mask]],
            post_idx=remap[self.post_idx[edge_mask]],
            syn_count=self.syn_count[edge_mask],
            sign=self.sign[edge_mask],
            source=self.source,
            version=self.version,
            is_surrogate=self.is_surrogate,
            meta={**self.meta, "subgraph_of": self.n_neurons},
        )

    def degree_preserving_shuffle(self, seed: int = 0) -> Connectome:
        """Rewire edges keeping each neuron's in- and out-degree.

        This is the control the brief calls for: if the readout does as well on
        this as on the real wiring, the connectome is contributing nothing.
        """
        rng = np.random.default_rng(seed)
        shuffled_post = self.post_idx.copy()
        rng.shuffle(shuffled_post)
        return Connectome(
            neuron_ids=self.neuron_ids,
            cell_types=self.cell_types,
            super_classes=self.super_classes,
            transmitters=self.transmitters,
            positions=self.positions,
            pre_idx=self.pre_idx,
            post_idx=shuffled_post,
            syn_count=self.syn_count,
            sign=self.sign,
            source=self.source,
            version=self.version,
            is_surrogate=self.is_surrogate,
            meta={**self.meta, "degree_preserving_shuffle_seed": seed},
        )
