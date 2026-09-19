"""Speed benchmark and subgraph-pruning equivalence check.

The brief's target is at least one image per second at a batch of 32 or more.
This module measures that rather than estimating it, across backends and across
the ``dt`` and window settings the brief allows trading off.

It also checks the acceleration ladder's most invasive rung: pruning the graph
to what is reachable from the input cells within k hops is only legitimate if
the features it produces still match the full graph, so the correlation between
the two is measured and reported.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from flymog.config import LifConfig
from flymog.data.connectome import Connectome
from flymog.sim.backends import BackendKind, resolve_backend_kind, select_device
from flymog.sim.lif import LifNetwork

TARGET_IMAGES_PER_SECOND = 1.0
TARGET_BATCH = 32


def _timed_steps(net: LifNetwork, drive: torch.Tensor, n_steps: int, warmup: int = 3) -> float:
    """Median seconds per step, after a warmup."""
    net.reset(drive.shape[1])
    for _ in range(warmup):
        net.step(drive)
    timings = []
    for _ in range(n_steps):
        start = time.perf_counter()
        net.step(drive)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def benchmark_backends(
    connectome: Connectome,
    cfg: LifConfig,
    *,
    batch_sizes: tuple[int, ...] = (32,),
    dt_values_ms: tuple[float, ...] = (0.1, 0.2, 0.5),
    window_values_ms: tuple[float, ...] = (100.0, 200.0, 300.0),
    kinds: tuple[BackendKind, ...] = ("csr", "gather"),
    device: torch.device | None = None,
    measured_steps: int = 20,
    seed: int = 0,
    drive_mv_scale: float = 10.0,
) -> dict[str, Any]:
    """Measure throughput for each backend and timing configuration.

    Only the per-step cost is measured directly; full-window throughput is that
    cost times the step count, which is exact because every step costs the same.
    """
    device = device or select_device()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows: list[dict[str, Any]] = []

    for kind in kinds:
        try:
            backend = connectome.to_backend(kind=kind, device=device)
        except (RuntimeError, NotImplementedError) as exc:
            rows.append({"backend": kind, "error": f"{type(exc).__name__}: {exc}"})
            continue

        for batch in batch_sizes:
            # Drive must straddle threshold, or the network is silent and the
            # timings measure an unrealistically sparse spike pattern.
            drive = (
                torch.rand((connectome.n_neurons, batch), generator=generator) * drive_mv_scale
            ).to(device)
            for dt_ms in dt_values_ms:
                step_cfg = cfg.model_copy(update={"dt_ms": dt_ms})
                net = LifNetwork(backend, step_cfg, device=device)
                try:
                    per_step = _timed_steps(net, drive, measured_steps)
                except (RuntimeError, NotImplementedError) as exc:
                    rows.append(
                        {
                            "backend": kind,
                            "batch": batch,
                            "dt_ms": dt_ms,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                for window_ms in window_values_ms:
                    n_steps = int(round(window_ms / dt_ms))
                    window_seconds = per_step * n_steps
                    rows.append(
                        {
                            "backend": kind,
                            "batch": batch,
                            "dt_ms": dt_ms,
                            "window_ms": window_ms,
                            "n_steps": n_steps,
                            "ms_per_step": round(per_step * 1000.0, 3),
                            "seconds_per_window": round(window_seconds, 3),
                            "images_per_second": round(batch / window_seconds, 4),
                            "meets_target": bool(
                                batch >= TARGET_BATCH
                                and batch / window_seconds >= TARGET_IMAGES_PER_SECOND
                            ),
                        }
                    )

    ok = [r for r in rows if r.get("meets_target")]
    best = max(
        (r for r in rows if "images_per_second" in r),
        key=lambda r: r["images_per_second"],
        default=None,
    )
    return {
        "device": device.type,
        "auto_backend": resolve_backend_kind("auto", device),
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "is_surrogate": connectome.is_surrogate,
        "surrogate_warning": (
            "Timings come from the SURROGATE connectome. Its size and degree "
            "distribution are realistic, so the timings are indicative, but they "
            "are not measurements of the real connectome."
            if connectome.is_surrogate
            else None
        ),
        "target": {
            "images_per_second": TARGET_IMAGES_PER_SECOND,
            "batch": TARGET_BATCH,
        },
        "rows": rows,
        "n_configurations_meeting_target": len(ok),
        "best": best,
        "verdict": (
            "target met by at least one configuration"
            if ok
            else "target not met by any configuration measured here"
        ),
    }


def pruning_equivalence(
    connectome: Connectome,
    cfg: LifConfig,
    seed_indices: np.ndarray,
    *,
    k_hops: int = 5,
    batch: int = 4,
    device: torch.device | None = None,
    seed: int = 0,
    min_correlation: float = 0.99,
    drive_mv_scale: float = 10.0,
) -> dict[str, Any]:
    """Compare firing rates between the full graph and its k-hop subgraph.

    The same external drive is applied to the same neurons in both runs, so any
    difference comes from the edges pruning removed. Correlation is computed over
    the neurons the two graphs share.
    """
    device = device or select_device()
    subgraph, kept = connectome.k_hop_subgraph(np.asarray(seed_indices, dtype=np.int64), k_hops)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    full_drive = (
        torch.rand((connectome.n_neurons, batch), generator=generator) * drive_mv_scale
    ).to(device)
    sub_drive = full_drive[torch.from_numpy(kept).to(device)]

    full_net = LifNetwork(connectome.to_backend(device=device), cfg, device=device)
    full_rates = full_net.run(full_drive, batch_size=batch).rates_hz[
        torch.from_numpy(kept).to(device)
    ]

    sub_net = LifNetwork(subgraph.to_backend(device=device), cfg, device=device)
    sub_rates = sub_net.run(sub_drive, batch_size=batch).rates_hz

    a = full_rates.flatten().double().cpu().numpy()
    b = sub_rates.flatten().double().cpu().numpy()
    if a.std() < 1e-12 or b.std() < 1e-12:
        correlation = float("nan")
        note = "at least one run produced constant rates, so correlation is undefined"
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])
        note = ""

    return {
        "k_hops": k_hops,
        "n_seed_cells": len(seed_indices),
        "full_n_neurons": connectome.n_neurons,
        "pruned_n_neurons": subgraph.n_neurons,
        "pruned_n_edges": subgraph.n_edges,
        "neuron_reduction": round(1.0 - subgraph.n_neurons / connectome.n_neurons, 4),
        "edge_reduction": round(1.0 - subgraph.n_edges / max(1, connectome.n_edges), 4),
        "rate_correlation": None if np.isnan(correlation) else round(correlation, 6),
        "min_correlation": min_correlation,
        "equivalent": bool(not np.isnan(correlation) and correlation >= min_correlation),
        "note": note,
        "is_surrogate": connectome.is_surrogate,
    }
