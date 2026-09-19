"""Operating-regime sweep.

A reservoir is useless when it is silent and equally useless when it is
saturated: in both cases every input produces the same output. The brief asks
for a sweep over input gain and global weight scale, keeping firing rates of
responsive neurons in the 5-50 Hz band and choosing the point that maximises the
effective dimensionality of the features, measured as participation ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from flymog.config import LifConfig
from flymog.data.connectome import Connectome
from flymog.sim.backends import select_device
from flymog.sim.lif import LifNetwork


def participation_ratio(features: np.ndarray) -> float:
    """Effective dimensionality of a ``[n_samples, n_features]`` matrix.

    ``PR = (sum of eigenvalues)^2 / sum of squared eigenvalues`` of the
    covariance. It equals 1 when all variance sits on one axis and equals the
    rank when variance is spread evenly. Computed from singular values, which
    avoids forming a huge covariance matrix.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    centred = x - x.mean(axis=0, keepdims=True)
    if not np.isfinite(centred).all():
        return float("nan")
    singular = np.linalg.svd(centred, compute_uv=False)
    eigenvalues = singular**2
    total = eigenvalues.sum()
    if total <= 0:
        return float("nan")
    return float(total**2 / (eigenvalues**2).sum())


def sweep_regime(
    connectome: Connectome,
    cfg: LifConfig,
    *,
    input_gains: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
    weight_scales: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
    batch: int = 16,
    device: torch.device | None = None,
    seed: int = 0,
    target_rate_hz: tuple[float, float] = (5.0, 50.0),
) -> dict[str, Any]:
    """Sweep gain and weight scale, returning every point and the chosen one."""
    device = device or select_device()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # A fixed set of input patterns, reused at every sweep point so the
    # comparison is about the network, not about the stimulus.
    patterns = torch.rand((connectome.n_neurons, batch), generator=generator).to(device)

    rows: list[dict[str, Any]] = []
    for weight_scale in weight_scales:
        backend = connectome.to_backend(device=device)
        for gain in input_gains:
            net = LifNetwork(backend, cfg, weight_scale=weight_scale, device=device)
            result = net.run(patterns * gain, batch_size=batch)
            rates = result.rates_hz

            responsive = rates > 0
            frac_responsive = float(responsive.any(dim=1).to(torch.float64).mean())
            responsive_rates = rates[responsive]
            mean_rate = float(responsive_rates.mean()) if responsive_rates.numel() else 0.0
            max_rate = float(rates.max()) if rates.numel() else 0.0

            features = rates.t().double().cpu().numpy()
            pr = participation_ratio(features)

            in_band = bool(target_rate_hz[0] <= mean_rate <= target_rate_hz[1])
            rows.append(
                {
                    "input_gain": gain,
                    "global_weight_scale": weight_scale,
                    "fraction_responsive": round(frac_responsive, 4),
                    "mean_responsive_rate_hz": round(mean_rate, 3),
                    "max_rate_hz": round(max_rate, 3),
                    "participation_ratio": None if np.isnan(pr) else round(pr, 4),
                    "in_target_band": in_band,
                }
            )

    candidates = [r for r in rows if r["in_target_band"] and r["participation_ratio"] is not None]
    best = max(candidates, key=lambda r: r["participation_ratio"], default=None)

    # Firing rate is a spike count divided by the window, so the shortest
    # non-zero rate the window can express is 1 / window. If that is coarse
    # relative to the target band, the rates are quantised and the sweep is
    # choosing between a handful of levels rather than a continuum.
    # Participation ratio is bounded by the number of samples minus one. With a
    # small batch it saturates at that ceiling for every sweep point and stops
    # discriminating between them, which is easy to mistake for "all points are
    # equally good".
    pr_ceiling = float(batch - 1)
    measured_pr = [r["participation_ratio"] for r in rows if r["participation_ratio"] is not None]
    pr_saturated = bool(measured_pr and min(measured_pr) >= 0.95 * pr_ceiling)

    resolution_hz = 1000.0 / (cfg.n_steps * cfg.dt_ms)
    band_width = target_rate_hz[1] - target_rate_hz[0]
    distinguishable_levels = band_width / resolution_hz

    return {
        "device": device.type,
        "batch": batch,
        "participation_ratio_ceiling": pr_ceiling,
        "participation_ratio_saturated": pr_saturated,
        "rate_resolution_hz": round(resolution_hz, 3),
        "distinguishable_levels_in_band": round(distinguishable_levels, 1),
        "is_surrogate": connectome.is_surrogate,
        "surrogate_warning": (
            "Swept on the SURROGATE connectome. The chosen operating point is "
            "meaningless for the real network and must be re-swept on real data."
            if connectome.is_surrogate
            else None
        ),
        "target_rate_hz": list(target_rate_hz),
        "criterion": "participation_ratio",
        "rows": rows,
        "chosen": best,
        "verdict": (
            "found an operating point inside the target rate band"
            if best
            else "no swept point kept rates inside the target band; widen the sweep"
        ),
    }
