"""LIF dynamics: analytic agreement, delays, signs and determinism."""

from __future__ import annotations

import pytest
import torch

from flymog.sim.backends import build_backend
from flymog.sim.lif import LifNetwork, analytic_rate_hz


def _disconnected(n: int, cpu: torch.device):
    """A network with neurons but no synapses."""
    empty = torch.zeros(0, dtype=torch.int64)
    return build_backend(empty, empty, torch.zeros(0), (n, n), kind="gather", device=cpu)


@pytest.mark.parametrize("drive", [8.0, 12.0, 20.0])
def test_single_neuron_matches_analytic_rate(lif_cfg, cpu, drive):
    """An isolated cell under constant current must fire at the analytic rate.

    Tolerance is 3%: the simulation advances in 0.2 ms steps, so threshold
    crossings are quantised and a small bias is expected.
    """
    net = LifNetwork(_disconnected(1, cpu), lif_cfg, device=cpu)
    result = net.run(external_current_mv=drive, batch_size=1, n_steps=20_000)
    simulated = float(result.rates_hz[0, 0])
    expected = analytic_rate_hz(lif_cfg, drive)
    assert expected > 0
    assert simulated == pytest.approx(expected, rel=0.03)


def test_subthreshold_drive_is_silent(lif_cfg, cpu):
    """Below threshold the cell must never fire, not fire rarely."""
    gap = lif_cfg.v_threshold_mv - lif_cfg.v_rest_mv
    net = LifNetwork(_disconnected(1, cpu), lif_cfg, device=cpu)
    result = net.run(external_current_mv=gap * 0.9, batch_size=1, n_steps=5_000)
    assert float(result.spike_counts.sum()) == 0.0
    assert analytic_rate_hz(lif_cfg, gap * 0.9) == 0.0


def test_refractory_period_caps_the_rate(lif_cfg, cpu):
    """However hard it is driven, a cell cannot exceed 1 / refractory period."""
    net = LifNetwork(_disconnected(1, cpu), lif_cfg, device=cpu)
    result = net.run(external_current_mv=500.0, batch_size=1, n_steps=10_000)
    ceiling = 1000.0 / lif_cfg.refractory_ms
    assert float(result.rates_hz[0, 0]) <= ceiling + 1e-6


def test_synaptic_delay_is_respected(lif_cfg, cpu):
    """A postsynaptic cell cannot respond before the synaptic delay elapses."""
    row = torch.tensor([1])  # post
    col = torch.tensor([0])  # pre
    backend = build_backend(row, col, torch.tensor([1.0]), (2, 2), kind="gather", device=cpu)
    # Weight big enough that one presynaptic spike alone drives the target.
    net = LifNetwork(backend, lif_cfg, weight_scale=500.0, device=cpu)
    net.reset(1)

    drive = torch.tensor([[20.0], [0.0]])  # only neuron 0 is driven
    first_pre = first_post = None
    for step in range(400):
        spikes = net.step(drive)
        if first_pre is None and spikes[0, 0] > 0:
            first_pre = step
        if first_post is None and spikes[1, 0] > 0:
            first_post = step
            break

    assert first_pre is not None, "the driven neuron never fired"
    assert first_post is not None, "the target neuron never fired"
    assert first_post - first_pre >= lif_cfg.delay_steps


def test_inhibition_lowers_the_rate_excitation_raises_it(lif_cfg, cpu):
    """Sign of the synapse must actually change the target's firing."""
    row = torch.tensor([1])
    col = torch.tensor([0])
    drive = torch.tensor([[20.0], [9.0]])

    rates = {}
    for label, weight in (("excitatory", 1.0), ("inhibitory", -1.0)):
        backend = build_backend(row, col, torch.tensor([weight]), (2, 2), kind="gather", device=cpu)
        net = LifNetwork(backend, lif_cfg, weight_scale=20.0, device=cpu)
        rates[label] = float(net.run(drive, batch_size=1, n_steps=6_000).rates_hz[1, 0])

    backend = build_backend(row, col, torch.tensor([0.0]), (2, 2), kind="gather", device=cpu)
    baseline_net = LifNetwork(backend, lif_cfg, device=cpu)
    baseline = float(baseline_net.run(drive, batch_size=1, n_steps=6_000).rates_hz[1, 0])

    assert rates["excitatory"] > baseline
    assert rates["inhibitory"] < baseline


def test_run_is_deterministic(lif_cfg, cpu, small_surrogate):
    """The same input must give bit-identical spike counts every time."""
    backend = small_surrogate.to_backend(kind="csr", device=cpu)
    drive = torch.full((small_surrogate.n_neurons, 3), 10.0)
    first = LifNetwork(backend, lif_cfg, weight_scale=0.01, device=cpu).run(drive, batch_size=3)
    second = LifNetwork(backend, lif_cfg, weight_scale=0.01, device=cpu).run(drive, batch_size=3)
    torch.testing.assert_close(first.spike_counts, second.spike_counts, rtol=0, atol=0)


def test_batch_entries_are_independent(lif_cfg, cpu):
    """Each image in a batch must be simulated separately.

    Running two drives together must give exactly what running them apart does;
    otherwise batching would silently leak activity between users' photos.
    """
    net_shared = LifNetwork(_disconnected(2, cpu), lif_cfg, device=cpu)
    together = net_shared.run(
        torch.tensor([[9.0, 30.0], [30.0, 9.0]]), batch_size=2, n_steps=2_000
    ).spike_counts

    separate = []
    for column in ([[9.0], [30.0]], [[30.0], [9.0]]):
        net = LifNetwork(_disconnected(2, cpu), lif_cfg, device=cpu)
        separate.append(net.run(torch.tensor(column), batch_size=1, n_steps=2_000).spike_counts)
    expected = torch.cat(separate, dim=1)
    torch.testing.assert_close(together, expected, rtol=0, atol=0)


def test_window_counts_sum_to_total(lif_cfg, cpu):
    net = LifNetwork(_disconnected(4, cpu), lif_cfg, device=cpu)
    result = net.run(external_current_mv=15.0, batch_size=2, n_steps=1_000, n_windows=5)
    assert result.window_counts is not None
    torch.testing.assert_close(result.window_counts.sum(dim=0), result.spike_counts)
    assert result.window_rates_hz().shape == (5, 4, 2)


def test_step_before_reset_is_an_error(lif_cfg, cpu):
    net = LifNetwork(_disconnected(2, cpu), lif_cfg, device=cpu)
    with pytest.raises(RuntimeError, match="reset"):
        net.step(1.0)


def test_non_square_connectivity_is_rejected(lif_cfg, cpu):
    empty = torch.zeros(0, dtype=torch.int64)
    backend = build_backend(empty, empty, torch.zeros(0), (3, 4), kind="gather", device=cpu)
    with pytest.raises(ValueError, match="square"):
        LifNetwork(backend, lif_cfg, device=cpu)
