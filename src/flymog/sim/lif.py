"""Vectorised leaky integrate-and-fire network over a connectome.

State is ``[n_neurons, batch]`` so a batch of images is simulated at once, and
connectivity is applied through one of the sparse backends in
:mod:`flymog.sim.backends`.

Numerical convention, stated explicitly because the brief's parameters are not
yet verified against the reference model:

* Membrane and synaptic currents are expressed in millivolts. ``w_syn_mv`` is
  the increment a single presynaptic spike adds to the postsynaptic synaptic
  current, signed by the presynaptic neurotransmitter.
* Both state variables are integrated exactly over a step, assuming the driving
  term is constant within that step:
  ``I <- I * exp(-dt/tau_syn)`` then ``I += w * spikes``, and
  ``V <- v_rest + (V - v_rest) * exp(-dt/tau_m) + I_total * (1 - exp(-dt/tau_m))``.
* A neuron spikes when ``V >= v_threshold``; it is then reset to ``v_reset`` and
  clamped there for the refractory period.

The absolute scale of ``w_syn_mv`` is calibrated by ``flymog sweep-regime`` via
``regime.global_weight_scale``, so an error in its units shows up as a scale
factor, not as silently wrong dynamics. See TODO(verify) in configs/sim.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flymog.config import LifConfig
from flymog.sim.backends import SparseMatmul


@dataclass
class SimulationResult:
    """Outcome of a run.

    ``spike_counts`` is ``[n_neurons, batch]`` over the whole window;
    ``window_counts`` is ``[n_windows, n_neurons, batch]`` when window recording
    was requested, otherwise ``None``.
    """

    spike_counts: torch.Tensor
    window_counts: torch.Tensor | None
    n_steps: int
    dt_ms: float

    @property
    def rates_hz(self) -> torch.Tensor:
        duration_s = self.n_steps * self.dt_ms / 1000.0
        return self.spike_counts / duration_s

    def window_rates_hz(self) -> torch.Tensor:
        if self.window_counts is None:
            raise ValueError("run() was called without window recording")
        steps_per_window = self.n_steps / self.window_counts.shape[0]
        duration_s = steps_per_window * self.dt_ms / 1000.0
        return self.window_counts / duration_s


class LifNetwork:
    """A frozen LIF reservoir driven by external current."""

    def __init__(
        self,
        connectivity: SparseMatmul,
        cfg: LifConfig,
        *,
        weight_scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        n_post, n_pre = connectivity.shape
        if n_post != n_pre:
            raise ValueError(
                f"connectivity must be square (n_neurons x n_neurons), got {connectivity.shape}"
            )
        self.connectivity = connectivity
        self.cfg = cfg
        self.n_neurons = n_post
        self.weight_scale = weight_scale
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.alpha_m = math.exp(-cfg.dt_ms / cfg.tau_m_ms)
        self.alpha_syn = math.exp(-cfg.dt_ms / cfg.tau_syn_ms)
        self.delay_steps = cfg.delay_steps
        self.refractory_steps = cfg.refractory_steps

        self._v: torch.Tensor | None = None
        self._i_syn: torch.Tensor | None = None
        self._refractory: torch.Tensor | None = None
        self._delay_buffer: torch.Tensor | None = None
        self._delay_pos = 0

    def reset(self, batch_size: int) -> None:
        """Initialise state for a batch. Resting, silent, no spikes in flight."""
        shape = (self.n_neurons, batch_size)
        self._v = torch.full(shape, self.cfg.v_rest_mv, device=self.device, dtype=self.dtype)
        self._i_syn = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._refractory = torch.zeros(shape, device=self.device, dtype=torch.int32)
        self._delay_buffer = torch.zeros(
            (self.delay_steps, *shape), device=self.device, dtype=self.dtype
        )
        self._delay_pos = 0

    def step(self, external_current_mv: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Advance one time step, returning the binary spike tensor."""
        if self._v is None or self._i_syn is None or self._refractory is None:
            raise RuntimeError("call reset(batch_size) before step()")
        assert self._delay_buffer is not None

        # Spikes emitted delay_steps ago arrive now.
        arriving = self._delay_buffer[self._delay_pos]
        synaptic_input = self.connectivity(arriving) * (self.cfg.w_syn_mv * self.weight_scale)

        self._i_syn = self._i_syn * self.alpha_syn + synaptic_input
        drive = self._i_syn + external_current_mv

        v_rest = self.cfg.v_rest_mv
        v_new = v_rest + (self._v - v_rest) * self.alpha_m + drive * (1.0 - self.alpha_m)

        # Refractory cells are held at reset and cannot integrate.
        in_refractory = self._refractory > 0
        v_new = torch.where(in_refractory, torch.full_like(v_new, self.cfg.v_reset_mv), v_new)

        spikes = (v_new >= self.cfg.v_threshold_mv) & ~in_refractory
        spikes_f = spikes.to(self.dtype)

        v_new = torch.where(spikes, torch.full_like(v_new, self.cfg.v_reset_mv), v_new)
        self._v = v_new
        self._refractory = torch.where(
            spikes,
            torch.full_like(self._refractory, self.refractory_steps),
            torch.clamp(self._refractory - 1, min=0),
        )

        # Queue this step's spikes for delivery after the synaptic delay.
        self._delay_buffer[self._delay_pos] = spikes_f
        self._delay_pos = (self._delay_pos + 1) % self.delay_steps
        return spikes_f

    def run(
        self,
        external_current_mv: torch.Tensor | float = 0.0,
        *,
        batch_size: int | None = None,
        n_steps: int | None = None,
        n_windows: int | None = None,
        reset: bool = True,
    ) -> SimulationResult:
        """Run the window and return spike counts.

        ``external_current_mv`` is either a scalar or a ``[n_neurons, batch]``
        tensor held constant for the whole run.
        """
        steps = n_steps if n_steps is not None else self.cfg.n_steps
        if reset:
            if batch_size is None:
                if isinstance(external_current_mv, torch.Tensor):
                    batch_size = external_current_mv.shape[1]
                else:
                    raise ValueError("batch_size is required when the drive is a scalar")
            self.reset(batch_size)
        assert self._v is not None
        batch = self._v.shape[1]

        counts = torch.zeros((self.n_neurons, batch), device=self.device, dtype=self.dtype)
        window_counts = None
        if n_windows:
            if steps % n_windows != 0:
                raise ValueError(f"n_steps={steps} is not divisible by n_windows={n_windows}")
            window_counts = torch.zeros(
                (n_windows, self.n_neurons, batch), device=self.device, dtype=self.dtype
            )
            steps_per_window = steps // n_windows

        for step_idx in range(steps):
            spikes = self.step(external_current_mv)
            counts += spikes
            if window_counts is not None:
                window_counts[step_idx // steps_per_window] += spikes

        return SimulationResult(
            spike_counts=counts,
            window_counts=window_counts,
            n_steps=steps,
            dt_ms=self.cfg.dt_ms,
        )


def analytic_rate_hz(cfg: LifConfig, drive_mv: float) -> float:
    """Firing rate of one isolated LIF cell under a constant drive.

    Used by the tests as ground truth. Returns 0.0 when the drive cannot reach
    threshold, since the cell then never fires.
    """
    v_gap = cfg.v_threshold_mv - cfg.v_reset_mv
    steady = cfg.v_rest_mv + drive_mv
    if steady <= cfg.v_threshold_mv:
        return 0.0
    # Time for V to climb from v_reset to threshold under an exponential approach.
    numerator = steady - cfg.v_reset_mv
    time_to_threshold = cfg.tau_m_ms * math.log(numerator / (numerator - v_gap))
    period_ms = time_to_threshold + cfg.refractory_ms
    return 1000.0 / period_ms
