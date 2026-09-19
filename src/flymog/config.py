"""Configuration loading for FlyMog.

Configs live in ``configs/*.yaml`` and are parsed into pydantic models so that a
typo in a YAML file fails loudly at startup instead of silently producing a
nonsense simulation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


def repo_root() -> Path:
    """Return the repository root, i.e. the directory that holds ``configs/``."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    """Directory for large downloads. Never committed to git."""
    override = os.environ.get("FLYMOG_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data"


class LifConfig(BaseModel):
    """Leaky integrate-and-fire parameters.

    Values are unverified starting points from the project brief; see the
    TODO(verify) markers in ``configs/sim.yaml``.
    """

    v_rest_mv: float
    v_reset_mv: float
    v_threshold_mv: float
    tau_m_ms: float = Field(gt=0)
    tau_syn_ms: float = Field(gt=0)
    refractory_ms: float = Field(ge=0)
    synaptic_delay_ms: float = Field(ge=0)
    w_syn_mv: float
    dt_ms: float = Field(gt=0)
    window_ms: float = Field(gt=0)

    @field_validator("v_threshold_mv")
    @classmethod
    def _threshold_above_rest(cls, v: float, info):
        rest = info.data.get("v_rest_mv")
        if rest is not None and v <= rest:
            raise ValueError("v_threshold_mv must be above v_rest_mv, else the cell never fires")
        return v

    @property
    def n_steps(self) -> int:
        return int(round(self.window_ms / self.dt_ms))

    @property
    def delay_steps(self) -> int:
        """Synaptic delay in simulation steps, at least one."""
        return max(1, int(round(self.synaptic_delay_ms / self.dt_ms)))

    @property
    def refractory_steps(self) -> int:
        return int(round(self.refractory_ms / self.dt_ms))


class ConnectomeConfig(BaseModel):
    min_synapse_count: int = Field(ge=0)
    excitatory_transmitters: list[str]
    inhibitory_transmitters: list[str]
    unknown_transmitter_policy: Literal["drop", "excitatory", "zero"] = "drop"

    def sign_for(self, transmitter: str | None) -> float | None:
        """Map a neurotransmitter name to a synaptic sign.

        Returns ``None`` when the transmitter is unknown and the policy says to
        drop the edge rather than guess at its sign.
        """
        name = (transmitter or "").strip().lower()
        if name in {t.lower() for t in self.excitatory_transmitters}:
            return 1.0
        if name in {t.lower() for t in self.inhibitory_transmitters}:
            return -1.0
        if self.unknown_transmitter_policy == "excitatory":
            return 1.0
        if self.unknown_transmitter_policy == "zero":
            return 0.0
        return None


class RegimeConfig(BaseModel):
    input_gain: float
    global_weight_scale: float
    target_rate_hz: tuple[float, float]
    criterion: Literal["participation_ratio"] = "participation_ratio"


class PruningConfig(BaseModel):
    enabled: bool = False
    k_hops: int = Field(ge=1)
    equivalence_min_correlation: float = Field(ge=0.0, le=1.0)


class SequenceConfig(BaseModel):
    n_frames: int = Field(ge=1)
    fps: int = Field(gt=0)
    jitter_px: float = Field(ge=0)
    jitter_seed_from_image: bool = True


class VisionConfig(BaseModel):
    hex_extent: int = Field(ge=1)
    n_columns: int = Field(ge=1)
    fov_fraction: float = Field(gt=0.0, le=1.0)
    sequence: SequenceConfig


class DeterminismConfig(BaseModel):
    seed_from_image_hash: bool = True
    n_trials: int = Field(ge=1)


class BackendConfig(BaseModel):
    kind: Literal["auto", "csr", "gather"] = "auto"
    dtype: Literal["float32", "float16"] = "float32"


class SimConfig(BaseModel):
    lif: LifConfig
    connectome: ConnectomeConfig
    regime: RegimeConfig
    pruning: PruningConfig
    vision: VisionConfig
    determinism: DeterminismConfig
    backend: BackendConfig


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_sim_config(path: Path | None = None) -> SimConfig:
    """Load ``configs/sim.yaml`` (or an explicit path) into a validated model."""
    target = path or (config_dir() / "sim.yaml")
    return SimConfig.model_validate(load_yaml(target))


@lru_cache(maxsize=1)
def default_sim_config() -> SimConfig:
    return load_sim_config()
