"""Config parsing. A typo in YAML must fail loudly, not produce a silent default."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from flymog.config import LifConfig, SimConfig, config_dir, load_sim_config


def test_shipped_config_parses():
    cfg = load_sim_config()
    assert cfg.vision.n_columns == 721
    assert cfg.lif.dt_ms > 0


def test_derived_step_counts(lif_cfg):
    assert lif_cfg.n_steps == round(lif_cfg.window_ms / lif_cfg.dt_ms)
    assert lif_cfg.delay_steps == round(lif_cfg.synaptic_delay_ms / lif_cfg.dt_ms)
    assert lif_cfg.refractory_steps == round(lif_cfg.refractory_ms / lif_cfg.dt_ms)


def test_delay_is_at_least_one_step():
    """A delay shorter than one step must not become an instantaneous synapse."""
    cfg = LifConfig(
        v_rest_mv=-52,
        v_reset_mv=-52,
        v_threshold_mv=-45,
        tau_m_ms=20,
        tau_syn_ms=5,
        refractory_ms=2,
        synaptic_delay_ms=0.01,
        w_syn_mv=0.275,
        dt_ms=0.5,
        window_ms=200,
    )
    assert cfg.delay_steps == 1


def test_threshold_below_rest_is_rejected():
    """Otherwise the cell would fire permanently and nothing would warn us."""
    with pytest.raises(ValidationError, match="above v_rest"):
        LifConfig(
            v_rest_mv=-45,
            v_reset_mv=-52,
            v_threshold_mv=-52,
            tau_m_ms=20,
            tau_syn_ms=5,
            refractory_ms=2,
            synaptic_delay_ms=1.8,
            w_syn_mv=0.275,
            dt_ms=0.2,
            window_ms=200,
        )


@pytest.mark.parametrize("field", ["tau_m_ms", "tau_syn_ms", "dt_ms", "window_ms"])
def test_non_positive_time_constants_are_rejected(field):
    values = dict(
        v_rest_mv=-52,
        v_reset_mv=-52,
        v_threshold_mv=-45,
        tau_m_ms=20,
        tau_syn_ms=5,
        refractory_ms=2,
        synaptic_delay_ms=1.8,
        w_syn_mv=0.275,
        dt_ms=0.2,
        window_ms=200,
    )
    values[field] = 0
    with pytest.raises(ValidationError):
        LifConfig(**values)


def test_unknown_backend_kind_is_rejected():
    raw = yaml.safe_load((config_dir() / "sim.yaml").read_text(encoding="utf-8"))
    raw["backend"]["kind"] = "quantum"
    with pytest.raises(ValidationError):
        SimConfig.model_validate(raw)


def test_fov_fraction_must_be_a_fraction():
    raw = yaml.safe_load((config_dir() / "sim.yaml").read_text(encoding="utf-8"))
    raw["vision"]["fov_fraction"] = 1.5
    with pytest.raises(ValidationError):
        SimConfig.model_validate(raw)


def test_unverified_parameters_are_marked_in_the_yaml():
    """Every LIF parameter is unverified, and the file must say so.

    This test exists so that removing a TODO(verify) marker is a deliberate act
    recorded in a diff, not something that quietly rots away.
    """
    text = (config_dir() / "sim.yaml").read_text(encoding="utf-8")
    lif_block = text.split("lif:")[1].split("\nconnectome:")[0]
    for parameter in (
        "v_rest_mv",
        "v_reset_mv",
        "v_threshold_mv",
        "tau_m_ms",
        "tau_syn_ms",
        "refractory_ms",
        "synaptic_delay_ms",
        "w_syn_mv",
    ):
        line = next(ln for ln in lif_block.splitlines() if ln.strip().startswith(parameter))
        assert "TODO(verify)" in line, f"{parameter} is presented as verified but is not"
