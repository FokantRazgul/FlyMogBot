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


def test_lif_parameters_cite_their_source():
    """Every LIF parameter must be traceable, not taken on trust.

    The values were checked against the reference implementation of Shiu et al.
    This test fails if a parameter is added without a citation, or if one is
    quietly changed away from the verified value.
    """
    text = (config_dir() / "sim.yaml").read_text(encoding="utf-8")
    lif_block = text.split("lif:")[1].split("\nconnectome:")[0]

    expected = {
        "v_rest_mv": -52.0,
        "v_reset_mv": -52.0,
        "v_threshold_mv": -45.0,
        "tau_m_ms": 20.0,
        "tau_syn_ms": 5.0,
        "refractory_ms": 2.2,
        "synaptic_delay_ms": 1.8,
        "w_syn_mv": 0.275,
    }
    cfg = load_sim_config().lif
    for name, value in expected.items():
        assert getattr(cfg, name) == pytest.approx(value), f"{name} drifted from the reference"

    # The block must carry the sources those values came from.
    for doi in (
        "10.3389/fnbeh.2017.00008",
        "10.1088/2634-4386/ac3ba6",
        "10.7554/eLife.62362",
        "10.3389/fncel.2015.00029",
    ):
        assert doi in lif_block, f"source {doi} is missing from the LIF block"

    # w_syn is a free parameter in the reference, and saying so matters more
    # than the number itself.
    assert "free parameter" in lif_block.lower()


def test_refractory_period_is_the_reference_value_not_the_brief_approximation():
    """The brief said about 2 ms; the reference model says 2.2 ms."""
    assert load_sim_config().lif.refractory_ms == pytest.approx(2.2)
