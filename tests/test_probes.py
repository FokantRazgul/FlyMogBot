"""Probes: the input ladder, the bridge match, the regime sweep and the report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flymog.data.connectome import Connectome
from flymog.probes.bridge import (
    BRIDGE_GO_THRESHOLD,
    drop_subtype,
    match_cell_types,
    normalise_type,
    probe_bridge,
)
from flymog.probes.inputs import probe_input_cells
from flymog.probes.regime import participation_ratio
from flymog.probes.report import build_m0_report


def _connectome_with_types(types: list[str], *, positions: bool = True) -> Connectome:
    n = len(types)
    return Connectome(
        neuron_ids=np.arange(n),
        cell_types=np.asarray(types, dtype=object),
        super_classes=np.asarray(["optic"] * n, dtype=object),
        transmitters=np.asarray(["acetylcholine"] * n, dtype=object),
        positions=np.random.default_rng(0).normal(size=(n, 3)) if positions else None,
        pre_idx=np.asarray([], dtype=np.int64),
        post_idx=np.asarray([], dtype=np.int64),
        syn_count=np.asarray([], dtype=np.int64),
        sign=np.asarray([], dtype=np.float64),
    )


# --- input ladder ---------------------------------------------------------


def test_ladder_reports_absent_types():
    result = probe_input_cells(_connectome_with_types(["Foo", "Bar"]))
    assert all(r["total_cells"] == 0 for r in result["rungs"])
    assert result["recommended_rung"] is None
    assert "flyvis" in result["recommendation"]


def test_ladder_finds_photoreceptors_at_plausible_counts():
    """Roughly 800 cells per type per eye is the one-per-column signature."""
    types = [f"R{i}" for i in range(1, 9) for _ in range(1600)]
    result = probe_input_cells(_connectome_with_types(types))
    photoreceptors = next(r for r in result["rungs"] if r["name"] == "photoreceptors")
    assert photoreceptors["total_cells"] == 12_800
    assert photoreceptors["plausible_one_per_column"] is True
    assert result["recommended_rung"] == "photoreceptors"


def test_ladder_matches_side_suffixes_but_not_other_numbers():
    """L1_L and L1_R are L1; L10 is a different cell type entirely."""
    types = ["L1_L", "L1_R", "L10", "L10"]
    result = probe_input_cells(_connectome_with_types(types))
    lamina = next(r for r in result["rungs"] if r["name"] == "lamina_monopolar")
    assert lamina["per_type_counts"]["L1"] == 2


def test_ladder_flags_missing_positions():
    types = [f"R{i}" for i in range(1, 9) for _ in range(1600)]
    result = probe_input_cells(_connectome_with_types(types, positions=False))
    photoreceptors = next(r for r in result["rungs"] if r["name"] == "photoreceptors")
    assert photoreceptors["has_positions"] is False
    assert "without positions" in photoreceptors["verdict"]


def test_surrogate_warning_is_present_on_surrogate_data(small_surrogate):
    result = probe_input_cells(small_surrogate)
    assert result["is_surrogate"] is True
    assert "SURROGATE" in result["surrogate_warning"]


# --- bridge ---------------------------------------------------------------


def test_name_normalisation():
    assert normalise_type("L1_L") == "l1"
    assert normalise_type("Mi1") == "mi1"
    assert normalise_type("T4a") == "t4a"
    assert drop_subtype("t4a") == "t4"
    assert drop_subtype("tm3") == "tm3"  # 3 is not a subtype letter


def test_match_kinds_are_ranked_exact_first():
    results, stats = match_cell_types(["L1", "T4a"], ["L1", "T4"])
    kinds = {r.flyvis_type: r.match_kind for r in results}
    assert kinds["L1"] == "exact"
    assert kinds["T4a"] == "subtype"
    assert stats["coverage"] == 1.0


def test_match_is_one_to_many_across_hemispheres():
    """A flyvis type maps onto every FlyWire entry that normalises to it."""
    results, stats = match_cell_types(["L1"], ["L1_L", "L1_R", "L1_L"])
    assert results[0].matched_flywire_types == ["L1_L", "L1_R"]
    assert results[0].n_flywire_neurons == 3
    assert stats["n_flywire_neurons_reached"] == 3


def test_unmatched_types_are_listed():
    _, stats = match_cell_types(["L1", "Nonesuch"], ["L1"])
    assert stats["unmatched_flyvis_types"] == ["Nonesuch"]
    assert stats["coverage"] == 0.5


def test_bridge_verdict_follows_the_threshold():
    connectome = _connectome_with_types(["L1", "L2", "Mi1"])
    good = probe_bridge(connectome, ["L1", "L2", "Mi1", "Zzz"])
    assert good["stats"]["coverage"] >= BRIDGE_GO_THRESHOLD
    assert good["verdict"] == "go"

    poor = probe_bridge(connectome, ["Aaa", "Bbb", "Ccc", "Ddd", "L1"])
    assert poor["verdict"] == "no-go"
    assert "directly on" in poor["recommendation"]


def test_bridge_refuses_to_guess_when_flyvis_is_absent(monkeypatch):
    """Without the real type list, coverage would be meaningless, so we stop."""
    import flymog.vision.flyvis_frontend as frontend

    monkeypatch.setattr(frontend, "find_spec", lambda name: None)
    result = probe_bridge(_connectome_with_types(["L1"]), None)
    assert result["ran"] is False
    assert "not installed" in result["reason"]


# --- regime ---------------------------------------------------------------


def test_participation_ratio_bounds():
    rng = np.random.default_rng(0)
    rank_one = rng.normal(size=(300, 1)) @ rng.normal(size=(1, 30))
    assert participation_ratio(rank_one) == pytest.approx(1.0, abs=0.01)

    isotropic = participation_ratio(rng.normal(size=(2000, 20)))
    assert 15.0 < isotropic <= 20.0


def test_participation_ratio_handles_degenerate_input():
    assert np.isnan(participation_ratio(np.zeros((5, 3))))
    assert np.isnan(participation_ratio(np.ones((1, 3))))


# --- report ---------------------------------------------------------------


def test_report_marks_missing_sections_rather_than_hiding_them(tmp_path):
    text, found, missing = build_m0_report(tmp_path)
    assert found == []
    assert "probe_env" in missing
    assert "Не заполнено" in text
    assert "Go/no-go пока не вынесен" in text


def test_report_fills_a_section_it_has_data_for(tmp_path):
    (tmp_path / "probe_env.json").write_text(
        json.dumps(
            {
                "platform": {
                    "system": "Darwin",
                    "release": "24.0",
                    "machine": "arm64",
                    "python": "3.11.9",
                },
                "cpu": {"logical_cores": 12, "physical_cores": 12},
                "memory": {"total_gb": 18.0},
                "torch": {"version": "2.4.0", "cuda_available": False, "mps_available": True},
                "selected_device": "mps",
                "auto_backend": "gather",
                "nvidia": {"present": False},
            }
        ),
        encoding="utf-8",
    )
    text, found, missing = build_m0_report(tmp_path)
    assert found == ["probe_env"]
    assert "Darwin" in text and "mps" in text
    assert "probe_bridge" in missing


def test_report_raises_a_surrogate_banner(tmp_path):
    (tmp_path / "bench_sim.json").write_text(
        json.dumps(
            {
                "device": "cpu",
                "n_neurons": 10,
                "n_edges": 20,
                "is_surrogate": True,
                "target": {"images_per_second": 1.0, "batch": 32},
                "rows": [],
                "n_configurations_meeting_target": 0,
                "verdict": "none",
            }
        ),
        encoding="utf-8",
    )
    text, _, _ = build_m0_report(tmp_path)
    assert "суррогате" in text


# --- cell-type name matching against real FlyWire spellings -----------------


def test_merged_range_covers_every_subtype():
    """FlyWire annotates the outer photoreceptors as one type, ``R1-6``.

    Reading that literally reports R2..R6 as absent, which is what this probe
    did on its first run against the real data.
    """
    types = ["R1-6"] * 8452 + ["R7"] * 1342 + ["R8"] * 1324
    result = probe_input_cells(_connectome_with_types(types))
    photoreceptors = next(r for r in result["rungs"] if r["name"] == "photoreceptors")
    assert photoreceptors["total_cells"] == 11_118
    for subtype in ("R1", "R3", "R6"):
        assert photoreceptors["per_type_counts"][subtype] == 8452
    assert photoreceptors["observed_type_names"] == ["R1-6", "R7", "R8"]
    assert result["recommended_rung"] == "photoreceptors"


def test_lateral_neurons_are_not_counted_as_lamina_cells():
    """``l2LN19`` is a lateral neuron, not the lamina monopolar cell L2.

    A loose prefix rule silently added several hundred of these to the L2 count.
    """
    types = ["L2"] * 1699 + ["l2LN19"] * 10 + ["l2LN20"] * 8
    result = probe_input_cells(_connectome_with_types(types))
    lamina = next(r for r in result["rungs"] if r["name"] == "lamina_monopolar")
    assert lamina["per_type_counts"]["L2"] == 1699
    assert "l2LN19" not in lamina["observed_type_names"]


def test_longer_numbers_are_different_cell_types():
    types = ["L1"] * 5 + ["L10"] * 5 + ["Tm3"] * 5 + ["Tm30"] * 5
    result = probe_input_cells(_connectome_with_types(types))
    lamina = next(r for r in result["rungs"] if r["name"] == "lamina_monopolar")
    medulla = next(r for r in result["rungs"] if r["name"] == "medulla_columnar")
    assert lamina["per_type_counts"]["L1"] == 5
    assert medulla["per_type_counts"]["Tm3"] == 5


def test_side_and_subtype_suffixes_still_match():
    types = ["Mi1_L"] * 3 + ["Mi1-R"] * 3 + ["Mi1a"] * 2
    result = probe_input_cells(_connectome_with_types(types))
    medulla = next(r for r in result["rungs"] if r["name"] == "medulla_columnar")
    assert medulla["per_type_counts"]["Mi1"] == 8


def test_range_parsing_rejects_nonsense():
    from flymog.probes.inputs import _parse_range, _type_matches

    assert _parse_range("r1-6") == ("r", 1, 6)
    assert _parse_range("r6-1") is None  # inverted range
    assert _parse_range("tm3") is None
    assert _type_matches("l1-3", "l2") is True
    assert _type_matches("l1-3", "l5") is False
