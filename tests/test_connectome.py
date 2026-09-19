"""Connectome structure, subgraphs and the degree-preserving control."""

from __future__ import annotations

import numpy as np
import pytest

from flymog.config import ConnectomeConfig
from flymog.data.connectome import Connectome
from flymog.data.download import make_surrogate_connectome


def _toy() -> Connectome:
    """0 -> 1 -> 2, plus an isolated neuron 3."""
    return Connectome(
        neuron_ids=np.arange(4),
        cell_types=np.asarray(["R1", "L1", "Mi1", "Tm3"], dtype=object),
        super_classes=np.asarray(["optic"] * 4, dtype=object),
        transmitters=np.asarray(["acetylcholine"] * 4, dtype=object),
        positions=np.zeros((4, 3)),
        pre_idx=np.asarray([0, 1]),
        post_idx=np.asarray([1, 2]),
        syn_count=np.asarray([10, 10]),
        sign=np.asarray([1.0, 1.0]),
    )


def test_inconsistent_tables_are_rejected():
    with pytest.raises(ValueError, match="rows"):
        Connectome(
            neuron_ids=np.arange(3),
            cell_types=np.asarray(["a", "b"], dtype=object),
            super_classes=np.asarray(["x"] * 3, dtype=object),
            transmitters=np.asarray(["ach"] * 3, dtype=object),
            positions=None,
            pre_idx=np.asarray([]),
            post_idx=np.asarray([]),
            syn_count=np.asarray([]),
            sign=np.asarray([]),
        )


def test_edges_pointing_outside_the_neuron_table_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        Connectome(
            neuron_ids=np.arange(2),
            cell_types=np.asarray(["a", "b"], dtype=object),
            super_classes=np.asarray(["x", "x"], dtype=object),
            transmitters=np.asarray(["ach", "ach"], dtype=object),
            positions=None,
            pre_idx=np.asarray([0]),
            post_idx=np.asarray([5]),
            syn_count=np.asarray([1]),
            sign=np.asarray([1.0]),
        )


@pytest.mark.parametrize("k,expected", [(1, 2), (2, 3), (5, 3)])
def test_k_hop_reaches_exactly_the_right_neurons(k, expected):
    toy = _toy()
    sub, kept = toy.k_hop_subgraph(np.asarray([0]), k)
    assert sub.n_neurons == expected
    assert 3 not in kept  # the isolated neuron is never reachable


def test_subgraph_keeps_only_internal_edges():
    toy = _toy()
    sub = toy.subgraph(np.asarray([0, 1]))
    assert sub.n_neurons == 2
    assert sub.n_edges == 1  # the 1 -> 2 edge leaves the subgraph


def test_cell_type_lookup():
    toy = _toy()
    assert toy.indices_of_cell_types(["R1"], exact=True).tolist() == [0]
    assert toy.indices_of_cell_types(["Tm"], exact=False).tolist() == [3]


def test_degree_preserving_shuffle_keeps_both_degree_sequences():
    """The ablation control is only valid if the degrees really are preserved."""
    original = make_surrogate_connectome(500, 4000, seed=11)
    shuffled = original.degree_preserving_shuffle(seed=3)

    for attr in ("pre_idx", "post_idx"):
        before = np.bincount(getattr(original, attr), minlength=original.n_neurons)
        after = np.bincount(getattr(shuffled, attr), minlength=original.n_neurons)
        np.testing.assert_array_equal(before, after)
    # And the wiring really did change.
    assert not np.array_equal(original.post_idx, shuffled.post_idx)


def test_surrogate_is_labelled_as_such():
    """Nothing should be able to mistake surrogate output for real data."""
    surrogate = make_surrogate_connectome(200, 1000, seed=1)
    assert surrogate.is_surrogate is True
    assert "SURROGATE" in surrogate.meta["warning"]
    assert surrogate.summary()["is_surrogate"] is True
    assert all(str(t).startswith("SURROGATE_") for t in surrogate.cell_types[:20])


def test_surrogate_is_reproducible():
    a = make_surrogate_connectome(300, 1500, seed=5)
    b = make_surrogate_connectome(300, 1500, seed=5)
    np.testing.assert_array_equal(a.pre_idx, b.pre_idx)
    np.testing.assert_array_equal(a.post_idx, b.post_idx)


def test_surrogate_has_no_self_loops():
    surrogate = make_surrogate_connectome(400, 3000, seed=2)
    assert not np.any(surrogate.pre_idx == surrogate.post_idx)


def test_transmitter_signs():
    cfg = ConnectomeConfig(
        min_synapse_count=5,
        excitatory_transmitters=["acetylcholine", "ach"],
        inhibitory_transmitters=["gaba", "glutamate", "glu"],
        unknown_transmitter_policy="drop",
    )
    assert cfg.sign_for("ACh") == 1.0
    assert cfg.sign_for("GABA") == -1.0
    assert cfg.sign_for("glutamate") == -1.0
    # An unknown transmitter must not be silently guessed at.
    assert cfg.sign_for("dopamine") is None
    assert cfg.sign_for(None) is None


def test_unknown_transmitter_policies():
    base = dict(
        min_synapse_count=5,
        excitatory_transmitters=["ach"],
        inhibitory_transmitters=["gaba"],
    )
    assert ConnectomeConfig(**base, unknown_transmitter_policy="excitatory").sign_for("x") == 1.0
    assert ConnectomeConfig(**base, unknown_transmitter_policy="zero").sign_for("x") == 0.0


def test_weights_carry_sign_and_synapse_count():
    toy = _toy()
    weights = toy.weights(weight_scale=2.0)
    np.testing.assert_allclose(weights, [20.0, 20.0])
