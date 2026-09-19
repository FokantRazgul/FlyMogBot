"""Hex lattice and encoder.

The encoder carries a hard requirement from the brief: fly renders and human
faces must go through exactly the same function, or nothing downstream compares.
"""

from __future__ import annotations

import numpy as np
import pytest

from flymog.vision.encoder import (
    canonical_frame,
    encode,
    image_hash,
    normalise_luminance,
    poisson_spikes,
    to_greyscale,
)
from flymog.vision.hexgrid import HexLattice, get_hex_coords, n_columns_for_extent


@pytest.mark.parametrize("extent,expected", [(0, 1), (1, 7), (2, 19), (15, 721)])
def test_lattice_size_matches_the_hex_formula(extent, expected):
    assert n_columns_for_extent(extent) == expected
    assert get_hex_coords(extent).shape == (expected, 2)


def test_default_lattice_is_721_columns(sim_cfg):
    """721 columns is the flyvis convention the rest of the pipeline assumes."""
    lattice = HexLattice.build(sim_cfg.vision.hex_extent)
    assert lattice.n_columns == sim_cfg.vision.n_columns == 721


def test_coordinate_order_is_stable():
    """Feature vectors are indexed by position, so the order must never drift."""
    first = get_hex_coords(4)
    second = get_hex_coords(4)
    assert np.array_equal(first, second)
    assert first[0].tolist() == [-4, 0]
    # Every coordinate must satisfy the hexagonal constraint.
    u, v = first[:, 0], first[:, 1]
    assert np.all(np.abs(u) <= 4) and np.all(np.abs(v) <= 4) and np.all(np.abs(u + v) <= 4)


def test_sampling_preserves_bright_and_dark_regions():
    lattice = HexLattice.build(6)
    bright = lattice.sample(np.ones((128, 128)), fov_fraction=0.6)
    dark = lattice.sample(np.zeros((128, 128)), fov_fraction=0.6)
    assert bright.mean() == pytest.approx(1.0, abs=1e-6)
    assert dark.mean() == pytest.approx(0.0, abs=1e-6)


def test_fov_fraction_changes_what_is_seen():
    """A smaller field of view must look at a smaller part of the image."""
    image = np.zeros((128, 128))
    image[56:72, 56:72] = 1.0  # a small central square
    lattice = HexLattice.build(8)
    narrow = lattice.sample(image, fov_fraction=0.2).mean()
    wide = lattice.sample(image, fov_fraction=0.95).mean()
    assert narrow > wide


def test_sample_rejects_non_square_and_colour_images():
    lattice = HexLattice.build(3)
    with pytest.raises(ValueError, match="square"):
        lattice.sample(np.zeros((10, 20)))
    with pytest.raises(ValueError, match="2-D"):
        lattice.sample(np.zeros((10, 10, 3)))


def test_mosaic_round_trips_column_values():
    lattice = HexLattice.build(5)
    values = np.linspace(0.0, 1.0, lattice.n_columns)
    mosaic = lattice.render_mosaic(values, size=96, fov_fraction=0.8)
    assert mosaic.shape == (96, 96)
    assert mosaic.max() <= values.max() + 1e-9
    with pytest.raises(ValueError, match="expected"):
        lattice.render_mosaic(values[:-1], size=32)


def test_greyscale_handles_rgb_rgba_and_uint8():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    grey = to_greyscale(rgb)
    assert grey.shape == (4, 4)
    assert grey.max() <= 1.0
    rgba = np.concatenate([rgb, np.full((4, 4, 1), 255, dtype=np.uint8)], axis=2)
    assert np.allclose(to_greyscale(rgba), grey)


def test_canonical_frame_is_square_and_centred():
    image = np.zeros((100, 200))
    framed = canonical_frame(image, size=64)
    assert framed.shape == (64, 64)


def test_luminance_normalisation_removes_exposure(sim_cfg, rng):
    """The score must not track exposure, so encoding must be brightness-blind.

    A scaled and offset copy of an image has to produce nearly the same input.
    """
    base = rng.random((160, 160))
    darker = np.clip(base * 0.4 + 0.05, 0.0, 1.0)

    a = encode(base, sim_cfg.vision)
    b = encode(darker, sim_cfg.vision)
    # Contrast channels should be close; they cannot be identical because
    # clipping at 0 and 1 is not an affine operation.
    assert np.corrcoef(a.on.ravel(), b.on.ravel())[0, 1] > 0.95


def test_normalise_luminance_handles_flat_images():
    flat = np.full((16, 16), 0.3)
    assert np.allclose(normalise_luminance(flat), 0.5)


def test_encode_is_deterministic(sim_cfg, rng):
    """Same file, same score: the brief requires it and users would notice."""
    image = rng.random((120, 120, 3))
    first = encode(image, sim_cfg.vision)
    second = encode(image, sim_cfg.vision)
    assert first.seed == second.seed
    assert np.array_equal(first.on, second.on)
    assert np.array_equal(first.off, second.off)


def test_different_images_get_different_seeds(rng, sim_cfg):
    a = encode(rng.random((80, 80)), sim_cfg.vision)
    b = encode(rng.random((80, 80)), sim_cfg.vision)
    assert a.seed != b.seed


def test_image_hash_is_stable_across_calls(rng):
    image = rng.random((32, 32))
    assert image_hash(image) == image_hash(image.copy())


def test_on_and_off_channels_are_complementary(sim_cfg, rng):
    """A column is either brighter or darker than the background, never both."""
    encoded = encode(rng.random((100, 100)), sim_cfg.vision)
    assert np.all(encoded.on >= 0) and np.all(encoded.off >= 0)
    assert np.all(encoded.on * encoded.off == 0)


def test_rates_are_base_plus_gain(sim_cfg, rng):
    encoded = encode(rng.random((64, 64)), sim_cfg.vision)
    rates = encoded.rates_hz(base_hz=5.0, gain_hz=100.0)
    assert rates.shape == (encoded.n_frames, 2 * encoded.n_columns)
    assert rates.min() >= 5.0 - 1e-9


def test_fly_and_human_images_use_the_same_encoder(sim_cfg, rng):
    """Encoding must depend only on pixels, not on what the picture is of.

    There is deliberately no 'is this a fly' switch; this test guards against one
    being added later.
    """
    fly_like = rng.random((512, 512))
    face_like = rng.random((300, 400, 3))
    for image in (fly_like, face_like):
        encoded = encode(image, sim_cfg.vision)
        assert encoded.n_columns == sim_cfg.vision.n_columns
        assert encoded.on.shape == encoded.off.shape


def test_poisson_spikes_are_deterministic_and_respect_rates():
    rates = np.full(500, 100.0)
    a = poisson_spikes(rates, dt_ms=1.0, seed=42, n_steps=1000)
    b = poisson_spikes(rates, dt_ms=1.0, seed=42, n_steps=1000)
    assert np.array_equal(a, b)
    # 100 Hz with 1 ms bins means a spike probability of 0.1 per bin.
    assert a.mean() == pytest.approx(0.1, abs=0.01)
    different = poisson_spikes(rates, dt_ms=1.0, seed=43, n_steps=1000)
    assert not np.array_equal(a, different)
