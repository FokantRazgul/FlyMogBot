"""Image to photoreceptor input.

This is the one place where a fly render and a human face are turned into
network input, and it is deliberately the *same* function for both: if the two
were encoded differently, every comparison downstream would be meaningless. The
brief makes this a hard requirement and the test suite checks it.

What happens here, in order:

1. Canonical framing: square, greyscale, subject occupying a fixed share of the
   field of view.
2. Luminance normalisation, so the score cannot track exposure.
3. Sampling onto the hexagonal lattice through the acceptance-angle blur.
4. Contrast rather than raw luminance, split into ON and OFF channels. In the
   fly's lamina L1 carries ON and L2 carries OFF.
   TODO(verify) the L1/ON and L2/OFF assignment against the FlyWire annotations.
5. Optional microsaccade jitter, because flyvis is motion-tuned and a single
   still frame would only produce an onset transient.

v1 encodes luminance only. Colour (R7/R8) is v2, so eye colour reaches the
network as a brightness difference; that path is covered by its own test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from flymog.config import VisionConfig
from flymog.vision.hexgrid import HexLattice

# Luminance weights for converting RGB to grey (Rec. 709).
_RGB_TO_GREY = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


@dataclass(frozen=True)
class EncodedInput:
    """Network input for one image.

    ``on`` and ``off`` are ``[n_frames, n_columns]`` non-negative contrast
    channels; ``hex_values`` holds the raw sampled luminance for the mosaic that
    goes on the shareable card.
    """

    hex_values: np.ndarray
    on: np.ndarray
    off: np.ndarray
    seed: int

    @property
    def n_frames(self) -> int:
        return self.on.shape[0]

    @property
    def n_columns(self) -> int:
        return self.on.shape[1]

    def rates_hz(self, base_hz: float, gain_hz: float) -> np.ndarray:
        """Poisson rates per channel: ``base + gain * channel``.

        Returns ``[n_frames, 2 * n_columns]`` with ON channels first.
        """
        return base_hz + gain_hz * np.concatenate([self.on, self.off], axis=1)


def image_hash(image: np.ndarray) -> int:
    """Stable 63-bit hash of the preprocessed image.

    Used to seed the Poisson draws so the same file always yields the same
    score, which the brief requires and a test enforces.
    """
    quantised = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    quantised = (quantised * 65535.0).astype(np.uint16)
    digest = hashlib.blake2b(quantised.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def to_greyscale(image: np.ndarray) -> np.ndarray:
    """Convert to a float greyscale image in [0, 1]."""
    arr = np.asarray(image)
    arr = arr.astype(np.float64) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float64)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[2] != 3:
            raise ValueError(f"expected 3 or 4 colour channels, got {arr.shape[2]}")
        arr = arr @ _RGB_TO_GREY
    elif arr.ndim != 2:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {arr.shape}")
    if arr.max() > 1.0 + 1e-6:
        arr = arr / arr.max()
    return np.clip(arr, 0.0, 1.0)


def canonical_frame(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Crop to a centred square and resample to ``size``.

    Callers that already know where the subject is (the face detector in M2, the
    renderer in M1) should crop before calling this; the centre crop is only a
    sane fallback.
    """
    grey = to_greyscale(image)
    h, w = grey.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    square = grey[top : top + side, left : left + side]
    if side == size:
        return square
    # Nearest-neighbour resample: the acceptance-angle blur that follows makes a
    # higher-order filter here pointless.
    idx = np.linspace(0, side - 1, size)
    rows = np.rint(idx).astype(np.int64)
    return square[np.ix_(rows, rows)]


def normalise_luminance(image: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean, unit-std luminance, mapped back into [0, 1].

    Without this the raw score tracks exposure, which the bias audit in M2 would
    then report as the model "preferring" brighter photos.
    """
    mean = float(image.mean())
    std = float(image.std())
    if std < eps:
        return np.full_like(image, 0.5)
    standardised = (image - mean) / std
    return np.clip(standardised * 0.25 + 0.5, 0.0, 1.0)


def _jitter_offsets(n_frames: int, jitter_px: float, seed: int) -> np.ndarray:
    """Small random walk standing in for microsaccades. First frame is centred."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, jitter_px, size=(n_frames, 2))
    steps[0] = 0.0
    return np.cumsum(steps, axis=0)


def _shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Integer-pixel shift with edge clamping."""
    ix, iy = int(round(dx)), int(round(dy))
    if ix == 0 and iy == 0:
        return image
    return np.roll(np.roll(image, iy, axis=0), ix, axis=1)


def encode(
    image: np.ndarray,
    cfg: VisionConfig,
    *,
    lattice: HexLattice | None = None,
    normalise: bool = True,
) -> EncodedInput:
    """Encode one image into ON/OFF contrast on the ommatidial lattice.

    The same call is used for fly renders and human faces.
    """
    lattice = lattice or HexLattice.build(cfg.hex_extent)
    if lattice.n_columns != cfg.n_columns:
        raise ValueError(f"lattice has {lattice.n_columns} columns but config says {cfg.n_columns}")

    framed = canonical_frame(image)
    if normalise:
        framed = normalise_luminance(framed)
    seed = image_hash(framed)

    n_frames = cfg.sequence.n_frames
    offsets = (
        _jitter_offsets(n_frames, cfg.sequence.jitter_px, seed)
        if cfg.sequence.jitter_px > 0
        else np.zeros((n_frames, 2))
    )

    sampled = np.stack(
        [
            lattice.sample(_shift(framed, dx, dy), fov_fraction=cfg.fov_fraction)
            for dx, dy in offsets
        ],
        axis=0,
    )

    # Contrast relative to the mean luminance across the lattice, per frame.
    background = sampled.mean(axis=1, keepdims=True)
    contrast = (sampled - background) / (background + 1e-6)
    on = np.clip(contrast, 0.0, None)
    off = np.clip(-contrast, 0.0, None)

    return EncodedInput(hex_values=sampled[0], on=on, off=off, seed=seed)


def poisson_spikes(rates_hz: np.ndarray, dt_ms: float, seed: int, n_steps: int) -> np.ndarray:
    """Draw deterministic Poisson spike trains from per-channel rates.

    ``rates_hz`` is ``[n_channels]``; the result is ``[n_steps, n_channels]``.
    The seed comes from the image hash, so repeated scoring of the same file
    gives bit-identical input.
    """
    rng = np.random.default_rng(seed)
    p = np.clip(np.asarray(rates_hz, dtype=np.float64) * dt_ms / 1000.0, 0.0, 1.0)
    return (rng.random((n_steps, p.shape[0])) < p).astype(np.float64)
