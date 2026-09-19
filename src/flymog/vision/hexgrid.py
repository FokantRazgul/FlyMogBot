"""Hexagonal ommatidial lattice: coordinates, sampling and mosaic rendering.

The lattice follows the flyvis convention so the two can be used together: axial
coordinates ``(u, v)`` with ``|u| <= extent``, ``|v| <= extent`` and
``|u + v| <= extent``, enumerated u-then-v. ``extent=15`` gives 721 columns,
which is the model's coverage of the fly's central visual field.

Sampling an image is two steps, matching fly optics: blur by the ommatidial
acceptance angle, then read one value per column. Encoding contrast rather than
raw luminance happens one level up, in :mod:`flymog.vision.encoder`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQRT3 = float(np.sqrt(3.0))


def n_columns_for_extent(extent: int) -> int:
    """Number of hexagons in a hexagonal region of the given radius."""
    return 3 * extent * extent + 3 * extent + 1


def get_hex_coords(extent: int) -> np.ndarray:
    """Axial coordinates of every column, ordered u-then-v.

    Returns an ``[n_columns, 2]`` integer array. The ordering is fixed and must
    stay stable, because feature vectors are indexed by position in it.
    """
    coords = []
    for u in range(-extent, extent + 1):
        v_min = max(-extent, -extent - u)
        v_max = min(extent, extent - u)
        for v in range(v_min, v_max + 1):
            coords.append((u, v))
    out = np.asarray(coords, dtype=np.int64)
    expected = n_columns_for_extent(extent)
    if out.shape[0] != expected:
        raise AssertionError(f"built {out.shape[0]} columns, expected {expected}")
    return out


def hex_to_cartesian(coords: np.ndarray) -> np.ndarray:
    """Map axial coordinates to unit-spaced cartesian centres (pointy-top)."""
    u = coords[:, 0].astype(np.float64)
    v = coords[:, 1].astype(np.float64)
    x = SQRT3 * (u + v / 2.0)
    y = 1.5 * v
    return np.stack([x, y], axis=1)


@dataclass(frozen=True)
class HexLattice:
    """A fixed lattice plus the image-space geometry used to sample it."""

    extent: int
    coords: np.ndarray
    centres: np.ndarray

    @classmethod
    def build(cls, extent: int = 15) -> HexLattice:
        coords = get_hex_coords(extent)
        return cls(extent=extent, coords=coords, centres=hex_to_cartesian(coords))

    @property
    def n_columns(self) -> int:
        return self.coords.shape[0]

    def pixel_centres(self, image_size: int, fov_fraction: float) -> np.ndarray:
        """Column centres in pixel coordinates for a square image.

        ``fov_fraction`` is the share of the image the lattice spans. It is the
        knob that makes a fly head and a human face occupy the same portion of
        the visual field, which the brief requires for the two to be comparable.
        """
        if not 0.0 < fov_fraction <= 1.0:
            raise ValueError("fov_fraction must be in (0, 1]")
        radius = np.abs(self.centres).max()
        if radius == 0:
            raise ValueError("degenerate lattice")
        scale = (image_size * fov_fraction / 2.0) / radius
        return self.centres * scale + image_size / 2.0

    def sample(
        self,
        image: np.ndarray,
        *,
        fov_fraction: float = 0.65,
        acceptance_sigma_columns: float = 0.5,
    ) -> np.ndarray:
        """Sample a square greyscale image onto the lattice.

        The image is blurred by a Gaussian standing in for the ommatidial
        acceptance angle, then read at each column centre with bilinear
        interpolation. Returns one float per column.
        """
        if image.ndim != 2:
            raise ValueError(f"expected a 2-D greyscale image, got shape {image.shape}")
        if image.shape[0] != image.shape[1]:
            raise ValueError(f"expected a square image, got {image.shape}")
        size = image.shape[0]
        centres = self.pixel_centres(size, fov_fraction)

        spacing = self._column_spacing_px(size, fov_fraction)
        sigma_px = acceptance_sigma_columns * spacing
        blurred = _gaussian_blur(image.astype(np.float64), sigma_px)
        return _bilinear_sample(blurred, centres)

    def _column_spacing_px(self, image_size: int, fov_fraction: float) -> float:
        """Centre-to-centre distance between neighbouring columns, in pixels."""
        radius = np.abs(self.centres).max()
        scale = (image_size * fov_fraction / 2.0) / radius
        return SQRT3 * scale

    def render_mosaic(
        self,
        values: np.ndarray,
        *,
        size: int = 512,
        fov_fraction: float = 0.65,
        background: float = 0.0,
    ) -> np.ndarray:
        """Render column values back into an image: "how the fly sees you".

        Each pixel takes the value of its nearest column, which produces the
        faceted look. Used for the shareable card, not for any computation.
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.shape[0] != self.n_columns:
            raise ValueError(f"expected {self.n_columns} values, got {values.shape[0]}")
        centres = self.pixel_centres(size, fov_fraction)
        yy, xx = np.mgrid[0:size, 0:size]
        pixels = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

        # Nearest column per pixel, in blocks to keep the distance matrix small.
        out = np.full(pixels.shape[0], background, dtype=np.float64)
        spacing = self._column_spacing_px(size, fov_fraction)
        block = 65536
        for start in range(0, pixels.shape[0], block):
            chunk = pixels[start : start + block]
            d2 = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1)
            within = d2[np.arange(chunk.shape[0]), nearest] <= spacing**2
            out[start : start + chunk.shape[0]] = np.where(within, values[nearest], background)
        return out.reshape(size, size)


def _gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur with edge clamping.

    Implemented here rather than pulled from scipy.ndimage so the vision path has
    one less heavy import and behaves identically everywhere.
    """
    if sigma_px <= 0:
        return image
    radius = max(1, int(round(3.0 * sigma_px)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_px) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
    rows = np.stack([padded[i : i + image.shape[0]] for i in range(2 * radius + 1)], axis=0)
    blurred = np.tensordot(kernel, rows, axes=(0, 0))
    padded = np.pad(blurred, ((0, 0), (radius, radius)), mode="edge")
    cols = np.stack([padded[:, i : i + image.shape[1]] for i in range(2 * radius + 1)], axis=0)
    return np.tensordot(kernel, cols, axes=(0, 0))


def _bilinear_sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear lookup at floating-point pixel coordinates, clamped at edges."""
    h, w = image.shape
    x = np.clip(points[:, 0], 0, w - 1)
    y = np.clip(points[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = x - x0
    fy = y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy
