"""Thin wrapper around flyvis, the frozen optic-lobe front end.

flyvis (Lappalainen et al., Nature 2024; MIT licensed) models 65 cell types of
the fly optic lobe on a 721-column hexagonal lattice. FlyMog uses it frozen: it
is the "eye", and only the readout downstream is trained.

The dependency is optional and heavy, so nothing here is imported at module
load. If flyvis is absent, :func:`flyvis_availability` says so and the caller
decides whether that is fatal. Pretrained weights are a separate problem: their
host was not reachable from the development environment, so they have to be
fetched by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from typing import Any


@dataclass
class FlyvisAvailability:
    """What of the flyvis stack is actually usable right now."""

    installed: bool
    version: str | None = None
    import_error: str | None = None
    pretrained_available: bool = False
    pretrained_detail: str = "not checked"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "import_error": self.import_error,
            "pretrained_available": self.pretrained_available,
            "pretrained_detail": self.pretrained_detail,
            "notes": self.notes,
        }


def flyvis_availability() -> FlyvisAvailability:
    """Check whether flyvis can be imported and whether weights are present."""
    if find_spec("flyvis") is None:
        return FlyvisAvailability(
            installed=False,
            import_error="flyvis is not installed",
            notes=[
                "Install with: uv pip install --python .venv/bin/python -e '.[flyvis]'",
                "flyvis requires Python >=3.9,<3.13.",
            ],
        )
    try:
        module = import_module("flyvis")
    except Exception as exc:  # pragma: no cover - depends on the local install
        return FlyvisAvailability(
            installed=False,
            import_error=f"{type(exc).__name__}: {exc}",
            notes=["flyvis is on the path but failed to import."],
        )

    version = getattr(module, "__version__", None)
    available, detail = _probe_pretrained(module)
    return FlyvisAvailability(
        installed=True,
        version=version,
        pretrained_available=available,
        pretrained_detail=detail,
        notes=[
            "flyvis code is MIT licensed. The license of the connectome data it "
            "bundles is tracked separately in docs/DATA_LICENSES.md.",
        ],
    )


def _probe_pretrained(module: Any) -> tuple[bool, str]:
    """Look for a local pretrained model directory without downloading anything."""
    for attr in ("results_dir", "RESULTS_DIR", "root_dir"):
        candidate = getattr(module, attr, None)
        if candidate is None:
            continue
        try:
            from pathlib import Path

            path = Path(str(candidate))
            if path.exists() and any(path.iterdir()):
                return True, f"found local model directory at {path}"
            return False, f"model directory {path} is missing or empty"
        except OSError as exc:
            return False, f"could not inspect {candidate}: {exc}"
    return False, "flyvis exposes no known results directory attribute"


# flyvis ships its connectome as a JSON file inside the package. Verified on
# flyvis 1.2.0: connectome/fib25-fib19_v2.2.json, with 65 node types, 8 input
# units (the photoreceptors R1-R8) and 34 output units.
_CONNECTOME_JSON_GLOB = "connectome/*.json"


def _connectome_json(module: Any) -> dict[str, Any]:
    """Load the connectome definition flyvis ships with."""
    from pathlib import Path

    root = Path(module.__file__).parent
    candidates = sorted(root.glob(_CONNECTOME_JSON_GLOB))
    if not candidates:
        raise RuntimeError(
            f"no connectome JSON found under {root / 'connectome'}. "
            "The installed flyvis version may have moved it; update "
            "flymog.vision.flyvis_frontend."
        )
    # Newest version last when names sort lexically; take the last deliberately.
    with candidates[-1].open(encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("nodes", "input_units", "output_units"):
        if key not in data:
            raise RuntimeError(f"{candidates[-1]} has no '{key}' key; the format changed")
    return data


def cell_types(module: Any | None = None) -> list[str]:
    """All optic-lobe cell types flyvis models.

    Read from the package's own connectome file rather than guessed, because a
    wrong list would make the bridge coverage number meaningless.
    """
    module = module or import_module("flyvis")
    return [str(node["name"]) for node in _connectome_json(module)["nodes"]]


def input_cell_types(module: Any | None = None) -> list[str]:
    """Types that receive the image directly. These are the photoreceptors."""
    module = module or import_module("flyvis")
    return [str(name) for name in _connectome_json(module)["input_units"]]


def output_cell_types(module: Any | None = None) -> list[str]:
    """Types flyvis treats as leaving the optic lobe.

    These are the ones that have to be joined to FlyWire for the central brain
    to receive anything, so the bridge is measured against this list.
    """
    module = module or import_module("flyvis")
    return [str(name) for name in _connectome_json(module)["output_units"]]
