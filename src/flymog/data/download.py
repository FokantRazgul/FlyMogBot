"""Acquiring the connectome: real files if present, surrogate otherwise.

This environment's egress policy blocks zenodo.org, codex.flywire.ai,
huggingface.co and drive.google.com, so automatic download is not possible here.
Rather than pretend otherwise, :func:`ensure_connectome` fails with explicit
manual instructions naming the blocked host.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from flymog.config import ConnectomeConfig, data_dir
from flymog.data.connectome import (
    EDGE_COLUMN_ALIASES,
    NEURON_COLUMN_ALIASES,
    Connectome,
    ConnectomeDataMissing,
)


@dataclass(frozen=True)
class DataSource:
    """A file the operator may need to fetch by hand."""

    key: str
    filename: str
    url: str
    description: str
    license_note: str


# TODO(verify): these URLs are recorded from the project brief and from search
# results, and could NOT be opened from this environment because the hosts are
# blocked. Confirm each one resolves to the expected file before relying on it.
KNOWN_SOURCES: tuple[DataSource, ...] = (
    DataSource(
        key="flywire_connectivity",
        filename="flywire_connections.csv",
        url="https://zenodo.org/records/10676866",
        description=(
            "FlyWire FAFB adult brain connectivity (neuron pairs with synapse counts "
            "and predicted neurotransmitters). TODO(verify) the exact file and release."
        ),
        license_note="TODO(verify): sources disagree between CC BY 4.0 and CC BY-NC-SA 4.0.",
    ),
    DataSource(
        key="flywire_annotations",
        filename="flywire_neurons.tsv",
        url="https://github.com/flyconnectome/flywire_annotations",
        description=(
            "Per-neuron annotations: cell type, super class, side, position. "
            "Clonable over git even where web access is denied. The file wanted "
            "is supplemental_files/Supplemental_file1_neuron_annotations.tsv."
        ),
        license_note=(
            "No LICENSE file in the repository. Its README asks to cite Berg et al. "
            "(2025), Schlegel et al. (2024), Matsliah et al. (2024) and Dorkenwald "
            "et al. (2024). TODO(verify) the license itself."
        ),
    ),
    DataSource(
        key="flyvis_pretrained",
        filename="flyvis_pretrained/",
        url="https://github.com/TuragaLab/flyvis",
        description=(
            "Pretrained flyvis optic-lobe models. flyvis depends on "
            "google-api-python-client, which suggests the weights are hosted on "
            "Google Drive. TODO(verify) the actual download path."
        ),
        license_note="flyvis code is MIT. TODO(verify) the license of its bundled connectome data.",
    ),
)


def host_is_reachable(url: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Can this URL be opened through the normal, proxied path?

    Deliberately uses urllib, which honours the HTTPS_PROXY environment, rather
    than a raw socket. A raw socket would bypass a configured egress proxy and
    report a host as reachable that policy actually denies, which is both wrong
    and an invitation to route around the policy. A denial is reported, never
    retried and never worked around.
    """
    parsed = urlparse(url)
    if not parsed.hostname:
        return False, "could not parse a hostname out of the URL"

    request = Request(url, method="HEAD", headers={"User-Agent": "flymog/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return True, f"HTTP {response.status} through the configured proxy"
    except HTTPError as exc:
        # The host answered. 403 or 407 from a proxy is a policy denial; any
        # other status still proves the host is reachable.
        if exc.code in {403, 407}:
            return False, f"HTTP {exc.code}: denied by the egress policy, not by the host"
        return True, f"HTTP {exc.code} through the configured proxy"
    except URLError as exc:
        return False, f"{type(exc.reason).__name__ if exc.reason else 'URLError'}: {exc.reason}"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def manual_instructions(missing: list[DataSource], target_dir: Path) -> str:
    lines = [
        "Required data files are missing and cannot be downloaded from this environment.",
        "",
        "The egress policy of this session blocks the hosts that serve them.",
        "Retrying will not help; the files have to be placed by hand.",
        "",
        f"Put them under: {target_dir}",
        "",
    ]
    for src in missing:
        lines += [
            f"  [{src.key}]",
            f"    file:    {target_dir / src.filename}",
            f"    from:    {src.url}",
            f"    what:    {src.description}",
            f"    license: {src.license_note}",
            "",
        ]
    lines.append("Then re-run the command. Nothing is cached or guessed in the meantime.")
    return "\n".join(lines)


def _resolve_columns(
    header: list[str], aliases: dict[str, tuple[str, ...]], required: set[str]
) -> dict[str, int]:
    lowered = {name.strip().lower(): idx for idx, name in enumerate(header)}
    resolved: dict[str, int] = {}
    for field_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lowered:
                resolved[field_name] = lowered[candidate]
                break
    missing = required - set(resolved)
    if missing:
        raise ConnectomeDataMissing(
            f"columns {sorted(missing)} not found in header {header}. "
            f"Accepted spellings: " + ", ".join(f"{k}={aliases[k]}" for k in sorted(missing))
        )
    return resolved


def _delimiter_for(path: Path) -> str:
    """Pick the field separator from the file extension.

    FlyWire's published annotations are tab separated while the connectivity
    dumps are comma separated, so both have to work without a flag.
    """
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def load_neuron_table(path: Path, cfg: ConnectomeConfig) -> Connectome:
    """Load only the neuron annotations, with no edges.

    The input-cell ladder and the bridge probe ask questions about annotations
    alone, and the annotation file is two orders of magnitude smaller than the
    connectivity dump. Being able to answer them without the edges means those
    probes can run before the large download has happened.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=_delimiter_for(path))
        header = next(reader)
        cols = _resolve_columns(header, NEURON_COLUMN_ALIASES, {"id"})
        has_pos = {"x", "y", "z"} <= set(cols)

        ids: list[int] = []
        types: list[str] = []
        classes: list[str] = []
        transmitters: list[str] = []
        positions: list[tuple[float, float, float]] = []
        skipped_without_id = 0

        for row in reader:
            raw_id = row[cols["id"]].strip() if cols["id"] < len(row) else ""
            if not raw_id:
                skipped_without_id += 1
                continue
            ids.append(int(raw_id))
            types.append(row[cols["cell_type"]] if "cell_type" in cols else "")
            classes.append(row[cols["super_class"]] if "super_class" in cols else "")
            transmitters.append(row[cols["transmitter"]] if "transmitter" in cols else "")
            if has_pos:
                positions.append(
                    (
                        _to_float(row[cols["x"]]),
                        _to_float(row[cols["y"]]),
                        _to_float(row[cols["z"]]),
                    )
                )

    n = len(ids)
    empty_int = np.asarray([], dtype=np.int64)
    return Connectome(
        neuron_ids=np.asarray(ids, dtype=np.int64),
        cell_types=np.asarray(types, dtype=object),
        super_classes=np.asarray(classes, dtype=object),
        transmitters=np.asarray(transmitters, dtype=object),
        positions=np.asarray(positions, dtype=np.float64) if has_pos else None,
        pre_idx=empty_int,
        post_idx=empty_int,
        syn_count=empty_int,
        sign=np.asarray([], dtype=np.float64),
        source=str(path),
        version="TODO(verify): record the annotation release these rows came from",
        is_surrogate=False,
        meta={
            "annotations_only": True,
            "note": "No edges loaded. Anything needing connectivity must load the full dump.",
            "n_neurons": n,
            "skipped_rows_without_id": skipped_without_id,
        },
    )


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_connectome_from_csv(
    neurons_csv: Path, connections_csv: Path, cfg: ConnectomeConfig
) -> Connectome:
    """Load the real connectome from two CSV files into the internal format.

    Edges below the synapse threshold are dropped, and edges whose sign cannot be
    determined are dropped too rather than being assigned a guessed sign.
    """
    with neurons_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=_delimiter_for(neurons_csv))
        header = next(reader)
        cols = _resolve_columns(header, NEURON_COLUMN_ALIASES, {"id"})
        ids: list[int] = []
        types: list[str] = []
        classes: list[str] = []
        transmitters: list[str] = []
        positions: list[tuple[float, float, float]] = []
        has_pos = {"x", "y", "z"} <= set(cols)
        for row in reader:
            ids.append(int(row[cols["id"]]))
            types.append(row[cols["cell_type"]] if "cell_type" in cols else "")
            classes.append(row[cols["super_class"]] if "super_class" in cols else "")
            transmitters.append(row[cols["transmitter"]] if "transmitter" in cols else "")
            if has_pos:
                positions.append(
                    (
                        float(row[cols["x"]] or "nan"),
                        float(row[cols["y"]] or "nan"),
                        float(row[cols["z"]] or "nan"),
                    )
                )

    neuron_ids = np.asarray(ids, dtype=np.int64)
    index_of = {int(v): i for i, v in enumerate(neuron_ids)}
    nt_by_index = np.asarray(transmitters, dtype=object)

    pre_list: list[int] = []
    post_list: list[int] = []
    count_list: list[int] = []
    sign_list: list[float] = []
    dropped_threshold = dropped_sign = dropped_unknown_neuron = 0

    with connections_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=_delimiter_for(connections_csv))
        header = next(reader)
        cols = _resolve_columns(header, EDGE_COLUMN_ALIASES, {"pre", "post", "syn_count"})
        for row in reader:
            count = int(float(row[cols["syn_count"]]))
            if count < cfg.min_synapse_count:
                dropped_threshold += 1
                continue
            pre_id = int(row[cols["pre"]])
            post_id = int(row[cols["post"]])
            pre_i = index_of.get(pre_id)
            post_i = index_of.get(post_id)
            if pre_i is None or post_i is None:
                dropped_unknown_neuron += 1
                continue
            nt = row[cols["transmitter"]] if "transmitter" in cols else str(nt_by_index[pre_i])
            sign = cfg.sign_for(nt)
            if sign is None:
                dropped_sign += 1
                continue
            pre_list.append(pre_i)
            post_list.append(post_i)
            count_list.append(count)
            sign_list.append(sign)

    return Connectome(
        neuron_ids=neuron_ids,
        cell_types=np.asarray(types, dtype=object),
        super_classes=np.asarray(classes, dtype=object),
        transmitters=nt_by_index,
        positions=np.asarray(positions, dtype=np.float64) if has_pos else None,
        pre_idx=np.asarray(pre_list, dtype=np.int64),
        post_idx=np.asarray(post_list, dtype=np.int64),
        syn_count=np.asarray(count_list, dtype=np.int64),
        sign=np.asarray(sign_list, dtype=np.float64),
        source=str(connections_csv.parent),
        version="TODO(verify): record the release identifier of these files",
        is_surrogate=False,
        meta={
            "dropped_below_synapse_threshold": dropped_threshold,
            "dropped_unknown_sign": dropped_sign,
            "dropped_unknown_neuron": dropped_unknown_neuron,
            "min_synapse_count": cfg.min_synapse_count,
        },
    )


def make_surrogate_connectome(
    n_neurons: int = 140_000,
    n_edges: int = 2_700_000,
    *,
    seed: int = 0,
    cfg: ConnectomeConfig | None = None,
) -> Connectome:
    """Build a structurally plausible but scientifically meaningless connectome.

    Used only to exercise and benchmark the pipeline where the real data cannot
    be downloaded. Degrees follow a heavy-tailed distribution so that sparse
    kernels see a realistic access pattern, but nothing here reflects biology.
    """
    rng = np.random.default_rng(seed)

    # Heavy-tailed out-degree preference, roughly matching how a few hub neurons
    # dominate connectomic edge counts.
    out_pref = rng.pareto(1.6, size=n_neurons) + 1.0
    out_pref /= out_pref.sum()
    in_pref = rng.pareto(1.6, size=n_neurons) + 1.0
    in_pref /= in_pref.sum()

    pre_idx = rng.choice(n_neurons, size=n_edges, p=out_pref).astype(np.int64)
    post_idx = rng.choice(n_neurons, size=n_edges, p=in_pref).astype(np.int64)
    self_loops = pre_idx == post_idx
    post_idx[self_loops] = (post_idx[self_loops] + 1) % n_neurons

    syn_count = (rng.pareto(2.0, size=n_edges) * 4.0 + 5.0).astype(np.int64)

    nt_choices = np.array(["acetylcholine", "gaba", "glutamate"], dtype=object)
    nt_per_neuron = rng.choice(nt_choices, size=n_neurons, p=[0.60, 0.25, 0.15])
    cfg = cfg or ConnectomeConfig(
        min_synapse_count=5,
        excitatory_transmitters=["acetylcholine", "ach"],
        inhibitory_transmitters=["gaba", "glutamate", "glu"],
        unknown_transmitter_policy="drop",
    )
    sign_lookup = {str(nt): cfg.sign_for(str(nt)) or 0.0 for nt in nt_choices}
    sign = np.asarray([sign_lookup[str(nt)] for nt in nt_per_neuron], dtype=np.float64)[pre_idx]

    # A fake cell-type vocabulary shaped like the real one: a few very common
    # columnar types plus a long tail. Names are deliberately prefixed so they
    # can never be mistaken for real FlyWire annotations.
    n_types = 800
    type_names = np.asarray([f"SURROGATE_T{i:04d}" for i in range(n_types)], dtype=object)
    type_pref = rng.pareto(1.2, size=n_types) + 1.0
    type_pref /= type_pref.sum()
    cell_types = type_names[rng.choice(n_types, size=n_neurons, p=type_pref)]
    super_classes = np.asarray(
        rng.choice(
            np.array(["SURROGATE_optic", "SURROGATE_central", "SURROGATE_output"], dtype=object),
            size=n_neurons,
            p=[0.55, 0.35, 0.10],
        ),
        dtype=object,
    )
    positions = rng.normal(0.0, 1.0, size=(n_neurons, 3))

    return Connectome(
        neuron_ids=np.arange(n_neurons, dtype=np.int64),
        cell_types=cell_types,
        super_classes=super_classes,
        transmitters=np.asarray(nt_per_neuron, dtype=object),
        positions=positions,
        pre_idx=pre_idx,
        post_idx=post_idx,
        syn_count=syn_count,
        sign=sign,
        source="synthetic surrogate (flymog.data.download.make_surrogate_connectome)",
        version=f"surrogate-seed{seed}",
        is_surrogate=True,
        meta={
            "warning": (
                "SURROGATE DATA. Structure only, no biology. Any number derived from "
                "this connectome is a performance measurement, never a scientific result."
            ),
            "seed": seed,
        },
    )


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    """First of ``names`` that exists in ``directory``, or None."""
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def find_annotations(target_dir: Path | None = None) -> Path | None:
    """Locate the neuron annotation table, whichever spelling it was saved under."""
    target_dir = target_dir or (data_dir() / "connectome")
    return _first_existing(target_dir, ("flywire_neurons.tsv", "flywire_neurons.csv"))


def ensure_connectome(
    cfg: ConnectomeConfig,
    *,
    allow_surrogate: bool = False,
    target_dir: Path | None = None,
) -> Connectome:
    """Load the real connectome, or explain exactly what is missing.

    With ``allow_surrogate`` the surrogate is returned instead of raising, which
    is how benchmarks run in an environment that cannot reach the data.
    """
    target_dir = target_dir or (data_dir() / "connectome")
    neurons = _first_existing(target_dir, ("flywire_neurons.tsv", "flywire_neurons.csv"))
    connections = _first_existing(
        target_dir, ("flywire_connections.csv", "flywire_connections.tsv")
    )

    if neurons is not None and connections is not None:
        return load_connectome_from_csv(neurons, connections, cfg)

    if allow_surrogate:
        return make_surrogate_connectome(cfg=cfg)

    missing = [s for s in KNOWN_SOURCES if s.key.startswith("flywire")]
    raise ConnectomeDataMissing(manual_instructions(missing, target_dir))


def git_clone_available(url: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Can this repository be read over the git protocol?

    Web access and git access are governed separately: an environment can deny
    HTTPS browsing of github.com while still allowing ``git``. That distinction
    decides what the operator has to do, so it is checked rather than assumed.
    """
    if "github.com" not in urlparse(url).netloc:
        return False, "not a GitHub URL"
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", url],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return True, "git ls-remote succeeded, the repository can be cloned"
    return False, f"git ls-remote exited {result.returncode}: {result.stderr.strip()[:160]}"


def reachability_report() -> list[dict]:
    """Check every known source and report what this environment can actually get."""
    report = []
    for src in KNOWN_SOURCES:
        web_ok, web_detail = host_is_reachable(src.url)
        git_ok, git_detail = git_clone_available(src.url)
        report.append(
            {
                "key": src.key,
                "url": src.url,
                "reachable": bool(web_ok or git_ok),
                "web_reachable": web_ok,
                "git_reachable": git_ok,
                "detail": (git_detail if git_ok else f"web: {web_detail}; git: {git_detail}"),
                "license_note": src.license_note,
            }
        )
    return report
