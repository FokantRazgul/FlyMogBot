"""FlyMog command line interface.

One cross-platform entry point, ``flymog <command>``. No Makefile, because the
operator's machine may be Windows.

M0 commands write JSON to ``artifacts/m0/`` so that ``flymog m0-report`` can
assemble docs/M0_REPORT.md out of results gathered on a different machine than
the one the code was written on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from flymog.config import data_dir, load_sim_config, repo_root

app = typer.Typer(
    name="flymog",
    help="Rate a face through the eyes of a fruit fly brain. Entertainment only.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def artifacts_dir() -> Path:
    path = repo_root() / "artifacts" / "m0"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _emit(payload: dict[str, Any], name: str, output: Path | None) -> Path:
    """Write a probe result to disk and echo a short summary."""
    target = output or (artifacts_dir() / f"{name}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]wrote[/green] {target}")
    return target


def _warn_if_surrogate(payload: dict[str, Any]) -> None:
    warning = payload.get("surrogate_warning")
    if warning:
        console.print(f"[yellow]SURROGATE DATA:[/yellow] {warning}")


def _load_connectome(
    surrogate: bool,
    surrogate_neurons: int,
    surrogate_edges: int,
    annotations: Path | None = None,
):
    """Load a connectome for a probe.

    ``annotations`` loads the neuron table alone, with no edges. The input-cell
    ladder and the bridge probe only ask about annotations, and that file is far
    smaller than the connectivity dump, so they can run before it is downloaded.
    """
    from flymog.data.connectome import ConnectomeDataMissing
    from flymog.data.download import (
        ensure_connectome,
        load_neuron_table,
        make_surrogate_connectome,
    )

    cfg = load_sim_config()
    if annotations is not None:
        console.print(f"Loading neuron annotations only from {annotations}")
        return load_neuron_table(annotations, cfg.connectome)
    if surrogate:
        console.print(
            f"[yellow]Using the synthetic surrogate connectome "
            f"({surrogate_neurons} neurons, {surrogate_edges} edges).[/yellow]"
        )
        return make_surrogate_connectome(surrogate_neurons, surrogate_edges, cfg=cfg.connectome)
    try:
        return ensure_connectome(cfg.connectome, allow_surrogate=False)
    except ConnectomeDataMissing as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


@app.command("probe-env")
def probe_env(
    output: Annotated[Path | None, typer.Option(help="Where to write the JSON report")] = None,
) -> None:
    """Report hardware, accelerators and the backend that will be used."""
    from flymog.probes.env import probe_environment

    payload = probe_environment()
    table = Table(title="Environment", show_header=False)
    table.add_row("platform", f"{payload['platform']['system']} {payload['platform']['machine']}")
    table.add_row("python", payload["platform"]["python"])
    table.add_row("cpu cores", str(payload["cpu"]["logical_cores"]))
    table.add_row("memory", f"{payload['memory']['total_gb']} GB")
    table.add_row("torch", str(payload["torch"].get("version", "not installed")))
    table.add_row("selected device", str(payload["selected_device"]))
    table.add_row("auto backend", str(payload["auto_backend"]))
    for gpu in payload["nvidia"].get("gpus", []):
        table.add_row("nvidia", gpu)
    console.print(table)
    _emit(payload, "probe_env", output)


@app.command("fetch-data")
def fetch_data(
    output: Annotated[Path | None, typer.Option(help="Where to write the JSON report")] = None,
) -> None:
    """Check which large data files are present and which hosts are reachable."""
    from flymog.data.download import KNOWN_SOURCES, reachability_report

    target = data_dir() / "connectome"
    report = reachability_report()
    present = {src.key: (target / src.filename).exists() for src in KNOWN_SOURCES}

    table = Table(title="Data sources")
    table.add_column("key")
    table.add_column("present locally")
    table.add_column("host reachable")
    table.add_column("detail", overflow="fold")
    for src, reach in zip(KNOWN_SOURCES, report, strict=True):
        table.add_row(
            src.key,
            "yes" if present[src.key] else "no",
            "yes" if reach["reachable"] else "no",
            reach["detail"][:80],
        )
    console.print(table)

    missing = [src for src in KNOWN_SOURCES if not present[src.key]]
    if missing:
        from flymog.data.download import manual_instructions

        console.print()
        # markup=False: the instructions contain [source_key] markers that rich
        # would otherwise swallow as style tags.
        console.print(manual_instructions(missing, target), markup=False)

    payload = {
        "target_dir": str(target),
        "present": present,
        "reachability": report,
        "all_present": not missing,
    }
    _emit(payload, "fetch_data", output)


@app.command("probe-inputs")
def probe_inputs(
    surrogate: Annotated[bool, typer.Option(help="Run against the synthetic surrogate")] = False,
    annotations: Annotated[
        Path | None,
        typer.Option(help="Neuron annotation table to use instead of the full connectome"),
    ] = None,
    surrogate_neurons: int = 140_000,
    surrogate_edges: int = 2_700_000,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Check which input-cell rung of the ladder the connectome supports."""
    from flymog.data.download import find_annotations
    from flymog.probes.inputs import probe_input_cells

    if annotations is None and not surrogate:
        annotations = find_annotations()
    connectome = _load_connectome(surrogate, surrogate_neurons, surrogate_edges, annotations)
    payload = probe_input_cells(connectome)
    _warn_if_surrogate(payload)

    table = Table(title="Input-cell ladder")
    table.add_column("rung")
    table.add_column("cells", justify="right")
    table.add_column("positions")
    table.add_column("verdict", overflow="fold")
    for rung in payload["rungs"]:
        table.add_row(
            rung["name"],
            str(rung["total_cells"]),
            "yes" if rung["has_positions"] else "no",
            rung["verdict"],
        )
    console.print(table)
    console.print(f"[bold]{payload['recommendation']}[/bold]")
    _emit(payload, "probe_inputs", output)


@app.command("probe-bridge")
def probe_bridge_cmd(
    surrogate: Annotated[bool, typer.Option(help="Run against the synthetic surrogate")] = False,
    flyvis_types_file: Annotated[
        Path | None,
        typer.Option(help="Text file with one flyvis cell type per line"),
    ] = None,
    annotations: Annotated[
        Path | None,
        typer.Option(help="Neuron annotation table to use instead of the full connectome"),
    ] = None,
    all_types: Annotated[
        bool,
        typer.Option(help="Match all 65 flyvis types, not just the ones leaving the optic lobe"),
    ] = False,
    surrogate_neurons: int = 140_000,
    surrogate_edges: int = 2_700_000,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure how well flyvis cell types map onto FlyWire annotations."""
    from flymog.data.download import find_annotations
    from flymog.probes.bridge import probe_bridge

    if annotations is None and not surrogate:
        annotations = find_annotations()
    connectome = _load_connectome(surrogate, surrogate_neurons, surrogate_edges, annotations)
    types = None
    if flyvis_types_file is not None:
        types = [
            line.strip()
            for line in flyvis_types_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        console.print(f"read {len(types)} flyvis cell types from {flyvis_types_file}")

    payload = probe_bridge(connectome, types, only_output_types=not all_types)
    if not payload["ran"]:
        console.print(f"[red]bridge probe did not run:[/red] {payload['reason']}")
        console.print(payload.get("next_step", ""))
        _emit(payload, "probe_bridge", output)
        raise typer.Exit(code=3)

    _warn_if_surrogate(payload)
    stats = payload["stats"]
    table = Table(title="flyvis -> FlyWire bridge", show_header=False)
    table.add_row("flyvis cell types", str(stats["n_flyvis_types"]))
    table.add_row("FlyWire cell types", str(stats["n_flywire_types"]))
    table.add_row("matched", str(stats["n_matched"]))
    table.add_row("coverage", f"{stats['coverage']:.1%}")
    table.add_row("matches by kind", json.dumps(stats["matches_by_kind"]))
    table.add_row("FlyWire neurons reached", str(stats["n_flywire_neurons_reached"]))
    table.add_row("in projection classes", str(payload["n_matched_neurons_in_projection_classes"]))
    table.add_row("verdict", payload["verdict"])
    console.print(table)
    console.print(f"[bold]{payload['recommendation']}[/bold]")
    console.print(f"Wiring: [bold]{payload['wiring']}[/bold]. {payload['wiring_note']}")
    _emit(payload, "probe_bridge", output)


@app.command("bench-sim")
def bench_sim(
    surrogate: Annotated[bool, typer.Option(help="Run against the synthetic surrogate")] = False,
    batch: Annotated[int, typer.Option(help="Batch size to benchmark")] = 32,
    measured_steps: int = 20,
    surrogate_neurons: int = 140_000,
    surrogate_edges: int = 2_700_000,
    check_pruning: Annotated[bool, typer.Option(help="Also run the pruning check")] = False,
    k_hops: int = 5,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure simulation throughput, and optionally the pruning equivalence."""
    import numpy as np

    from flymog.probes.bench import benchmark_backends, pruning_equivalence

    cfg = load_sim_config()
    connectome = _load_connectome(surrogate, surrogate_neurons, surrogate_edges)
    payload = benchmark_backends(
        connectome, cfg.lif, batch_sizes=(batch,), measured_steps=measured_steps
    )
    _warn_if_surrogate(payload)

    table = Table(title=f"Throughput on {payload['device']} (batch {batch})")
    for column in ("backend", "dt ms", "window ms", "ms/step", "s/window", "img/s", "target"):
        table.add_column(column, justify="right")
    for row in payload["rows"]:
        if "error" in row:
            continue
        table.add_row(
            row["backend"],
            f"{row['dt_ms']}",
            f"{row['window_ms']:.0f}",
            f"{row['ms_per_step']:.2f}",
            f"{row['seconds_per_window']:.2f}",
            f"{row['images_per_second']:.3f}",
            "yes" if row["meets_target"] else "no",
        )
    console.print(table)
    console.print(f"[bold]{payload['verdict']}[/bold]")

    if check_pruning:
        seeds = np.arange(min(2000, connectome.n_neurons // 10))
        payload["pruning"] = pruning_equivalence(connectome, cfg.lif, seeds, k_hops=k_hops, batch=4)
        pruning = payload["pruning"]
        console.print(
            f"pruning k={k_hops}: kept {pruning['pruned_n_neurons']}/"
            f"{pruning['full_n_neurons']} neurons, rate correlation "
            f"{pruning['rate_correlation']}, equivalent={pruning['equivalent']}"
        )
    _emit(payload, "bench_sim", output)


@app.command("sweep-regime")
def sweep_regime_cmd(
    surrogate: Annotated[bool, typer.Option(help="Run against the synthetic surrogate")] = False,
    batch: int = 16,
    surrogate_neurons: int = 140_000,
    surrogate_edges: int = 2_700_000,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Find an operating point where the network is neither silent nor saturated."""
    from flymog.probes.regime import sweep_regime

    cfg = load_sim_config()
    connectome = _load_connectome(surrogate, surrogate_neurons, surrogate_edges)
    payload = sweep_regime(connectome, cfg.lif, batch=batch)
    _warn_if_surrogate(payload)
    for key in ("resolution_warning", "saturation_warning"):
        if payload.get(key):
            console.print(f"[yellow]{payload[key]}[/yellow]")

    table = Table(title="Regime sweep")
    for column in ("gain", "weight scale", "responsive", "mean Hz", "max Hz", "PR", "in band"):
        table.add_column(column, justify="right")
    for row in payload["rows"]:
        table.add_row(
            f"{row['input_gain']}",
            f"{row['global_weight_scale']:g}",
            f"{row['fraction_responsive']:.2f}",
            f"{row['mean_responsive_rate_hz']:.1f}",
            f"{row['max_rate_hz']:.1f}",
            "n/a" if row["participation_ratio"] is None else f"{row['participation_ratio']:.2f}",
            "yes" if row["in_target_band"] else "no",
        )
    console.print(table)
    console.print(f"[bold]{payload['verdict']}[/bold]")
    if payload["chosen"]:
        console.print(
            "Write these into configs/sim.yaml under regime: "
            f"input_gain={payload['chosen']['input_gain']}, "
            f"global_weight_scale={payload['chosen']['global_weight_scale']}"
        )
    _emit(payload, "sweep_regime", output)


@app.command("m0-report")
def m0_report(
    artifacts: Annotated[Path | None, typer.Option(help="Directory holding probe JSON")] = None,
    output: Annotated[Path | None, typer.Option(help="Where to write the report")] = None,
) -> None:
    """Assemble docs/M0_REPORT.md from whatever probe results are present."""
    from flymog.probes.report import build_m0_report

    source = artifacts or artifacts_dir()
    target = output or (repo_root() / "docs" / "M0_REPORT.md")
    text, found, missing = build_m0_report(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    console.print(f"[green]wrote[/green] {target}")
    console.print(f"filled from: {', '.join(found) if found else 'nothing'}")
    if missing:
        console.print(f"[yellow]still missing:[/yellow] {', '.join(missing)}")


@app.command("hex-demo")
def hex_demo(
    image: Annotated[Path, typer.Argument(help="Image to run through the fly eye")],
    output: Annotated[Path, typer.Option(help="Where to write the mosaic PNG")] = Path(
        "artifacts/hex_demo.png"
    ),
) -> None:
    """Render "how the fly sees you" for one image. Useful for eyeballing framing."""
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        console.print("[red]Pillow is required for hex-demo. Install it and retry.[/red]")
        raise typer.Exit(code=2) from None

    from flymog.vision.encoder import encode
    from flymog.vision.hexgrid import HexLattice

    cfg = load_sim_config().vision
    array = np.asarray(Image.open(image).convert("RGB"))
    lattice = HexLattice.build(cfg.hex_extent)
    encoded = encode(array, cfg, lattice=lattice)
    mosaic = lattice.render_mosaic(encoded.hex_values, size=512, fov_fraction=cfg.fov_fraction)
    scaled = (np.clip(mosaic, 0.0, 1.0) * 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scaled).save(output)
    console.print(f"[green]wrote[/green] {output} ({lattice.n_columns} columns)")


def main() -> None:  # pragma: no cover - console script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
