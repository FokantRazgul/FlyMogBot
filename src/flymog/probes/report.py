"""Assemble docs/M0_REPORT.md from probe outputs.

The report is deliberately assembled from JSON rather than written by hand, for
one reason: the development machine has no accelerator and cannot reach the data
hosts, so the numbers have to come from somewhere else. Any section whose input
is missing says so and names the command that fills it, instead of being quietly
omitted or filled with a plausible guess.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Probe name -> (filename, the command that produces it)
PROBES: dict[str, tuple[str, str]] = {
    "probe_env": ("probe_env.json", "flymog probe-env"),
    "fetch_data": ("fetch_data.json", "flymog fetch-data"),
    "probe_inputs": ("probe_inputs.json", "flymog probe-inputs"),
    "probe_bridge": ("probe_bridge.json", "flymog probe-bridge"),
    "bench_sim": ("bench_sim.json", "flymog bench-sim --check-pruning"),
    "sweep_regime": ("sweep_regime.json", "flymog sweep-regime"),
}

_MISSING = "_Не заполнено._ Запусти `{command}` на машине с целевым железом."


def _load(source: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    loaded: dict[str, Any] = {}
    found: list[str] = []
    missing: list[str] = []
    for key, (filename, _command) in PROBES.items():
        path = source / filename
        if path.exists():
            try:
                loaded[key] = json.loads(path.read_text(encoding="utf-8"))
                found.append(key)
                continue
            except json.JSONDecodeError:
                missing.append(key)
                continue
        missing.append(key)
    return loaded, found, missing


def _surrogate_banner(data: dict[str, Any]) -> str | None:
    surrogate_probes = [
        key
        for key, payload in data.items()
        if isinstance(payload, dict) and payload.get("is_surrogate")
    ]
    if not surrogate_probes:
        return None
    return (
        "> **Внимание: часть чисел получена на синтетическом суррогате коннектома, "
        "а не на настоящих данных.** Суррогат повторяет масштаб и распределение "
        "степеней, поэтому замеры скорости показательны, но любые утверждения о "
        "биологии по ним делать нельзя. Затронутые разделы: "
        + ", ".join(f"`{p}`" for p in sorted(surrogate_probes))
        + "."
    )


def _hardware_section(data: dict[str, Any]) -> str:
    env = data.get("probe_env")
    if not env:
        return _MISSING.format(command=PROBES["probe_env"][1])
    platform = env["platform"]
    torch_info = env.get("torch", {})
    lines = [
        "| Параметр | Значение |",
        "|---|---|",
        f"| ОС | {platform['system']} {platform['release']} ({platform['machine']}) |",
        f"| Python | {platform['python']} |",
        f"| Ядра CPU | {env['cpu']['logical_cores']} логических, "
        f"{env['cpu']['physical_cores']} физических |",
        f"| RAM | {env['memory']['total_gb']} ГБ |",
        f"| torch | {torch_info.get('version', 'не установлен')} |",
        f"| CUDA | {'да' if torch_info.get('cuda_available') else 'нет'} |",
        f"| MPS | {'да' if torch_info.get('mps_available') else 'нет'} |",
        f"| Выбранное устройство | {env.get('selected_device')} |",
        f"| Backend по умолчанию | {env.get('auto_backend')} |",
    ]
    for gpu in env.get("nvidia", {}).get("gpus", []) or []:
        lines.append(f"| nvidia-smi | {gpu} |")
    for device in torch_info.get("cuda_devices", []) or []:
        lines.append(
            f"| CUDA устройство {device['index']} | {device['name']}, "
            f"{device['total_memory_gb']} ГБ, capability {device['capability']} |"
        )
    return "\n".join(lines)


def _data_section(data: dict[str, Any]) -> str:
    fetch = data.get("fetch_data")
    if not fetch:
        return _MISSING.format(command=PROBES["fetch_data"][1])
    lines = [
        f"Целевой каталог: `{fetch['target_dir']}`",
        "",
        "| Источник | Файл на месте | Хост доступен | Детали |",
        "|---|---|---|---|",
    ]
    present = fetch["present"]
    for entry in fetch["reachability"]:
        key = entry["key"]
        lines.append(
            f"| `{key}` | {'да' if present.get(key) else 'нет'} | "
            f"{'да' if entry['reachable'] else 'нет'} | {entry['detail']} |"
        )
    lines += ["", "Лицензии и атрибуция: см. `docs/DATA_LICENSES.md`."]
    return "\n".join(lines)


def _inputs_section(data: dict[str, Any]) -> str:
    probe = data.get("probe_inputs")
    if not probe:
        return _MISSING.format(command=PROBES["probe_inputs"][1])
    lines = [
        "| Ступень | Клеток | Позиции | Оценка колонок | Вердикт |",
        "|---|---:|---|---:|---|",
    ]
    for rung in probe["rungs"]:
        lines.append(
            f"| {rung['name']} | {rung['total_cells']} | "
            f"{'да' if rung['has_positions'] else 'нет'} | "
            f"{rung['estimated_columns'] if rung['estimated_columns'] is not None else '—'} | "
            f"{rung['verdict']} |"
        )
    lines += ["", f"**{probe['recommendation']}**"]
    return "\n".join(lines)


def _bridge_section(data: dict[str, Any]) -> str:
    probe = data.get("probe_bridge")
    if not probe:
        return _MISSING.format(command=PROBES["probe_bridge"][1])
    if not probe.get("ran"):
        return (
            f"Probe не отработал: {probe.get('reason')}\n\n"
            f"Следующий шаг: {probe.get('next_step', '—')}"
        )
    stats = probe["stats"]
    lines = [
        "| Метрика | Значение |",
        "|---|---|",
        f"| Типов клеток в flyvis | {stats['n_flyvis_types']} |",
        f"| Типов клеток в FlyWire | {stats['n_flywire_types']} |",
        f"| Сошлось | {stats['n_matched']} |",
        f"| **Покрытие** | **{stats['coverage']:.1%}** |",
        f"| Порог go/no-go | {probe['go_threshold']:.1%} |",
        f"| Нейронов FlyWire достигнуто | {stats['n_flywire_neurons_reached']} |",
        f"| По типу совпадения | {json.dumps(stats['matches_by_kind'], ensure_ascii=False)} |",
        f"| **Вердикт** | **{probe['verdict']}** |",
        "",
        f"**{probe['recommendation']}**",
    ]
    unmatched = stats.get("unmatched_flyvis_types") or []
    if unmatched:
        shown = ", ".join(f"`{t}`" for t in unmatched[:25])
        suffix = f" и ещё {len(unmatched) - 25}" if len(unmatched) > 25 else ""
        lines += ["", f"Не сошлись: {shown}{suffix}."]
    return "\n".join(lines)


def _speed_section(data: dict[str, Any]) -> str:
    bench = data.get("bench_sim")
    if not bench:
        return _MISSING.format(command=PROBES["bench_sim"][1])
    lines = [
        f"Устройство: `{bench['device']}`, нейронов {bench['n_neurons']}, "
        f"рёбер {bench['n_edges']}.",
        f"Цель: не менее {bench['target']['images_per_second']} изобр./с при батче "
        f"{bench['target']['batch']}.",
        "",
        "| Backend | dt, мс | Окно, мс | мс/шаг | с/окно | изобр./с | Цель |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in bench["rows"]:
        if "error" in row:
            lines.append(f"| {row['backend']} | — | — | — | — | — | ошибка: {row['error']} |")
            continue
        lines.append(
            f"| {row['backend']} | {row['dt_ms']} | {row['window_ms']:.0f} | "
            f"{row['ms_per_step']:.2f} | {row['seconds_per_window']:.2f} | "
            f"{row['images_per_second']:.3f} | {'да' if row['meets_target'] else 'нет'} |"
        )
    lines += ["", f"**{bench['verdict']}**"]

    pruning = bench.get("pruning")
    if pruning:
        lines += [
            "",
            "### Обрезка до подграфа",
            "",
            "| Метрика | Значение |",
            "|---|---|",
            f"| k хопов | {pruning['k_hops']} |",
            f"| Нейронов до / после | {pruning['full_n_neurons']} / "
            f"{pruning['pruned_n_neurons']} |",
            f"| Сокращение нейронов | {pruning['neuron_reduction']:.1%} |",
            f"| Сокращение рёбер | {pruning['edge_reduction']:.1%} |",
            f"| Корреляция частот | {pruning['rate_correlation']} |",
            f"| Порог | {pruning['min_correlation']} |",
            f"| **Эквивалентно** | **{'да' if pruning['equivalent'] else 'нет'}** |",
        ]
        if pruning.get("note"):
            lines += ["", f"Замечание: {pruning['note']}"]
        if pruning["neuron_reduction"] < 0.05:
            lines += [
                "",
                "> Обрезка практически ничего не сократила. У графа с тяжёлыми "
                "хвостами степеней маленький диаметр, поэтому за k хопов "
                "достижима почти вся сеть. На эту ступень лестницы ускорения "
                "рассчитывать не стоит.",
            ]
    return "\n".join(lines)


def _regime_section(data: dict[str, Any]) -> str:
    sweep = data.get("sweep_regime")
    if not sweep:
        return _MISSING.format(command=PROBES["sweep_regime"][1])
    lines = [
        f"Целевой диапазон частот: {sweep['target_rate_hz'][0]}–{sweep['target_rate_hz'][1]} Гц. "
        f"Критерий выбора: {sweep['criterion']}.",
    ]
    for key in ("resolution_warning", "saturation_warning"):
        if sweep.get(key):
            lines += ["", f"> {sweep[key]}"]
    lines += [
        "",
        "| Gain | Масштаб весов | Отвечают | Средняя Гц | Макс Гц | PR | В полосе |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sweep["rows"]:
        pr = "—" if row["participation_ratio"] is None else f"{row['participation_ratio']:.2f}"
        lines.append(
            f"| {row['input_gain']} | {row['global_weight_scale']:g} | "
            f"{row['fraction_responsive']:.2f} | {row['mean_responsive_rate_hz']:.1f} | "
            f"{row['max_rate_hz']:.1f} | {pr} | {'да' if row['in_target_band'] else 'нет'} |"
        )
    chosen = sweep.get("chosen")
    lines += ["", f"**{sweep['verdict']}**"]
    if chosen:
        lines += [
            "",
            f"Выбранная точка: `input_gain={chosen['input_gain']}`, "
            f"`global_weight_scale={chosen['global_weight_scale']}` "
            f"(PR {chosen['participation_ratio']}, средняя частота "
            f"{chosen['mean_responsive_rate_hz']} Гц).",
        ]
    return "\n".join(lines)


def _verdict_section(data: dict[str, Any], missing: list[str]) -> str:
    if missing:
        return (
            "**Go/no-go пока не вынесен.** Не хватает результатов: "
            + ", ".join(f"`{PROBES[key][1]}`" for key in missing)
            + ".\n\nРешение принимается после того, как эти команды отработают на "
            "целевом железе с настоящими данными."
        )
    bridge = data.get("probe_bridge", {})
    bench = data.get("bench_sim", {})
    blockers = []
    if bridge.get("ran") and bridge.get("verdict") == "no-go":
        blockers.append("мост flyvis → FlyWire не набрал порог покрытия")
    if bench and bench.get("n_configurations_meeting_target", 0) == 0:
        blockers.append("ни одна конфигурация не вышла на целевую скорость")
    if blockers:
        return "**Go с оговорками.** Требуют решения: " + "; ".join(blockers) + "."
    return "**Go.** Блокирующих находок нет."


def build_m0_report(source: Path) -> tuple[str, list[str], list[str]]:
    """Build the report text. Returns ``(text, found_probes, missing_probes)``."""
    data, found, missing = _load(source)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    banner = _surrogate_banner(data)

    parts = [
        "# M0 — отчёт разведки",
        "",
        f"Сгенерирован `flymog m0-report` {generated} из `{source}`.",
        "Этот файл собирается из JSON-выводов probe-команд. Незаполненные разделы "
        "означают, что соответствующая команда ещё не отработала, а не что проверка "
        "прошла успешно.",
    ]
    if banner:
        parts += ["", banner]

    sections = [
        ("1. Железо", _hardware_section(data)),
        ("2. Данные и доступность", _data_section(data)),
        ("3. Входные клетки (лестница 5.4)", _inputs_section(data)),
        ("4. Мост flyvis → FlyWire", _bridge_section(data)),
        ("5. Скорость симуляции", _speed_section(data)),
        ("6. Режим работы сети", _regime_section(data)),
        ("7. Go / no-go", _verdict_section(data, missing)),
    ]
    for title, body in sections:
        parts += ["", f"## {title}", "", body]

    parts += [
        "",
        "## Что ещё не проверено",
        "",
    ]
    if missing:
        parts += [f"- `{PROBES[key][1]}` — раздел остаётся незаполненным." for key in missing]
    else:
        parts.append("Все probe-команды отработали.")
    parts += [
        "",
        "Отдельно: все параметры LIF в `configs/sim.yaml` помечены `TODO(verify)` и "
        "не сверены с первоисточником. Это нужно закрыть до M1.",
        "",
    ]
    return "\n".join(parts), found, missing
