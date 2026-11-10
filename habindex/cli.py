"""Командная строка.

  python -m habindex run  --pairs P.csv --pop POP.csv [--scenarios S.json] [--hub Казань] [--out outputs/x]
  python -m habindex demo                      # пересчёт обеих встроенных выборок
  python -m habindex check --pairs P.csv --pop POP.csv   # только проверка входных таблиц
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import DataError, load_pairs, load_population, validate_inputs
from .index import IndexConfig
from .report import build_report
from .scenarios import load_scenarios

ROOT = Path(__file__).resolve().parent.parent


def _cfg(a) -> IndexConfig:
    return IndexConfig(t_ref=a.t_ref, w_time=a.w_time, pop_exponent=a.exponent)


def _load(a):
    cmap = json.loads(a.column_map) if getattr(a, "column_map", None) else None
    pairs = load_pairs(a.pairs, cmap)
    pop = load_population(a.pop, unit=a.pop_unit)
    info = validate_inputs(pairs, pop)
    for w in info["warnings"]:
        print(f"[предупреждение] {w}", file=sys.stderr)
    return pairs, pop, info


def cmd_check(a):
    _, _, info = _load(a)
    print(f"OK: городов {info['n_cities']}, пар {info['n_pairs']}")


def cmd_run(a):
    pairs, pop, _ = _load(a)
    hub, scenarios = (None, [])
    if a.scenarios:
        hub, scenarios = load_scenarios(a.scenarios)
    res = build_report(a.out, pairs, pop, _cfg(a), scenarios, a.hub or hub, a.label)
    print(res["ranking"].round(2).to_string(index=False))
    print(f"\nРезультаты сохранены в {a.out}")


def cmd_demo(a):
    for name, label, sc in (("golden", "учебный пример, 4 города (синтетика)", "golden_scenarios.json"),
                            ("demo", "демо-выборка, 12 городов (синтетика)", "demo_scenarios.json")):
        d = ROOT / "data" / name
        pairs = load_pairs(d / f"{name}_pairs.csv")
        pop = load_population(d / f"{name}_population.csv")
        validate_inputs(pairs, pop)
        hub, scenarios = load_scenarios(ROOT / "scenarios" / sc)
        out = ROOT / "outputs" / name
        res = build_report(out, pairs, pop, IndexConfig(), scenarios, hub, label)
        print(f"\n=== {label} ===")
        print(res["ranking"].round(2).to_string(index=False))
        print(res["scenario_summary"].round(2).to_string(index=False))
        print(f"-> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="habindex", description="ХабИндекс: связанность городов и потенциал хаба")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--pairs", required=True, help="CSV/XLSX с парами городов")
        sp.add_argument("--pop", required=True, help="CSV/XLSX с населением")
        sp.add_argument("--pop-unit", default="mln", choices=["mln", "thousand", "people"])
        sp.add_argument("--column-map", help='JSON переименования колонок, напр. {"from":"city_a"}')

    sp = sub.add_parser("check"); common(sp); sp.set_defaults(func=cmd_check)
    sp = sub.add_parser("run"); common(sp)
    sp.add_argument("--scenarios"); sp.add_argument("--hub"); sp.add_argument("--label", default="")
    sp.add_argument("--out", default=str(ROOT / "outputs" / "run"))
    sp.add_argument("--t-ref", type=float, default=6.0, dest="t_ref")
    sp.add_argument("--w-time", type=float, default=0.7, dest="w_time")
    sp.add_argument("--exponent", type=float, default=1.0)
    sp.set_defaults(func=cmd_run)
    sp = sub.add_parser("demo"); sp.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    try:
        a.func(a)
    except (DataError, KeyError, ValueError, FileNotFoundError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0
