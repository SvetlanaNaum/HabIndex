from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import canonicalize
from .index import MODES, IndexConfig, hub_potential, pair_scores


@dataclass(frozen=True)
class Change:
    """Изменение одной пары: задать время по виду транспорта (time_h=None — убрать вид)."""
    a: str
    b: str
    mode: str                      # 'auto' | 'rail' | 'air'
    time_h: Optional[float]

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Неизвестный вид транспорта '{self.mode}', допустимо: {MODES}")
        if self.time_h is not None and self.time_h <= 0:
            raise ValueError("time_h должно быть > 0 или None")


@dataclass
class Scenario:
    name: str
    changes: list[Change] = field(default_factory=list)
    description: str = ""


def apply_changes(pairs: pd.DataFrame, changes: list[Change]) -> pd.DataFrame:
    """Копия таблицы пар с применёнными изменениями. Исходная таблица не меняется."""
    p = pairs.copy()
    tcols = [f"time_{m}" for m in MODES]
    p[tcols] = p[tcols].astype(float)
    for ch in changes:
        a, b = sorted((ch.a, ch.b))
        col = f"time_{ch.mode}"
        mask = (p["city_a"] == a) & (p["city_b"] == b)
        if not mask.any():
            row = {"city_a": a, "city_b": b, **{f"time_{m}": np.nan for m in MODES}}
            p = pd.concat([p, pd.DataFrame([row])], ignore_index=True)
            mask = (p["city_a"] == a) & (p["city_b"] == b)
        p.loc[mask, col] = np.nan if ch.time_h is None else float(ch.time_h)
    return canonicalize(p)


def evaluate_scenario(pairs: pd.DataFrame, population: pd.Series, scenario: Scenario,
                      cfg: IndexConfig = IndexConfig()) -> dict:
    """Пересчёт индекса при сценарии.

    Возвращает: pair_effect (S до/после для затронутых пар) и city_effect (H и место до/после).
    """
    base_scores = pair_scores(pairs, cfg)
    new_pairs = apply_changes(pairs, scenario.changes)
    new_scores = pair_scores(new_pairs, cfg)

    keys = sorted({tuple(sorted((c.a, c.b))) for c in scenario.changes})
    rows = []
    for a, b in keys:
        b0 = base_scores[(base_scores.city_a == a) & (base_scores.city_b == b)]
        n0 = new_scores[(new_scores.city_a == a) & (new_scores.city_b == b)]
        rows.append({
            "pair": f"{a} — {b}",
            "t_before_h": b0["t_min_h"].iloc[0] if len(b0) else np.nan,
            "modes_before": int(b0["n_modes"].iloc[0]) if len(b0) else 0,
            "S_before": b0["S"].iloc[0] if len(b0) else 0.0,
            "t_after_h": n0["t_min_h"].iloc[0],
            "modes_after": int(n0["n_modes"].iloc[0]),
            "S_after": n0["S"].iloc[0],
        })
    pair_effect = pd.DataFrame(rows)
    pair_effect["dS"] = pair_effect["S_after"] - pair_effect["S_before"]

    h0 = hub_potential(population, base_scores, cfg)[["city", "H", "rank"]]
    h1 = hub_potential(population, new_scores, cfg)[["city", "H", "rank"]]
    city = h0.merge(h1, on="city", suffixes=("_before", "_after"))
    city["dH"] = city["H_after"] - city["H_before"]
    city["rank_change"] = city["rank_before"] - city["rank_after"]   # >0 — город поднялся
    city = city.sort_values("dH", ascending=False).reset_index(drop=True)
    return {"scenario": scenario, "pair_effect": pair_effect, "city_effect": city}


def compare_scenarios(pairs, population, scenarios, hub: str,
                      cfg: IndexConfig = IndexConfig()) -> pd.DataFrame:
    """Сводная таблица: эффект каждого сценария для выбранного хаба."""
    rows = []
    for sc in scenarios:
        res = evaluate_scenario(pairs, population, sc, cfg)
        c = res["city_effect"].set_index("city").loc[hub]
        rows.append({
            "scenario": sc.name,
            "S_pair_after": res["pair_effect"]["S_after"].iloc[0] if len(res["pair_effect"]) else np.nan,
            "H_before": c["H_before"], "H_after": c["H_after"], "dH": c["dH"],
            "rank_before": int(c["rank_before"]), "rank_after": int(c["rank_after"]),
        })
    return pd.DataFrame(rows)


def priority_table(pairs: pd.DataFrame, population: pd.Series, hub: str,
                   cfg: IndexConfig = IndexConfig(), speedup: float = 0.25,
                   new_link_time_h: float = 9.0) -> pd.DataFrame:
    """Какое улучшение связей хаба даёт наибольший прирост H — «с чего начать».

    Для каждого города-отправителя i проверяются типовые варианты:
      «ускорение»      — время сокращается на speedup (по умолчанию 25 %);
      «+1 вид»         — добавляется ещё один вид транспорта с тем же минимальным временем;
      «ускорение+вид»  — оба изменения;
      для пары вообще без связи — «новая связь» (один вид, new_link_time_h часов).
    Прирост dH считается аналитически: dH = W_i * (S_new - S_old) / sum_{k != hub} W_k.
    """
    if hub not in population.index:
        raise KeyError(f"Города '{hub}' нет в таблице населения")
    scores = pair_scores(pairs, cfg)
    W = population ** cfg.pop_exponent
    denom = W.sum() - W[hub]

    def s_of(t, m):
        if m <= 0:
            return 0.0
        return 100.0 * (cfg.w_time * min(1.0, cfg.t_ref / t) + cfg.w_modes * min(1.0, m / cfg.m_ref))

    rows = []
    for city in population.index:
        if city == hub:
            continue
        a, b = sorted((city, hub))
        r = scores[(scores.city_a == a) & (scores.city_b == b)]
        t, m = (float(r["t_min_h"].iloc[0]), int(r["n_modes"].iloc[0])) if len(r) and r["n_modes"].iloc[0] > 0 else (np.nan, 0)
        s_old = s_of(t, m) if m else 0.0
        variants = []
        if m == 0:
            variants.append(("новая связь", s_of(new_link_time_h, 1)))
        else:
            variants.append((f"ускорение на {int(speedup * 100)} %", s_of(t * (1 - speedup), m)))
            if m < cfg.m_ref:
                variants.append(("+1 вид транспорта", s_of(t, m + 1)))
                variants.append(("ускорение + доп. вид", s_of(t * (1 - speedup), m + 1)))
        for name, s_new in variants:
            rows.append({
                "from_city": city, "population_mln": population[city],
                "t_now_h": t, "modes_now": m, "S_now": s_old,
                "variant": name, "S_new": s_new,
                "dH": W[city] * (s_new - s_old) / denom,
            })
    out = pd.DataFrame(rows)
    return out.sort_values("dH", ascending=False).reset_index(drop=True)


def load_scenarios(path) -> tuple[str | None, list[Scenario]]:
    """JSON: {"hub": "Казань", "scenarios": [{"name": ..., "changes": [{"a":..,"b":..,"mode":..,"time_h":..}]}]}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for s in data["scenarios"]:
        ch = [Change(c["a"], c["b"], c["mode"], c.get("time_h")) for c in s["changes"]]
        out.append(Scenario(s["name"], ch, s.get("description", "")))
    return data.get("hub"), out
