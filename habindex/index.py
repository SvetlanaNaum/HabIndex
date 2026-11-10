from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODES = ("auto", "rail", "air")


@dataclass(frozen=True)
class IndexConfig:
    t_ref: float = 6.0
    w_time: float = 0.7
    m_ref: int = 3
    pop_exponent: float = 1.0
    good_link: float = 60.0

    def __post_init__(self):
        if self.t_ref <= 0:
            raise ValueError("t_ref должен быть > 0")
        if not 0.0 <= self.w_time <= 1.0:
            raise ValueError("w_time должен лежать в [0, 1]")
        if self.m_ref < 1:
            raise ValueError("m_ref должен быть >= 1")
        if self.pop_exponent < 0:
            raise ValueError("pop_exponent должен быть >= 0")

    @property
    def w_modes(self) -> float:
        return 1.0 - self.w_time


def pair_scores(pairs: pd.DataFrame, cfg: IndexConfig = IndexConfig()) -> pd.DataFrame:
    """Индекс S для каждой строки таблицы пар."""

    t = pairs[[f"time_{m}" for m in MODES]].astype(float)
    t_min = t.min(axis=1, skipna=True)
    m = t.notna().sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(m > 0, np.minimum(1.0, cfg.t_ref / t_min.clip(lower=1e-9)), 0.0)
    M = np.minimum(1.0, m / cfg.m_ref)
    S = np.where(m > 0, 100.0 * (cfg.w_time * T + cfg.w_modes * M), 0.0)
    out = pairs[["city_a", "city_b"]].copy()
    out["t_min_h"] = t_min.values
    out["n_modes"] = m.values
    out["T"] = T
    out["M"] = M
    out["S"] = S
    return out.reset_index(drop=True)


def score_matrix(scores: pd.DataFrame, cities) -> pd.DataFrame:
    """Симметричная матрица S. Пара без записи или без связи даёт 0; диагональ — NaN."""
    cities = list(cities)
    mat = pd.DataFrame(0.0, index=cities, columns=cities)
    for a, b, s in scores[["city_a", "city_b", "S"]].itertuples(index=False):
        if a in mat.index and b in mat.index:
            mat.loc[a, b] = s
            mat.loc[b, a] = s
    for c in cities:
        mat.loc[c, c] = np.nan
    return mat


def hub_potential(population: pd.Series, scores: pd.DataFrame,
                  cfg: IndexConfig = IndexConfig()) -> pd.DataFrame:
    """Потенциал хаба H для каждого города и рейтинг (1 — лучший)."""
    cities = list(population.index)
    if len(cities) < 2:
        raise ValueError("Для расчёта нужно не менее двух городов")
    S = score_matrix(scores, cities).fillna(0.0).to_numpy()
    P = population.to_numpy(dtype=float)
    W = P ** cfg.pop_exponent
    H = (S @ W) / (W.sum() - W)
    good = ((S >= cfg.good_link).astype(float) @ P) / (P.sum() - P)
    out = pd.DataFrame({
        "city": cities,
        "population_mln": P,
        "H": H,
        "share_pop_good_link": good,
    })
    out["rank"] = out["H"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["rank", "city"]).reset_index(drop=True)
