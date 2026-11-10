from __future__ import annotations

from dataclasses import replace
from itertools import product

import pandas as pd

from .index import IndexConfig, hub_potential, pair_scores


def sensitivity(pairs: pd.DataFrame, population: pd.Series, base: IndexConfig = IndexConfig(),
                t_refs=(4.0, 6.0, 8.0), w_times=(0.6, 0.7, 0.8), exponents=(1.0, 0.5, 0.0),
                top_k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (runs, stability)."""
    base_rank = hub_potential(population, pair_scores(pairs, base), base).set_index("city")["rank"]
    base_top = set(base_rank.nsmallest(top_k).index)

    runs, ranks = [], {}
    for t_ref, w, a in product(t_refs, w_times, exponents):
        cfg = replace(base, t_ref=t_ref, w_time=w, pop_exponent=a)
        res = hub_potential(population, pair_scores(pairs, cfg), cfg).set_index("city")
        rk = res["rank"].reindex(base_rank.index)
        key = (t_ref, w, a)
        ranks[key] = rk
        rho = base_rank.corr(rk, method="spearman")
        top = set(rk.nsmallest(top_k).index)
        runs.append({
            "t_ref_h": t_ref, "w_time": w, "pop_exponent": a,
            "spearman_vs_base": rho,
            f"top{top_k}_overlap": len(base_top & top) / top_k,
            "leader": res["H"].idxmax(),
            "is_base": (t_ref == base.t_ref and w == base.w_time and a == base.pop_exponent),
        })
    runs = pd.DataFrame(runs)

    R = pd.DataFrame(ranks)
    stab = pd.DataFrame({
        "city": R.index,
        "rank_base": base_rank.reindex(R.index).values,
        "rank_best": R.min(axis=1).values,
        "rank_worst": R.max(axis=1).values,
        f"share_runs_in_top{top_k}": (R <= top_k).mean(axis=1).values,
    }).sort_values(["rank_base", "city"]).reset_index(drop=True)
    return runs, stab
