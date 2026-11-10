"""Сборка результатов: CSV, Excel-книга и графики."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from .index import IndexConfig, hub_potential, pair_scores, score_matrix  # noqa: E402
from .scenarios import Scenario, compare_scenarios, evaluate_scenario, priority_table  # noqa: E402
from .sensitivity import sensitivity  # noqa: E402

PURPLE, GREY = "#6b4fbb", "#b9b3cc"


def _style_sheet(ws):
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="6B4FBB")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, col in enumerate(ws.columns, start=1):
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 48)
    ws.freeze_panes = "A2"


def _round(df: pd.DataFrame, nd=2) -> pd.DataFrame:
    return df.round(nd)


def plot_ranking(ranking: pd.DataFrame, path: Path, title: str):
    df = ranking.sort_values("H")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.38 * len(df) + 1)))
    ax.barh(df["city"], df["H"], color=PURPLE)
    for y, v in enumerate(df["H"]):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Потенциал хаба H (0–100)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(scores: pd.DataFrame, cities, path: Path):
    mat = score_matrix(scores, cities)
    fig, ax = plt.subplots(figsize=(0.6 * len(cities) + 3, 0.6 * len(cities) + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="Purples", vmin=0, vmax=100)
    ax.set_xticks(range(len(cities)), cities, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cities)), cities, fontsize=8)
    for i in range(len(cities)):
        for j in range(len(cities)):
            v = mat.iat[i, j]
            if v == v:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else "black")
    fig.colorbar(im, ax=ax, label="Индекс связанности пары S")
    ax.set_title("Матрица связанности S")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(summary: pd.DataFrame, hub: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(summary["scenario"], summary["dH"], color=PURPLE)
    for x, v in enumerate(summary["dH"]):
        ax.text(x, v + 0.03, f"+{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel(f"Прирост H для хаба «{hub}», пункты")
    ax.set_title("Эффект сценариев улучшения связи")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_priorities(pr: pd.DataFrame, hub: str, path: Path, top: int = 8):
    df = pr.head(top).iloc[::-1]
    labels = df["from_city"] + " · " + df["variant"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))
    ax.barh(labels, df["dH"], color=PURPLE)
    ax.set_xlabel(f"Прирост H хаба «{hub}», пункты")
    ax.set_title("С чего начать: наибольший прирост потенциала")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(outdir, pairs: pd.DataFrame, population: pd.Series, cfg: IndexConfig,
                 scenarios: list[Scenario], hub: str | None, label: str = "") -> dict:
    """Считает всё и сохраняет в outdir. Возвращает словарь с таблицами (для тестов и CLI)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pair_scores(pairs, cfg)
    ranking = hub_potential(population, scores, cfg)
    runs, stab = sensitivity(pairs, population, cfg)
    hub = hub or ranking["city"].iloc[0]

    res = {"scores": scores, "ranking": ranking, "sens_runs": runs, "sens_stability": stab, "hub": hub}
    if scenarios:
        res["scenario_summary"] = compare_scenarios(pairs, population, scenarios, hub, cfg)
        res["scenario_details"] = {s.name: evaluate_scenario(pairs, population, s, cfg) for s in scenarios}
    res["priorities"] = priority_table(pairs, population, hub, cfg)

    ranking.to_csv(out / "ranking.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(out / "pair_scores.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(out / "sensitivity_runs.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(out / "sensitivity_stability.csv", index=False, encoding="utf-8-sig")
    res["priorities"].to_csv(out / "priorities.csv", index=False, encoding="utf-8-sig")
    if scenarios:
        res["scenario_summary"].to_csv(out / "scenarios.csv", index=False, encoding="utf-8-sig")

    params = pd.DataFrame({"parameter": list(asdict(cfg)) + ["hub", "data_label"],
                           "value": list(asdict(cfg).values()) + [hub, label]})
    with pd.ExcelWriter(out / "habindex_results.xlsx", engine="openpyxl") as xw:
        _round(ranking).to_excel(xw, sheet_name="Рейтинг хабов", index=False)
        _round(scores).to_excel(xw, sheet_name="Пары (S)", index=False)
        if scenarios:
            _round(res["scenario_summary"]).to_excel(xw, sheet_name="Сценарии", index=False)
            pe = pd.concat([d["pair_effect"].assign(scenario=n)
                            for n, d in res["scenario_details"].items()], ignore_index=True)
            _round(pe).to_excel(xw, sheet_name="Сценарии - пары", index=False)
        _round(res["priorities"]).to_excel(xw, sheet_name="Приоритеты", index=False)
        _round(runs).to_excel(xw, sheet_name="Чувствительность", index=False)
        _round(stab).to_excel(xw, sheet_name="Устойчивость рейтинга", index=False)
        params.to_excel(xw, sheet_name="Параметры", index=False)
        for ws in xw.book.worksheets:
            _style_sheet(ws)

    title = "Потенциал городов как хабов" + (f" ({label})" if label else "")
    plot_ranking(ranking, out / "ranking.png", title)
    plot_heatmap(scores, list(population.index), out / "heatmap_S.png")
    if scenarios:
        plot_scenarios(res["scenario_summary"], hub, out / "scenarios.png")
    plot_priorities(res["priorities"], hub, out / "priorities.png")
    return res
