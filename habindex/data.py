from __future__ import annotations

from pathlib import Path

import pandas as pd

from .index import MODES

PAIR_COLS = ["city_a", "city_b"] + [f"time_{m}" for m in MODES]


class DataError(ValueError):
    pass


def _read(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def canonicalize(pairs: pd.DataFrame) -> pd.DataFrame:
    """Приводит пару (A, B) к виду city_a < city_b, чтобы (A,B) и (B,A) совпадали."""
    p = pairs.copy()
    swap = p["city_a"] > p["city_b"]
    p.loc[swap, ["city_a", "city_b"]] = p.loc[swap, ["city_b", "city_a"]].to_numpy()
    return p.sort_values(["city_a", "city_b"]).reset_index(drop=True)


def load_pairs(path, column_map: dict | None = None) -> pd.DataFrame:
    df = _read(path)
    if column_map:
        df = df.rename(columns=column_map)
    missing = [c for c in PAIR_COLS if c not in df.columns]
    if missing:
        raise DataError(f"В таблице пар нет колонок: {missing}. Есть: {list(df.columns)}")
    df = df[PAIR_COLS].copy()
    for c in PAIR_COLS[:2]:
        df[c] = df[c].astype(str).str.strip()
    for c in PAIR_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return canonicalize(df)


def load_population(path, column_map: dict | None = None, unit: str = "mln") -> pd.Series:
    """Series: город -> население в млн человек. unit: 'mln' | 'thousand' | 'people'."""
    df = _read(path)
    if column_map:
        df = df.rename(columns=column_map)
    for c in ("city", "population"):
        if c not in df.columns:
            raise DataError(f"В таблице населения нет колонки '{c}'. Есть: {list(df.columns)}")
    k = {"mln": 1.0, "thousand": 1e-3, "people": 1e-6}[unit]
    return pd.Series(pd.to_numeric(df["population"], errors="coerce").to_numpy() * k,
                     index=df["city"].astype(str).str.strip(), name="population_mln")


def validate_inputs(pairs: pd.DataFrame, population: pd.Series) -> dict:
    """Жёсткие ошибки -> DataError; мягкие замечания возвращаются в списке warnings."""
    errors, warnings = [], []
    if population.isna().any() or (population <= 0).any():
        errors.append("Население должно быть положительным числом для всех городов.")
    if population.index.duplicated().any():
        errors.append("Дубли городов в таблице населения: "
                      f"{sorted(set(population.index[population.index.duplicated()]))}")
    if (pairs["city_a"] == pairs["city_b"]).any():
        errors.append("Есть строки, где город связан сам с собой.")
    dups = pairs.duplicated(["city_a", "city_b"])
    if dups.any():
        errors.append(f"Повторяющиеся пары: {pairs.loc[dups, ['city_a', 'city_b']].values.tolist()[:5]}")
    times = pairs[[f"time_{m}" for m in MODES]]
    if (times <= 0).any().any():
        errors.append("Время поездки должно быть > 0 (пустая ячейка = нет связи).")
    unknown = (set(pairs["city_a"]) | set(pairs["city_b"])) - set(population.index)
    if unknown:
        errors.append(f"Города из таблицы пар отсутствуют в таблице населения: {sorted(unknown)[:10]}")
    if errors:
        raise DataError(" | ".join(errors))

    n = len(population)
    expected = n * (n - 1) // 2
    if len(pairs) < expected:
        warnings.append(f"Записей о парах {len(pairs)} из {expected} возможных; "
                        "недостающие считаются «нет связи» (S=0).")
    no_link = int(times.isna().all(axis=1).sum())
    if no_link:
        warnings.append(f"Пар без связи ни по одному виду транспорта: {no_link}.")
    if (times > 72).any().any():
        warnings.append("Есть времена > 72 ч — проверьте единицы измерения (часы, а не минуты).")
    return {"n_cities": n, "n_pairs": len(pairs), "warnings": warnings}
