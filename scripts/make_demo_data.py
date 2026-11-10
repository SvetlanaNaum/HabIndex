from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

CITIES = {  # город: (население млн, округлённо; широта; долгота)
    "Москва": (13.0, 55.75, 37.62), "Санкт-Петербург": (5.6, 59.93, 30.34),
    "Новосибирск": (1.6, 55.03, 82.92), "Екатеринбург": (1.5, 56.84, 60.60),
    "Казань": (1.3, 55.79, 49.12), "Нижний Новгород": (1.2, 56.33, 44.00),
    "Самара": (1.2, 53.20, 50.15), "Челябинск": (1.1, 55.16, 61.40),
    "Уфа": (1.1, 54.74, 55.97), "Ростов-на-Дону": (1.1, 47.23, 39.72),
    "Пермь": (1.0, 58.01, 56.23), "Воронеж": (1.0, 51.67, 39.18),
}
# Допущения (меняйте при необходимости — результат пересчитается)
ROAD_KMH, ROAD_DETOUR, ROAD_MAX_KM = 70.0, 1.25, 1500.0
RAIL_KMH, RAIL_DETOUR = 80.0, 1.20
AIR_KMH, AIR_OVERHEAD_H, AIR_MIN_KM = 700.0, 3.0, 500.0
# Допущение: прямая авиация только между Москвой/Санкт-Петербургом и остальными + Екатеринбург—Новосибирск
AIR_HUBS = {"Москва", "Санкт-Петербург"}
AIR_EXTRA = {frozenset(("Екатеринбург", "Новосибирск"))}


def haversine(a, b):
    la1, lo1, la2, lo2 = map(radians, (*a, *b))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def main():
    out = Path(__file__).resolve().parent.parent / "data" / "demo"
    names = list(CITIES)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = haversine(CITIES[a][1:], CITIES[b][1:])
            road = round(d * ROAD_DETOUR / ROAD_KMH, 1) if d * ROAD_DETOUR <= ROAD_MAX_KM else None
            rail = round(d * RAIL_DETOUR / RAIL_KMH, 1)
            has_air = (a in AIR_HUBS or b in AIR_HUBS or frozenset((a, b)) in AIR_EXTRA) and d >= AIR_MIN_KM
            air = round(AIR_OVERHEAD_H + d / AIR_KMH, 1) if has_air else None
            rows.append({"city_a": a, "city_b": b, "time_auto": road, "time_rail": rail, "time_air": air})
    pd.DataFrame(rows).to_csv(out / "demo_pairs.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"city": c, "population": v[0], "lat": v[1], "lon": v[2]} for c, v in CITIES.items()]
                 ).to_csv(out / "demo_population.csv", index=False, encoding="utf-8")
    print("демо-данные записаны в", out)


if __name__ == "__main__":
    main()
