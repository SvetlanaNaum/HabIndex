import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from habindex import (Change, IndexConfig, Scenario, evaluate_scenario, hub_potential,  # noqa: E402
                      load_pairs, load_population, pair_scores, priority_table, sensitivity,
                      validate_inputs)
from habindex.data import DataError  # noqa: E402
from habindex.report import build_report  # noqa: E402
from habindex.scenarios import apply_changes, load_scenarios  # noqa: E402

G = ROOT / "data" / "golden"


def golden():
    return load_pairs(G / "golden_pairs.csv"), load_population(G / "golden_population.csv")


def one_pair(**t):
    row = {"city_a": "A", "city_b": "B", "time_auto": np.nan, "time_rail": np.nan, "time_air": np.nan}
    row.update({f"time_{k}": v for k, v in t.items()})
    return pd.DataFrame([row])


class PairScore(unittest.TestCase):
    def s(self, **t):
        return pair_scores(one_pair(**t))["S"].iloc[0]

    def test_values_from_passport(self):
        pairs, _ = golden()
        sc = pair_scores(pairs).set_index(["city_a", "city_b"])["S"]
        exp = {("Москва", "Нижний Новгород"): 90.0, ("Москва", "Казань"): 90.0,
               ("Москва", "Екатеринбург"): 46.25, ("Казань", "Нижний Новгород"): 80.0,
               ("Екатеринбург", "Нижний Новгород"): 40.0, ("Екатеринбург", "Казань"): 52.0}
        for k, v in exp.items():
            self.assertAlmostEqual(sc[tuple(sorted(k))], v, places=6, msg=str(k))

    def test_no_link_is_zero(self):
        self.assertEqual(self.s(), 0.0)

    def test_time_below_reference_saturates(self):
        self.assertAlmostEqual(self.s(rail=1.0), self.s(rail=6.0))

    def test_bounds(self):
        for t in (0.5, 3, 6, 12, 40):
            v = self.s(auto=t, rail=t, air=t)
            self.assertTrue(0 <= v <= 100)
        self.assertAlmostEqual(self.s(auto=1, rail=1, air=1), 100.0)

    def test_monotone_in_time_and_modes(self):
        self.assertGreater(self.s(rail=5), self.s(rail=10))
        self.assertGreater(self.s(rail=10, air=10), self.s(rail=10))

    def test_min_time_over_modes_not_double_counted(self):
        self.assertAlmostEqual(self.s(rail=12, air=8), self.s(rail=8, air=12))

    def test_bad_config(self):
        for kw in ({"t_ref": 0}, {"w_time": 1.5}, {"m_ref": 0}, {"pop_exponent": -1}):
            with self.assertRaises(ValueError):
                IndexConfig(**kw)


class HubPotential(unittest.TestCase):
    def test_golden_values(self):
        pairs, pop = golden()
        h = hub_potential(pop, pair_scores(pairs)).set_index("city")["H"]
        self.assertAlmostEqual(h["Нижний Новгород"], 84.43, places=2)
        self.assertAlmostEqual(h["Казань"], 85.61, places=2)
        self.assertAlmostEqual(h["Екатеринбург"], 46.25, places=2)

    def test_manual_formula_moscow(self):
        pairs, pop = golden()
        h = hub_potential(pop, pair_scores(pairs)).set_index("city")["H"]
        manual = (1.2 * 90 + 1.3 * 90 + 1.5 * 46.25) / (1.2 + 1.3 + 1.5)
        self.assertAlmostEqual(h["Москва"], manual, places=9)

    def test_rank_order(self):
        pairs, pop = golden()
        r = hub_potential(pop, pair_scores(pairs))
        self.assertEqual(list(r["city"][:3]), ["Казань", "Нижний Новгород", "Москва"])
        self.assertEqual(r["city"].iloc[-1], "Екатеринбург")

    def test_equal_weights_is_plain_mean(self):
        pairs, pop = golden()
        cfg = IndexConfig(pop_exponent=0.0)
        h = hub_potential(pop, pair_scores(pairs, cfg), cfg).set_index("city")["H"]
        self.assertAlmostEqual(h["Казань"], (90 + 80 + 52) / 3, places=9)

    def test_missing_pair_counts_as_no_link(self):
        pairs, pop = golden()
        full = hub_potential(pop, pair_scores(pairs)).set_index("city")["H"]
        cut = pairs[~((pairs.city_a == "Казань") & (pairs.city_b == "Нижний Новгород"))
                    & ~((pairs.city_a == "Нижний Новгород") & (pairs.city_b == "Казань"))]
        part = hub_potential(pop, pair_scores(cut)).set_index("city")["H"]
        self.assertLess(part["Казань"], full["Казань"])


class Scenarios(unittest.TestCase):
    def setUp(self):
        self.pairs, self.pop = golden()
        _, self.scs = load_scenarios(ROOT / "scenarios" / "golden_scenarios.json")

    def test_passport_scenarios(self):
        exp = {"1. Ускорение до 5 ч": (80.0, 88.28, 2.68), "2. Второй вид транспорта": (62.0, 86.56, 0.96),
               "3. Ускорение и второй вид": (90.0, 89.24, 3.63)}
        for sc in self.scs:
            res = evaluate_scenario(self.pairs, self.pop, sc)
            s, h, d = exp[sc.name]
            self.assertAlmostEqual(res["pair_effect"]["S_after"].iloc[0], s, places=6)
            row = res["city_effect"].set_index("city").loc["Казань"]
            self.assertAlmostEqual(row["H_after"], h, places=2)
            self.assertAlmostEqual(row["dH"], d, delta=0.011)

    def test_original_not_mutated(self):
        before = self.pairs.copy()
        apply_changes(self.pairs, self.scs[0].changes)
        pd.testing.assert_frame_equal(before, self.pairs)

    def test_pair_order_irrelevant(self):
        a = apply_changes(self.pairs, [Change("Казань", "Екатеринбург", "rail", 5)])
        b = apply_changes(self.pairs, [Change("Екатеринбург", "Казань", "rail", 5)])
        pd.testing.assert_frame_equal(a, b)

    def test_remove_mode_lowers_score(self):
        sc = Scenario("убрать ЖД", [Change("Казань", "Москва", "rail", None)])
        r = evaluate_scenario(self.pairs, self.pop, sc)
        self.assertLess(r["pair_effect"]["dS"].iloc[0], 0)

    def test_new_pair_is_added(self):
        p = apply_changes(self.pairs.iloc[:0], [Change("A", "B", "air", 3.0)])
        self.assertEqual(len(p), 1)

    def test_bad_change(self):
        with self.assertRaises(ValueError):
            Change("A", "B", "ship", 3)
        with self.assertRaises(ValueError):
            Change("A", "B", "rail", -1)

    def test_priority_matches_full_recalculation(self):
        pr = priority_table(self.pairs, self.pop, "Казань", speedup=0.25)
        row = pr[(pr.from_city == "Екатеринбург") & (pr.variant.str.startswith("ускорение на"))].iloc[0]
        sc = Scenario("x", [Change("Казань", "Екатеринбург", "rail", 10 * 0.75)])
        dh = evaluate_scenario(self.pairs, self.pop, sc)["city_effect"].set_index("city").loc["Казань", "dH"]
        self.assertAlmostEqual(row["dH"], dh, places=9)
        self.assertTrue((pr["dH"].diff().dropna() <= 1e-12).all())  # отсортировано по убыванию

    def test_priority_unknown_hub(self):
        with self.assertRaises(KeyError):
            priority_table(self.pairs, self.pop, "Атлантида")


class Sensitivity(unittest.TestCase):
    def test_base_run_is_identical(self):
        pairs, pop = golden()
        runs, stab = sensitivity(pairs, pop)
        base = runs[runs.is_base].iloc[0]
        self.assertAlmostEqual(base["spearman_vs_base"], 1.0)
        self.assertEqual(len(runs), 27)
        self.assertTrue((stab["rank_best"] <= stab["rank_base"]).all())
        self.assertTrue((stab["rank_worst"] >= stab["rank_base"]).all())

    def test_weakest_city_stays_last(self):
        pairs, pop = golden()
        _, stab = sensitivity(pairs, pop)
        self.assertEqual(stab.set_index("city").loc["Екатеринбург", "rank_worst"], 4)


class DataChecks(unittest.TestCase):
    def test_golden_valid(self):
        pairs, pop = golden()
        info = validate_inputs(pairs, pop)
        self.assertEqual((info["n_cities"], info["n_pairs"]), (4, 6))
        self.assertEqual(info["warnings"], [])

    def test_errors(self):
        pairs, pop = golden()
        bad = pairs.copy(); bad.loc[0, "time_rail"] = -3
        with self.assertRaises(DataError):
            validate_inputs(bad, pop)
        with self.assertRaises(DataError):
            validate_inputs(pd.concat([pairs, pairs.iloc[:1]], ignore_index=True), pop)
        with self.assertRaises(DataError):
            validate_inputs(pairs, pop.drop("Казань"))

    def test_missing_pairs_warning(self):
        pairs, pop = golden()
        info = validate_inputs(pairs.iloc[:4], pop)
        self.assertTrue(info["warnings"])

    def test_column_map_and_units(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            p.write_text("from,to,auto,rail,air\nB,A,,5,\n", encoding="utf-8")
            df = load_pairs(p, {"from": "city_a", "to": "city_b", "auto": "time_auto",
                                "rail": "time_rail", "air": "time_air"})
            self.assertEqual((df.city_a[0], df.city_b[0]), ("A", "B"))
            q = Path(d) / "q.csv"
            q.write_text("city,population\nA,1200000\n", encoding="utf-8")
            self.assertAlmostEqual(load_population(q, unit="people")["A"], 1.2)

    def test_missing_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(DataError):
                load_pairs(p)


class EndToEnd(unittest.TestCase):
    def test_build_report_creates_files(self):
        pairs, pop = golden()
        hub, scs = load_scenarios(ROOT / "scenarios" / "golden_scenarios.json")
        with tempfile.TemporaryDirectory() as d:
            res = build_report(d, pairs, pop, IndexConfig(), scs, hub, "test")
            for f in ("ranking.csv", "habindex_results.xlsx", "ranking.png", "heatmap_S.png",
                      "scenarios.png", "priorities.png"):
                self.assertTrue((Path(d) / f).exists(), f)
            self.assertEqual(res["ranking"]["city"].iloc[0], "Казань")


if __name__ == "__main__":
    unittest.main()
