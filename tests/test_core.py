"""ForecastOps core test suite (unittest).

Run: python -m unittest discover -s tests -v   (from the project root)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commentary import deterministic_commentary  # noqa: E402
from forecasting import (  # noqa: E402
    BU,
    TD,
    apply_scenario,
    choose_reconciliation,
    forecast_all_leaves,
    historical_shares,
    ets_forecast,
    leaf_keys,
    load_data,
    national_history,
    reconcile,
    walk_forward_backtest,
    wape,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo_data", "demand_monthly.csv",
)


class TestData(unittest.TestCase):
    def test_csv_loads_with_expected_columns_and_48_months(self):
        df = load_data(DATA_PATH)
        self.assertEqual(set(df.columns), {"date", "component", "zone", "units"})
        self.assertEqual(df["date"].nunique(), 48)
        self.assertEqual(len(df), 576)  # 3 components x 4 zones x 48 months
        self.assertEqual(set(df["component"].unique()),
                         {"Compressor", "Condenser", "Evaporator"})
        self.assertEqual(set(df["zone"].unique()),
                         {"North", "South", "West", "East"})


class TestMetrics(unittest.TestCase):
    def test_wape_matches_hand_computed_value(self):
        actual = np.array([10.0, 10.0, 10.0, 10.0])
        forecast = np.array([8.0, 12.0, 10.0, 10.0])
        # sum|a-f| = 2+2+0+0 = 4 ; sum|a| = 40 -> 0.1
        self.assertAlmostEqual(wape(actual, forecast), 0.10, places=10)


class TestForecasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data(DATA_PATH)
        cls.keys = leaf_keys(cls.df)
        cls.leaf_fc = forecast_all_leaves(cls.df, horizon=6)

    def test_ets_forecast_returns_positive_floats(self):
        hist = national_history(self.df)
        fc = ets_forecast(hist, 6)
        self.assertEqual(len(fc), 6)
        self.assertTrue(all(isinstance(v, (int, float, np.floating)) for v in fc))
        self.assertTrue(np.all(np.isfinite(fc)))
        self.assertTrue(np.all(fc > 0))

    def test_bottom_up_national_equals_sum_of_leaves(self):
        shares = historical_shares(self.df)
        _, bu_national = reconcile(self.leaf_fc, np.zeros(6), shares, BU)
        leaf_sum = np.sum([self.leaf_fc[k] for k in self.keys], axis=0)
        np.testing.assert_allclose(bu_national, leaf_sum, rtol=1e-9, atol=1e-6)

    def test_top_down_equals_national_disaggregated(self):
        national_fc = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0])
        shares = historical_shares(self.df)
        leaves, td_national = reconcile(self.leaf_fc, national_fc, shares, TD)
        np.testing.assert_allclose(td_national, national_fc, rtol=1e-9)
        leaf_sum = np.sum([leaves[k] for k in self.keys], axis=0)
        np.testing.assert_allclose(leaf_sum, national_fc, rtol=1e-9, atol=1e-6)

    def test_reconciliation_chooser_returns_valid_method(self):
        self.assertIn(choose_reconciliation(0.10, 0.12), {BU, TD})
        self.assertIn(choose_reconciliation(0.12, 0.10), {BU, TD})
        self.assertIn(choose_reconciliation(0.10, 0.10), {BU, TD})
        self.assertEqual(choose_reconciliation(0.10, 0.12), BU)
        self.assertEqual(choose_reconciliation(0.12, 0.10), TD)

    def test_scenario_multiplier_changes_south_about_10_percent(self):
        base = apply_scenario(self.leaf_fc, zones=[], components=[], multiplier=1.0)
        scen = apply_scenario(self.leaf_fc, zones=["South"], components=[], multiplier=1.1)
        base_south = np.sum([base[k] for k in self.keys if k[1] == "South"], axis=0)
        scen_south = np.sum([scen[k] for k in self.keys if k[1] == "South"], axis=0)
        np.testing.assert_allclose(scen_south, base_south * 1.1, rtol=1e-9)
        # other zones untouched
        base_north = np.sum([base[k] for k in self.keys if k[1] == "North"], axis=0)
        scen_north = np.sum([scen[k] for k in self.keys if k[1] == "North"], axis=0)
        np.testing.assert_allclose(scen_north, base_north, rtol=1e-12)

    def test_backtest_completes_and_produces_metrics(self):
        bt = walk_forward_backtest(self.df, folds=6, horizon=3)
        self.assertIn(bt.chosen_method, {BU, TD})
        self.assertFalse(np.isnan(bt.bu_national_wape))
        self.assertFalse(np.isnan(bt.td_national_wape))
        self.assertFalse(bt.level_metrics.empty)
        self.assertFalse(bt.series_metrics.empty)
        self.assertEqual(len(bt.folds), 6)


class TestCommentary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data(DATA_PATH)

    def test_commentary_non_empty_and_mentions_seasonality(self):
        text = deterministic_commentary(self.df, scenario=None, backtest=None)
        self.assertTrue(text.strip())
        self.assertIn("[DEMO MODE", text)
        self.assertIn("easonality", text)  # 'Seasonality:' heading present

    def test_commentary_mentions_scenario(self):
        text = deterministic_commentary(
            self.df, scenario={"zones": ["South"], "components": [], "multiplier": 1.1})
        self.assertIn("x1.1", text)
        self.assertIn("South", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
