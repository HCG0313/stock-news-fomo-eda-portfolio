import unittest

import numpy as np
import pandas as pd

from src.fomo_eda import lag_correlations, next_trade_dates, spike_pairs, validate_inputs


class CalendarTests(unittest.TestCase):
    def test_weekend_holiday_and_unobserved_tail(self):
        dates = pd.Series(pd.to_datetime(["2025-11-07", "2025-11-08", "2025-11-09", "2025-11-11"]))
        # Monday omitted to represent a holiday; actual dates drive the mapping.
        mapped = next_trade_dates(dates, pd.DatetimeIndex(["2025-11-07", "2025-11-11"]))
        self.assertEqual(mapped.iloc[:3].tolist(), [pd.Timestamp("2025-11-11")] * 3)
        self.assertTrue(pd.isna(mapped.iloc[3]))


class PairTests(unittest.TestCase):
    def test_multiple_reactions_count_once_per_search_event(self):
        days = pd.bdate_range("2025-01-01", periods=40)
        values = np.zeros(40)
        values[10:13] = 100
        stocks = pd.DataFrame({"stock": "X", "date": days, "retail_net_buy": values,
                               "retail_buy": np.arange(40) * 1000})
        attention = pd.DataFrame({"stock": "X", "date": days, "search_index": 0.0})
        attention.loc[10, "search_index"] = 100
        details, summary = spike_pairs(stocks, attention)
        self.assertEqual(summary.iloc[0][["search_events", "matched", "unmatched"]].tolist(), [1, 1, 0])
        self.assertEqual(details.iloc[0].first_response_date, days[10])

    def test_unobserved_future_is_censored(self):
        days = pd.bdate_range("2025-01-01", periods=30)
        stocks = pd.DataFrame({"stock": "X", "date": days, "retail_net_buy": 0})
        attention = pd.DataFrame({"stock": "X", "date": days, "search_index": 0.0})
        attention.loc[29, "search_index"] = 100
        _, summary = spike_pairs(stocks, attention)
        self.assertEqual(summary.iloc[0][["matched", "unmatched", "censored"]].tolist(), [0, 0, 1])

    def test_weekend_uses_next_observed_trade_date(self):
        days = pd.bdate_range("2025-01-01", periods=40)
        attention_days = pd.date_range(days.min(), days.max(), freq="D")
        saturday = pd.Timestamp("2025-01-18")
        stocks = pd.DataFrame({"stock": "X", "date": days, "retail_net_buy": 0.0})
        stocks.loc[stocks.date == "2025-01-20", "retail_net_buy"] = 100
        attention = pd.DataFrame({"stock": "X", "date": attention_days, "search_index": 0.0})
        attention.loc[attention.date == saturday, "search_index"] = 100
        details, _ = spike_pairs(stocks, attention)
        self.assertEqual(details.iloc[0].first_response_date, pd.Timestamp("2025-01-20"))


class LagTests(unittest.TestCase):
    def test_forward_lag_has_correct_direction_and_normalization(self):
        x = pd.Series(np.random.default_rng(42).normal(size=100))
        y = x.shift(2)
        result = lag_correlations(x, y)
        self.assertAlmostEqual(result[2], 1.0)
        self.assertTrue(all(-1 <= v <= 1 for v in result))

    def test_constant_series_is_not_replaced_with_zero_correlation(self):
        self.assertTrue(all(np.isnan(v) for v in lag_correlations(pd.Series([1] * 10), pd.Series(range(10)))))


class InputTests(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        stocks = pd.DataFrame({"stock": ["X", "X"], "date": pd.to_datetime(["2025-01-01"] * 2),
                               "close": [1, 1], "return_pct": [0, 0], "volume": [1, 1],
                               "retail_buy": [2, 2], "retail_sell": [1, 1], "retail_net_buy": [1, 1]})
        attention = pd.DataFrame({"stock": ["X"], "date": pd.to_datetime(["2025-01-01"]),
                                  "search_index": [1], "news_count": [1]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_inputs(stocks, attention)


if __name__ == "__main__":
    unittest.main()
