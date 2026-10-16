"""Post-coursework cleanup of the news/search and retail trading EDA.

All indicators are descriptive proxies. They do not measure investor psychology
or establish a causal effect. See docs/review_status.md for changes from class.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["stock", "date"]
STOCK_NUMBERS = ["close", "return_pct", "volume", "retail_buy", "retail_sell", "retail_net_buy"]
DISPLAY_NAMES = {
    "삼성전자": "Samsung Electronics", "NAVER": "NAVER", "카카오": "Kakao",
    "LG에너지솔루션": "LG Energy Solution", "셀트리온": "Celltrion",
    "에코프로": "Ecopro", "현대차": "Hyundai Motor", "한화오션": "Hanwha Ocean",
    "펄어비스": "Pearl Abyss", "포스코퓨처엠": "POSCO Future M",
}


def validate_inputs(stocks: pd.DataFrame, attention: pd.DataFrame) -> None:
    for frame, required in [(stocks, KEY + STOCK_NUMBERS),
                            (attention, KEY + ["search_index", "news_count"])]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError("Empty input or missing source observations")
        if frame.duplicated(KEY).any():
            raise ValueError("Duplicate stock/date key")
        numbers = frame[[c for c in required if c not in KEY]].to_numpy(dtype=float)
        if not np.isfinite(numbers).all():
            raise ValueError("Non-finite source values")
    if set(stocks.stock) != set(attention.stock):
        raise ValueError("Stocks differ between source tables")
    if (attention[["search_index", "news_count"]] < 0).any().any():
        raise ValueError("Negative attention observation")
    if (stocks[["volume", "retail_buy", "retail_sell"]] < 0).any().any():
        raise ValueError("Negative trading volume")
    if not np.allclose(stocks.retail_net_buy, stocks.retail_buy - stocks.retail_sell):
        raise ValueError("Retail net buy does not equal buy minus sell")
    calendar = pd.DatetimeIndex(sorted(stocks.date.unique()))
    all_days = pd.date_range(attention.date.min(), attention.date.max(), freq="D")
    for stock in sorted(stocks.stock.unique()):
        observed = stocks.loc[stocks.stock == stock, "date"]
        if len(calendar.difference(observed)):
            raise ValueError(f"Missing market-date stock observations: {stock}")
        days = attention.loc[attention.stock == stock, "date"]
        if len(days) != len(all_days) or len(all_days.difference(days)):
            raise ValueError(f"Missing calendar-day attention observations: {stock}")


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stocks = pd.read_csv(data_dir / "stock_daily.csv", parse_dates=["date"])
    attention = pd.read_csv(data_dir / "attention_daily.csv", parse_dates=["date"])
    validate_inputs(stocks, attention)
    return stocks.sort_values(KEY).reset_index(drop=True), attention.sort_values(KEY).reset_index(drop=True)


def next_trade_dates(dates: pd.Series, trades: pd.DatetimeIndex) -> pd.Series:
    """Strictly later observed market date; the unobserved tail stays NaT."""
    trades = pd.DatetimeIndex(trades).sort_values().unique()
    positions = trades.searchsorted(pd.DatetimeIndex(dates), side="right")
    values = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = positions < len(trades)
    values[valid] = trades.to_numpy(dtype="datetime64[ns]")[positions[valid]]
    return pd.Series(values, index=dates.index)


def make_panel(stocks: pd.DataFrame, attention: pd.DataFrame) -> pd.DataFrame:
    validate_inputs(stocks, attention)
    panels = []
    for stock, a in attention.groupby("stock", sort=True):
        s = stocks[stocks.stock == stock].sort_values("date").copy()
        a = a.sort_values("date").copy()
        g = a.merge(s, on=KEY, how="left", validate="one_to_one", indicator=True)
        g["is_trading_day"] = g.pop("_merge").eq("both")
        g["close"] = g["close"].ffill()
        flows = [c for c in STOCK_NUMBERS if c != "close"]
        g.loc[~g.is_trading_day, flows] = 0
        g["issue_intensity"] = np.log1p(g.news_count) + np.log1p(g.search_index)
        g["abs_return_pct"] = g.return_pct.abs()
        g["log_volume"] = np.log1p(g.volume)
        g["next_trade_date"] = next_trade_dates(g.date, pd.DatetimeIndex(s.date))
        future = s[["date", "return_pct", "volume"]].rename(columns={
            "date": "next_trade_date", "return_pct": "next_return_pct", "volume": "next_volume",
        })
        g = g.merge(future, on="next_trade_date", how="left", validate="many_to_one")
        g["next_log_volume"] = np.log1p(g.next_volume)
        panels.append(g)
    return pd.concat(panels, ignore_index=True).sort_values(KEY).reset_index(drop=True)


def lag_correlations(search: pd.Series, net_buy: pd.Series, max_lag: int = 3) -> list[float]:
    """Pearson corr(search[t], net_buy[t+lag]) on aligned trading observations."""
    values = []
    for lag in range(max_lag + 1):
        pair = pd.concat([search.reset_index(drop=True), net_buy.reset_index(drop=True).shift(-lag)], axis=1).dropna()
        if len(pair) < 3 or (pair.std() == 0).any():
            values.append(np.nan)
        else:
            values.append(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
    return values


def fomo_features(stocks: pd.DataFrame, attention: pd.DataFrame) -> pd.DataFrame:
    """Exploratory, sample-dependent index; not a validated FOMO measurement."""
    features = []
    for stock, s in stocks.groupby("stock", sort=True):
        s = s.sort_values("date")
        a = attention[attention.stock == stock].sort_values("date")
        aligned = s.merge(a, on=KEY, validate="one_to_one")
        correlations = lag_correlations(aligned.search_index, aligned.retail_net_buy)
        finite = [(i, v) for i, v in enumerate(correlations) if np.isfinite(v)]
        best_lag, best_corr = max(finite, key=lambda item: item[1]) if finite else (np.nan, np.nan)
        row = {
            "stock": stock, "news_std": a.news_count.std(),
            "search_std": a.search_index.std(), "net_buy_std": s.retail_net_buy.std(),
            "max_forward_pearson": best_corr, "selected_lag_trading_days": best_lag,
        }
        row.update({f"corr_lag_{i}": v for i, v in enumerate(correlations)})
        features.append(row)
    result = pd.DataFrame(features)
    columns = ["news_std", "search_std", "net_buy_std", "max_forward_pearson"]
    x = result[columns]
    span = x.max() - x.min()
    scaled = (x - x.min()).div(span.replace(0, 1))
    result["FOMO_Index"] = scaled.mean(axis=1, skipna=False)
    return result


def spike_pairs(stocks: pd.DataFrame, attention: pd.DataFrame, max_lag: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each search event has at most one first response, observed over trading dates.

    A truncated tail is censored instead of counted as no response. A response
    may be shared by separate search events; it is not a causal attribution.
    """
    details, summaries = [], []
    for stock, a in attention.groupby("stock", sort=True):
        a = a.sort_values("date")
        s = stocks[stocks.stock == stock].sort_values("date").reset_index(drop=True)
        search_cutoff = a.search_index.mean() + 2 * a.search_index.std()
        order_cutoff = s.retail_net_buy.mean() + 2 * s.retail_net_buy.std()
        search_dates = a.loc[a.search_index > search_cutoff, "date"]
        order_dates = pd.DatetimeIndex(s.loc[s.retail_net_buy > order_cutoff, "date"])
        trades = pd.DatetimeIndex(s.date)
        rows = []
        for day in search_dates:
            start = trades.searchsorted(day, side="left")
            window = trades[start:start + max_lag + 1]
            hits = window.intersection(order_dates)
            complete = len(window) == max_lag + 1
            response = hits[0] if len(hits) else pd.NaT
            lag = int(window.get_loc(response)) if len(hits) else np.nan
            row = {"stock": stock, "search_date": day, "first_response_date": response,
                   "response_lag_trading_days": lag, "full_window_observed": complete,
                   "status": "matched" if len(hits) else ("unmatched" if complete else "censored")}
            rows.append(row)
            details.append(row)
        matched = sum(r["status"] == "matched" for r in rows)
        unmatched = sum(r["status"] == "unmatched" for r in rows)
        censored = sum(r["status"] == "censored" for r in rows)
        summaries.append({"stock": stock, "search_events": len(rows), "matched": matched,
                          "unmatched": unmatched, "censored": censored,
                          "matched_fraction_observed": matched / (matched + unmatched) if matched + unmatched else np.nan,
                          "search_cutoff": search_cutoff, "net_buy_cutoff": order_cutoff})
    return pd.DataFrame(details, columns=["stock", "search_date", "first_response_date", "response_lag_trading_days", "full_window_observed", "status"]), pd.DataFrame(summaries)


def cluster_events(stocks: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    rows = []
    for stock, g in panel.groupby("stock", sort=True):
        peak = g.loc[g.issue_intensity.idxmax()]
        s = stocks[stocks.stock == stock].sort_values("date").reset_index(drop=True)
        idx = pd.DatetimeIndex(s.date).searchsorted(peak.date, side="left")
        if idx == len(s):
            continue
        window = s.iloc[max(0, idx - 7):idx + 8]
        prior = s.iloc[max(0, idx - 20):idx]
        baseline = prior.volume.mean()
        rows.append({"stock": stock, "issue_peak_date": peak.date, "event_trade_date": s.iloc[idx].date,
                     "event_return_pct": s.iloc[idx].return_pct, "event_close": s.iloc[idx].close,
                     "issue_intensity": peak.issue_intensity, "return_std": window.return_pct.std(),
                     "volume_spike": s.iloc[idx].volume / baseline if len(prior) >= 5 and baseline > 0 else np.nan,
                     "baseline_trading_rows": len(prior), "event_window_rows": len(window)})
    events = pd.DataFrame(rows)
    if events[["return_std", "volume_spike"]].isna().any().any():
        raise ValueError("Incomplete event window or volume baseline")
    x = StandardScaler().fit_transform(events[["return_std", "volume_spike"]])
    events[["return_z", "volume_z"]] = x
    events["issue_sensitivity"] = x.sum(axis=1)
    diagnostics = []
    for k in range(2, min(6, len(events) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        diagnostics.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(x, model.labels_)})
    events["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(x)
    return events, pd.DataFrame(diagnostics)


def reaction_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stock, g in panel.groupby("stock", sort=True):
        valid = g.dropna(subset=["next_trade_date", "next_return_pct", "next_log_volume"])
        # One outcome per market day avoids repeating Monday for Friday/Sat/Sun.
        grouped = valid.groupby("next_trade_date", as_index=False).agg(
            issue_mean=("issue_intensity", "mean"), next_return_pct=("next_return_pct", "first"),
            next_log_volume=("next_log_volume", "first"), contributing_days=("date", "count"),
        )
        rows.append({"stock": stock, "reaction_dates": len(grouped),
                     "issue_next_return_corr": grouped.issue_mean.corr(grouped.next_return_pct),
                     "issue_next_volume_corr": grouped.issue_mean.corr(grouped.next_log_volume)})
    return pd.DataFrame(rows)


def make_figures(panel: pd.DataFrame, events: pd.DataFrame, features: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    colors = ["#2563eb", "#d97706", "#059669", "#dc2626"]
    g = panel[panel.stock == "삼성전자"].sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, layout="constrained")
    for column, label, color in zip(["search_index", "news_count", "retail_net_buy", "close"],
                                     ["Search index", "News count", "Retail net buy", "Close"], colors):
        x = g[column]
        axes[0].plot(g.date, (x - x.min()) / (x.max() - x.min()), label=label, color=color, lw=1.2)
    axes[0].set(title="Samsung Electronics: attention and market response", ylabel="Within-series min-max scale")
    axes[0].legend(ncol=4, loc="upper left")
    trade = g[g.is_trading_day]
    axes[1].plot(trade.date, trade.return_pct, color="#475569", lw=1.1, label="Daily return")
    axes[1].axhline(0, color="#94a3b8", lw=.8)
    axes[1].set(ylabel="Daily return (%)", xlabel="Date")
    marker = pd.Timestamp("2025-10-31")
    for ax in axes:
        ax.axvline(marker, color="#7c3aed", ls="--", lw=1, label="2025-10-31")
        ax.grid(alpha=.18)
    axes[1].legend(loc="upper left")
    fig.savefig(out / "samsung_timeline.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    for row in events.itertuples():
        ax.scatter(row.return_z, row.volume_z, color=colors[row.cluster], s=90)
        ax.annotate(DISPLAY_NAMES.get(row.stock, row.stock), (row.return_z, row.volume_z),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.margins(.22)
    ax.set(xlabel="Event return volatility (z-score)", ylabel="Event volume spike (z-score)",
           title="Event response clusters (K=3; descriptive choice)")
    ax.grid(alpha=.18)
    fig.savefig(out / "event_clusters.png", dpi=150)
    plt.close(fig)

    order = features.sort_values("FOMO_Index")
    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    ax.barh([DISPLAY_NAMES.get(s, s) for s in order.stock], order.FOMO_Index, color="#0f766e")
    ax.set(xlabel="Equal-weight min-max average (relative to these 10 stocks)", xlim=(0, 1),
           title="Exploratory FOMO proxy index: not a validated psychology measure")
    ax.grid(axis="x", alpha=.18)
    fig.savefig(out / "fomo_proxy_index.png", dpi=150)
    plt.close(fig)


def run(data_dir: Path, output_dir: Path) -> dict:
    stocks, attention = load_inputs(data_dir)
    panel = make_panel(stocks, attention)
    features = fomo_features(stocks, attention)
    pair_details, pair_summary = spike_pairs(stocks, attention)
    events, diagnostics = cluster_events(stocks, panel)
    correlations = reaction_correlations(panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"daily_panel": panel, "fomo_features": features, "spike_pair_details": pair_details,
              "spike_pair_summary": pair_summary, "event_responses": events,
              "cluster_diagnostics": diagnostics, "reaction_correlations": correlations}
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d", float_format="%.10g")
    make_figures(panel, events, features, output_dir)
    summary = {
        "period": [str(attention.date.min().date()), str(attention.date.max().date())],
        "stocks": sorted(stocks.stock.unique()), "stock_rows": len(stocks),
        "attention_rows": len(attention), "panel_rows": len(panel),
        "missing_next_reactions": int(panel.next_trade_date.isna().sum()),
        "pair_count_invariant": bool((pair_summary.search_events == pair_summary[["matched", "unmatched", "censored"]].sum(axis=1)).all()),
        "fomo_index_range": [float(features.FOMO_Index.min()), float(features.FOMO_Index.max())],
        "best_diagnostic_k": int(diagnostics.loc[diagnostics.silhouette.idxmax(), "k"]),
        "reported_cluster_k": 3,
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("*.csv"))},
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "scikit-learn"]},
        "limitations": ["Descriptive associations, not causal FOMO evidence or price prediction accuracy.",
                        "Calendar inferred from collected market dates, not independently certified exchange calendar.",
                        "Full-sample scaling, spike thresholds and selected lags are retrospective."]
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), ensure_ascii=True, indent=2))
