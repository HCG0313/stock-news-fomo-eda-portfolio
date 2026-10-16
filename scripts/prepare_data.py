"""Export the two coursework input workbooks to documented, minimal CSV inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def prepare(source: Path, destination: Path) -> dict:
    stock_file = source / "주식통합.xlsx"
    attention_file = source / "뉴스, 검색 데이터.xlsx"
    stocks = pd.read_excel(stock_file)
    wide = pd.read_excel(attention_file)
    mapping = {
        "종목": "stock", "일자": "date", "종가": "close",
        "등락률": "return_pct", "거래량": "volume",
        "개인(매수)": "retail_buy", "개인(매도)": "retail_sell",
        "개인(순매수)": "retail_net_buy",
    }
    stocks = stocks[list(mapping)].rename(columns=mapping)
    stocks["date"] = pd.to_datetime(stocks["date"])
    wide["일자"] = pd.to_datetime(wide["일자"])
    attention = []
    for column in wide:
        if column.startswith("검색량_"):
            stock = column.split("_", 1)[1]
            attention.append(pd.DataFrame({
                "date": wide["일자"], "stock": stock,
                "search_index": wide[column],
                "news_count": wide[f"NEWS 건수_{stock}"],
            }))
    attention = pd.concat(attention, ignore_index=True)
    start, end = attention.date.min(), attention.date.max()
    stocks = stocks[stocks.date.between(start, end)].copy()
    destination.mkdir(parents=True, exist_ok=True)
    for frame, filename in [(stocks, "stock_daily.csv"),
                            (attention, "attention_daily.csv")]:
        frame.sort_values(["stock", "date"]).to_csv(
            destination / filename, index=False, date_format="%Y-%m-%d",
        )
    manifest = {
        "period": [str(start.date()), str(end.date())],
        "stock_rows": len(stocks), "attention_rows": len(attention),
        "stock_count": stocks.stock.nunique(),
        "source_files": [
            {"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in [stock_file, attention_file]
        ],
        "csv_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.glob("*.csv"))
        },
        "processing": "Selected columns, wide-to-long attention reshape, common date range; no synthetic observations.",
    }
    (destination / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), ensure_ascii=True))
