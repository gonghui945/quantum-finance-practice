"""Extract a small, frozen research panel without modifying the source archive.

The raw CSV timestamps follow the source project's Asia/Shanghai convention.
We convert them to New York time before assigning sessions. Prices are vendor
quotes, NOT verified dividend-adjusted total-return prices. Redistribution
rights must be checked separately before publishing these data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import TICKERS, SECTORS

ROOT = Path(__file__).resolve().parents[1]
START, END = "2024-01-01", "2026-08-31"
HOLIDAYS = set("2024-01-01 2024-01-15 2024-02-19 2024-03-29 2024-05-27 2024-06-19 2024-07-04 2024-09-02 2024-11-28 2024-12-25 2025-01-01 2025-01-09 2025-01-20 2025-02-17 2025-04-18 2025-05-26 2025-06-19 2025-07-04 2025-09-01 2025-11-27 2025-12-25 2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19 2026-07-03".split())
EARLY_CLOSE = set("2024-07-03 2024-11-29 2024-12-24 2025-07-03 2025-11-28 2025-12-24".split())


def sessions():
    """NYSE equity-session fixture, including the 2025 Carter closure."""
    return pd.DatetimeIndex([d for d in pd.bdate_range(START, END) if str(d.date()) not in HOLIDAYS])


def extract(source_root: Path) -> dict:
    """Read only selected symbols; never recursively enumerate the stock archive."""
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows, issues, source_hashes = [], [], []
    rename = {"时间": "timestamp", "开盘价": "open", "收盘价": "close",
              "最高价": "high", "最低价": "low", "成交量": "volume"}
    folders = sorted(p for p in (source_root / "stock").iterdir()
                     if p.is_dir() and p.name.endswith("_5min")
                     and START.replace("-", "") <= p.name[:8] <= END.replace("-", ""))
    for folder in folders:
        day = pd.Timestamp(folder.name[:8])
        if day not in sessions():
            continue
        for ticker in TICKERS + ["SPX"]:
            kind = "index" if ticker == "SPX" else "stock"
            path = source_root / kind / folder.name / f"{ticker}.csv"
            if not path.exists():
                issues.append({"date": str(day.date()), "ticker": ticker, "issue": "missing_file"})
                continue
            raw = path.read_bytes()
            source_hashes.append(f"{kind}/{folder.name}/{ticker}.csv {hashlib.sha256(raw).hexdigest()}")
            frame = pd.read_csv(path, encoding="utf-8-sig").rename(columns=rename)
            stamp = pd.to_datetime(frame["timestamp"], errors="raise", format="mixed")
            ny = stamp.dt.tz_localize("Asia/Shanghai").dt.tz_convert("America/New_York")
            minute = ny.dt.hour * 60 + ny.dt.minute
            # Respect 13:00 early closes and exclude the auction/end-labelled bar.
            end_minute = 775 if str(day.date()) in EARLY_CLOSE else 955
            keep = (ny.dt.date == day.date()) & minute.between(575, end_minute)
            frame = frame.loc[keep].copy()
            frame["ny"] = ny.loc[keep]
            frame = frame.sort_values("ny")
            if frame.empty:
                issues.append({"date": str(day.date()), "ticker": ticker, "issue": "no_session_bars"})
                continue
            if frame["ny"].duplicated().any():
                raise ValueError(f"Duplicate session timestamps: {ticker} {day.date()}")
            for col in ["open", "close", "high", "low", "volume"]:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
            assert (frame[["open", "close", "high", "low"]] > 0).all().all()
            assert (frame["high"] >= frame["low"]).all()
            log_moves = np.diff(np.log(frame["close"].to_numpy()))
            rows.append({"date": str(day.date()), "ticker": ticker,
                         "open": frame["open"].iloc[0], "close": frame["close"].iloc[-1],
                         "high": frame["high"].max(), "low": frame["low"].min(),
                         "volume": frame["volume"].sum(), "rv": np.sqrt(np.sum(log_moves**2)),
                         "n_bars": len(frame), "first_time": frame["ny"].iloc[0].strftime("%H:%M"),
                         "last_time": frame["ny"].iloc[-1].strftime("%H:%M")})
    data = pd.DataFrame(rows).sort_values(["date", "ticker"])
    assert not data.duplicated(["date", "ticker"]).any()
    assert data["date"].between(START, END).all()
    destination = out / "daily_market.csv"
    data.to_csv(destination, index=False)
    pd.DataFrame({"date": sessions(), "last_bar_time": ["12:55" if str(d.date()) in EARLY_CLOSE else "15:55" for d in sessions()]}).to_csv(out / "sessions.csv", index=False)
    missing_sessions = sorted(set(str(d.date()) for d in sessions())-set(data.date))
    pd.DataFrame(issues, columns=["date", "ticker", "issue"]).to_csv(out / "data_issues.csv", index=False)
    wide = data.pivot(index="date", columns="ticker", values="close")
    changes = wide.pct_change(fill_method=None)
    jumps = changes.stack()[changes.stack().abs() > .25]
    meta = {"requested_start": START, "requested_end": END,
            "observed_start": data.date.min(), "observed_end": data.date.max(),
            "tickers": TICKERS, "reference_index": "SPX", "rows": len(data),
            "stock_complete_sessions": int(wide[TICKERS].dropna().shape[0]),
            "expected_sessions": len(sessions()), "missing_whole_sessions": missing_sessions,
            "counts": data.groupby("ticker").size().to_dict(),
            "bar_count_distribution": data.groupby("n_bars").size().to_dict(),
            "missing_files_or_sessions": len(issues),
            "large_price_moves_over_25pct": {str(k): v for k, v in jumps.items()},
            "timezone": "Asia/Shanghai converted to America/New_York, following source convention",
            "window": "09:35--15:55; NYSE early closes use09:35--12:55; last available quote not official close",
            "calendar_source": "NYSE 2024/2025/2026 holiday calendars and January9 2025 mourning closure",
            "source": "Bloomberg, as identified by the authors; author-prepared five-minute archive",
            "source_attribution_confirmed_by_author": "2026-10-01",
            "vendor_export_independently_authenticated": False,
            "price_adjustment": "Unverified; all returns are quoted-price changes, not total returns",
            "universe": "Fixed eight-name US equity teaching cohort across five broad sectors; not random, market-representative or point-in-time constituent reconstruction",
            "sectors": SECTORS,
            "redistribution": "Author plans Python-readable daily/weekly CSV release; public redistribution permission remains unconfirmed",
            "daily_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_manifest_sha256": hashlib.sha256("\n".join(source_hashes).encode()).hexdigest()}
    (out / "source_hashes.txt").write_text("\n".join(source_hashes) + "\n")
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Archive directory containing stock/ and index/")
    args = parser.parse_args()
    print(json.dumps(extract(args.source_root), indent=2))
