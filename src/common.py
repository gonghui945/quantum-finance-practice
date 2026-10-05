"""Shared paths, plotting and serialisation for the standalone experiments."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantum-elements-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "GS", "XOM"]
# A fixed teaching cohort across five sectors, not a representative market sample.
SECTORS = dict(zip(TICKERS, ["Information technology", "Information technology",
    "Consumer discretionary", "Communication services", "Communication services",
    "Financials", "Financials", "Energy"]))
PORTFOLIO_K = 4
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=["#174a68", "#ba6035", "#357b64", "#986c36"]),
                     "figure.figsize": (7.0, 3.6), "savefig.bbox": "tight"})


def market():
    frame = pd.read_csv(DATA / "daily_market.csv", parse_dates=["date"])
    assert frame.date.min() >= pd.Timestamp("2024-01-01")
    assert frame.date.max() <= pd.Timestamp("2026-08-31")
    assert not frame.duplicated(["date", "ticker"]).any()
    return frame


def close_panel():
    """Missing sessions remain missing so daily returns never bridge a gap."""
    days = pd.read_csv(DATA / "sessions.csv", parse_dates=["date"])["date"]
    return market().pivot(index="date", columns="ticker", values="close").reindex(days)


def save_json(name, content):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, pd.Timestamp): return str(x.date())
        raise TypeError(type(x).__name__)
    (RESULTS / name).write_text(json.dumps(content, indent=2, default=convert) + "\n")


def save_figure(name):
    plt.tight_layout()
    plt.savefig(RESULTS / f"{name}.pdf")
    plt.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.close()
