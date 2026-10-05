# Reader Input Schema

This document contains no market observations. The authors identify Bloomberg
as the upstream provider; raw vendor fields and export settings still need to
be recorded before a public data release. The schema describes the authors'
processed tables, not Bloomberg's native API or a guaranteed data entitlement.

## Daily Table

Expected filename: `data/daily_market.csv`, UTF-8 CSV without an index column.
One unique row per instrument and observed exchange session.

| Field | Type and interpretation |
| --- | --- |
| `date` | ISO date, New York trading session |
| `ticker` | AAPL, MSFT, AMZN, GOOGL, META, JPM, GS, XOM; optional reference SPX |
| `open`, `close`, `high`, `low` | Positive prices in consistent quoted units; the first/last retained quotes are not official auction executions |
| `volume` | Sum of retained bar volumes |
| `rv` | Square root of the sum of squared within-window consecutive log-close changes |
| `n_bars` | Count of retained source observations |
| `first_time`, `last_time` | First/last retained New York local times, `HH:MM` |

The reference window ends on 31 August 2026. Do not forward-fill missing sessions.
Ordinary full-session opens/closes or independently adjusted daily data are
legitimate alternatives, but they are not numerically identical to this extract.

## Calendar and Provenance

- `data/sessions.csv`: `date` and `last_bar_time`, one row per scheduled session,
  including early closes. It defines adjacency even when a quote is missing.
- `data/data_issues.csv`: `date`, `ticker`, `issue`; header-only is allowed when
  no issues are present. Missingness must not silently change a return horizon.
- `data/manifest.json`: provider and extraction information, requested and
  observed dates, instruments, transformations, adjustment status, missingness,
  redistribution status and SHA-256 of the daily CSV. Match the preparation
  module's schema for notebook 01 and the empirical tests.

An independently acquired dataset needs its own truthful metadata. Do not copy
the authors' sample counts or file hashes onto a replacement dataset. Some tests
verify the manuscript's exact extract and should be deliberately adapted, not
silenced, when undertaking a new experiment.

## Weekly Learning Table

Expected generated filename: `data/weekly_learning_panel.csv`.
The feature builder in `src/quantum_learning.py` produces:

- `ticker`, `week`, `end`: instrument, feature-week Monday, last feature date.
- `open`, `close`, `week_return`, `week_range`: feature-week prices and transformations.
- `volatility20`, `lag4`, `momentum4`: past-only volatility and four-calendar-week momentum inputs.
- `label_week`, `label_end`: next-calendar-week target and final outcome date.
- `forward_return`, `target`: next-week return and top-half cross-sectional label.

Features precede label outcomes. Only complete cohort weeks survive the stated
rules. Returns are decimal fractions, not percent numbers. Class labels are 0/1.
The weekly panel can be regenerated from authorised daily inputs and the calendar.

## Reading in Python

```python
from pathlib import Path
import pandas as pd

data = Path("data")
daily = pd.read_csv(data / "daily_market.csv", parse_dates=["date"])
assert not daily.duplicated(["date", "ticker"]).any()
weekly = pd.read_csv(
    data / "weekly_learning_panel.csv",
    parse_dates=["week", "end", "label_week", "label_end"],
)
assert (weekly["end"] < weekly["label_week"]).all()
```

These commands assume authorised files are already available locally. They do
not download Bloomberg data or grant permission to publish it.
