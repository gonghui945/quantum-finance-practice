# Reconstructing the Worked Examples

This is a transformation and execution contract, not a Bloomberg downloader or
a licence to redistribute data. The public route is code, tests, deterministic
synthetic fixtures and schemas. Empirical reproduction is conditional on
authorised access to the same processed extract.

## 1. Choose the Reproduction Claim

| Route | Inputs | What a successful run establishes |
| --- | --- | --- |
| Implementation checks | Synthetic arrays embedded in `teaching_interfaces.py` and the tests | Local mappings and numerical identities, without market observations |
| Exact empirical reproduction | The same authorised daily extract, calendar and provenance manifest | Reproduction of the stated historical calculations in the pinned environment |
| Adapted experiment | Independently obtained, documented inputs with the same schema | A new experiment, not a claim that the manuscript's numerical results match |

The author identifies Bloomberg as the upstream provider. A subscription alone
does not identify the original export: instrument identifiers, source fields,
corporate-action adjustments, bar alignment and timestamp conventions must match
as well. Those vendor-level settings have not all been independently verified.
Consequently, this package does not promise exact reconstruction from an arbitrary
Bloomberg query. Request the extract or missing export specification through an
authorised route; do not infer settings from familiar ticker names.

## 2. Run Without Market Data

After installing `requirements-pinned.txt`, run from the package root:

```bash
python src/teaching_interfaces.py
python -m unittest discover -s tests -v
```

The first command exercises eight printed interfaces using hypothetical arrays.
The tests also check a synthetic weekly resampling example. These fixtures are
constructed independently of market observations; they are not anonymised or
perturbed Bloomberg records. In the code-only package 14 tests run and 11
market/result-dependent checks explicitly skip. No statistical or investment
conclusion should be drawn from these synthetic checks.

## 3. Prepare Authorised Daily Inputs

Place the following files in `data/`, using [DATA_SCHEMA.md](DATA_SCHEMA.md):

- `daily_market.csv`: one unique instrument-session row, with daily prices,
  volume, within-window realised variation and coverage fields.
- `sessions.csv`: the scheduled session grid and last eligible bar time,
  including early closes and the 9 January 2025 mourning closure.
- `data_issues.csv`: explicitly identified missing files or empty session windows.
- `manifest.json`: truthful provenance, transformations, counts, adjustment status
  and the SHA-256 of that daily CSV.

For the manuscript extract, request dates from 1 January 2024 to 31 August 2026;
the first observed session is 2 January 2024. The fixed cohort is AAPL, MSFT,
AMZN, GOOGL, META, JPM, GS and XOM. SPX is a descriptive reference, not a feature
or a ninth portfolio decision variable. Do not include September observations.

### Original archive adapter

`prepare_data.py` is specific to the authors' prepared archive. Its layout is
`stock/YYYYMMDD_5min/TICKER.csv`, with the analogous `index/` directory for SPX.
Its column-translation map identifies the original CSV headers. It is not
Bloomberg's native export schema. For an authorised copy of that archive:

```bash
python src/prepare_data.py --source-root ../authorised_archive
```

The implemented transformation is:

1. Interpret the archive timestamps as Asia/Shanghai local time, then convert
   to America/New_York with daylight-saving rules. This is an archive convention,
   not a general claim about Bloomberg timestamps.
2. Retain 09:35-15:55 New York time, inclusive, or 09:35-12:55 on the specified
   early-close sessions. Reject duplicate timestamps within an instrument-session.
3. Use the first retained open, last retained close, highest high, lowest low
   and sum of retained bar volumes. These are window proxies, not auction prices.
4. Compute `rv` as the square root of the sum of squared consecutive log-close
   changes within the retained window. Record bar count and first/last times.
5. Sort by date and ticker, write the daily/calendar/issue tables and record
   hashes. The original source-file hash list stays local, outside the release.

Missing source bars are not fabricated or filled. Partial coverage may change a
price proxy. The adapter does not infer corporate-action adjustments. A new
source needs its own timezone, field adapter and truthful metadata; do not reuse
the original archive's attribution or sample counts unchanged.

### From daily inputs to weekly features

The code reindexes daily closes to scheduled sessions before computing returns,
so a missing close does not silently create a multi-session return. The weekly
builder in `quantum_learning.py` then:

1. Uses Monday-labelled calendar weeks and requires the scheduled daily rows.
2. Forms weekly open-to-close return, range, trailing 20-session annualised
   volatility and four-calendar-week close-to-close momentum.
3. Joins the next calendar week's return as the outcome, rather than shifting
   to the next available observation across a gap.
4. Assigns the top-four-of-eight label within each complete target week and
   retains complete eight-stock feature/target cross-sections.
5. Excludes an incomplete final week and writes `weekly_learning_panel.csv`.

Model selection uses the 2024/2025 target-week partitions. Final models refit
the latest 52 eligible pre-2026 weeks before evaluating the held-out 2026 rows.
The source records the exact preprocessing, circuit parameters and seeds.
Changing the cohort, window or adjustment basis creates a different experiment.

## 4. Execute and Reconcile

With authorised inputs and the pinned environment:

```bash
python src/run_cases.py --case all
python -m unittest discover -s tests -v
```

The driver runs portfolio, pricing/risk extensions, learning and hardware-aware
simulation in dependency order. Outputs are generated in `results/`; the weekly
panel is regenerated in `data/`. No order or physical quantum job is submitted.
For the notebook presentation instead, use:

```bash
python src/build_notebooks.py --execute --kernel quantum-finance
```

Register that kernel as described in the README. The five notebooks are another
interface to the same functions, not a separate implementation.

Check input identity before comparing numerical results. Record the source-code
revision, environment, data hash and configuration. Point estimates and seeded
draws can be reconciled in the tested environment; wall-clock measurements vary.
Some tests deliberately assert properties of the authors' exact extract. An
adapted dataset needs separately documented expected properties, not silent
removal of failing checks or copying the original manifest onto new data.

To refresh only uncertainty or the existing figure from saved predictions:

```bash
python src/qml_uncertainty.py
python src/quantum_learning.py --plot-only
python src/build_notebooks.py --refresh-learning-figure
```

The last command changes the stored Figure 3.1 image and HTML preview only;
it preserves execution counts and fitted-model outputs and records an artifact
refresh in notebook metadata. It requires an already executed local notebook.

## 5. Keep Access and Publication Separate

The temporary entry point is [Hui Gong's GitHub profile](https://github.com/gonghui945),
not an assertion that this revision is deposited there. A repository-specific
release, immutable identifier and exact code-licence text will replace the
placeholder when supplied. The author reports co-author confirmation; no
particular open-source licence has been inferred from that statement.

Only the code-only candidate is prepared for the public route. The local author
package retains empirical inputs and executed outputs but is not cleared for
public redistribution. See [DATA_AND_PUBLICATION.md](DATA_AND_PUBLICATION.md).
