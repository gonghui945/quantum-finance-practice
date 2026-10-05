"""Conditional paired uncertainty for frozen stock-week predictions.

Resample whole cross-sections, never individual stock rows. Moving blocks are
drawn only from consecutive calendar weeks and never bridge missing periods.
These small-sample percentile intervals do not include model-selection,
refitting, universe-selection or unobserved-market-history uncertainty.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import RESULTS, save_json

SEED = 20261001
REPLICATES = 5000
BLOCK_LENGTHS = (1, 2, 4)
PRIMARY_BLOCK = 4


def calendar_blocks(weeks, length):
    """Return overlapping, non-wrapping blocks of consecutive observed Mondays."""
    dates = pd.DatetimeIndex(weeks)
    if length < 1 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Use unique chronological weeks and a positive block length.")
    blocks = [np.arange(start, start + length)
              for start in range(len(dates) - length + 1)
              if length == 1 or np.all(np.diff(dates[start:start + length].asi8)
                                      == pd.Timedelta(weeks=1).value)]
    if not blocks:
        raise ValueError("No uninterrupted calendar blocks of the requested length.")
    return np.stack(blocks)


def draw_week_counts(weeks, length, replicates=REPLICATES, seed=SEED):
    """Draw blocks with replacement and retain exactly the observed week count."""
    blocks = calendar_blocks(weeks, length)
    n = len(weeks)
    rng = np.random.default_rng(np.random.SeedSequence([seed, length]))
    picked = blocks[rng.integers(len(blocks), size=(replicates, (n + length - 1) // length))]
    picked = picked.reshape(replicates, -1)[:, :n]
    counts = np.zeros((replicates, n), dtype=int)
    np.add.at(counts, (np.arange(replicates)[:, None], picked), 1)
    return counts, blocks


def auc_pair_matrix(groups, method):
    """Wins between positive rows in week t and negative rows in week u.

Ties count one half. A resampled week's multiplicity weights both sides of
the AUC U-statistic, exactly as explicitly duplicating its stock rows would.
"""
    positive = [g.loc[g.target == 1, method].to_numpy() for g in groups]
    negative = [g.loc[g.target == 0, method].to_numpy() for g in groups]
    wins = np.array([[np.sum(a[:, None] > b[None, :])
                      + .5 * np.sum(a[:, None] == b[None, :])
                      for b in negative] for a in positive], dtype=float)
    return wins, np.array([len(a) for a in positive]), np.array([len(b) for b in negative])


def weighted_auc(counts, wins, positive, negative):
    """Evaluate many clustered bootstrap AUCs without expanding duplicate rows."""
    denominator = (counts @ positive) * (counts @ negative)
    if np.any(denominator == 0):
        raise ValueError("Every bootstrap sample must contain both classes.")
    return np.einsum("bi,ij,bj->b", counts, wins, counts, optimize=True) / denominator


def analyse(predictions, methods, replicates=REPLICATES, seed=SEED):
    frame = predictions.copy()
    frame["label_week"] = pd.to_datetime(frame.label_week)
    frame = frame.sort_values(["label_week", "ticker"])
    if frame.duplicated(["label_week", "ticker"]).any():
        raise ValueError("Duplicate stock-week observations.")
    if not np.isfinite(frame[methods + ["forward_return"]].to_numpy()).all():
        raise ValueError("Scores and outcomes must be finite.")
    if not set(frame.target.unique()).issubset({0, 1}):
        raise ValueError("Binary targets are required.")
    grouped = list(frame.groupby("label_week", sort=True))
    weeks = pd.DatetimeIndex([w for w, _ in grouped])
    groups = [g for _, g in grouped]
    cohort = set(groups[0].ticker)
    if any(set(g.ticker) != cohort or g.target.nunique() != 2 for g in groups):
        raise ValueError("Each resampling unit must be a complete, two-class cross-section.")
    pairs = {m: auc_pair_matrix(groups, m) for m in methods}
    ics = {m: np.array([spearmanr(g[m], g.forward_return).statistic for g in groups])
           for m in methods}
    if not all(np.isfinite(v).all() for v in ics.values()):
        raise ValueError("Undefined weekly rank IC; specify a policy before resampling.")
    point_counts = np.ones((1, len(weeks)), dtype=int)
    points = {m: float(weighted_auc(point_counts, *pairs[m])[0]) for m in methods}
    interval_rows, sensitivity, draw_arrays = [], [], {}
    for length in BLOCK_LENGTHS:
        counts, blocks = draw_week_counts(weeks, length, replicates, seed)
        draws = {m: weighted_auc(counts, *pairs[m]) for m in methods}
        delta_auc = draws["Reuploading kernel"] - draws["RBF SVC"]
        delta_ic = counts @ (ics["Reuploading kernel"] - ics["RBF SVC"]) / len(weeks)
        auc_interval = np.quantile(delta_auc, [.025, .975])
        ic_interval = np.quantile(delta_ic, [.025, .975])
        sensitivity.append(dict(block_weeks=length, admissible_blocks=len(blocks),
            delta_auc=points["Reuploading kernel"] - points["RBF SVC"],
            auc_low=float(auc_interval[0]), auc_high=float(auc_interval[1]),
            delta_ic=float(np.mean(ics["Reuploading kernel"] - ics["RBF SVC"])),
            ic_low=float(ic_interval[0]), ic_high=float(ic_interval[1])))
        for m in methods:
            low, high = np.quantile(draws[m], [.025, .975])
            interval_rows.append(dict(method=m, block_weeks=length, auc=points[m],
                                      auc_low=float(low), auc_high=float(high)))
        draw_arrays[f"L{length}_week_counts"] = counts
        draw_arrays[f"L{length}_auc"] = np.column_stack([draws[m] for m in methods])
        draw_arrays[f"L{length}_delta_ic"] = delta_ic
    seeds = [m for m in methods if m.startswith("VQC ")]
    seed_summary = {}
    for metric, values in {
        "auc": [points[m] for m in seeds],
        "mean_ic": [float(ics[m].mean()) for m in seeds],
    }.items():
        seed_summary[metric] = dict(mean=float(np.mean(values)),
                                    sample_sd=float(np.std(values, ddof=1)))
    report = dict(replicates=replicates, seed=seed, primary_block_weeks=PRIMARY_BLOCK,
        block_lengths=list(BLOCK_LENGTHS), observations=len(frame), weeks=len(weeks),
        securities_per_week=len(cohort), methods=methods,
        estimator="Pooled AUC; paired complete-week resampling; non-wrapping calendar-contiguous blocks",
        interval="2.5th and 97.5th bootstrap percentiles; conditional diagnostic",
        model_refitting=False, crosses_missing_calendar_weeks=False,
        caveat="Small non-stationary sample; block edges are not equally weighted; not a market-wide or selection-adjusted interval.",
        primary=next(row for row in sensitivity if row["block_weeks"] == PRIMARY_BLOCK),
        sensitivity=sensitivity, vqc_seeds=seeds, vqc_seed_summary=seed_summary)
    return report, pd.DataFrame(interval_rows), draw_arrays


def run():
    path = RESULTS / "qml_predictions.csv"
    predictions = pd.read_csv(path)
    metrics = pd.read_csv(RESULTS / "qml_metrics.csv")
    report, intervals, draws = analyse(predictions, metrics.method.tolist())
    report["predictions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Independent reconciliation uses the already saved point estimates.
    check = intervals[intervals.block_weeks == PRIMARY_BLOCK].set_index("method").auc
    np.testing.assert_allclose(check.loc[metrics.method], metrics.auc, atol=1e-12, rtol=0)
    intervals.to_csv(RESULTS / "qml_auc_intervals.csv", index=False)
    pd.DataFrame(report["sensitivity"]).to_csv(RESULTS / "qml_bootstrap_sensitivity.csv", index=False)
    np.savez_compressed(RESULTS / "qml_bootstrap_draws.npz", **draws)
    save_json("qml_uncertainty.json", report)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
