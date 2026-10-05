"""Small worked examples supporting the tutorial text.

Asian paths and factor scenarios are generated from stated models. The quantum
Asian example encodes a supplied 16-path grid; it is not a scalable arithmetic
oracle. PQC sizes below are standard object sizes, not measured cryptography.
"""
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import norm
from common import TICKERS, SECTORS, DATA, RESULTS, close_panel, save_json
from pricing_risk import amplitude_matrix


def asian_path_grid(sigma, s0=100., strike=100., rate=.03, tenor=.25):
    """Encode two monitoring dates with two qubits per Gaussian increment."""
    edges = np.linspace(-4., 4., 5)
    z = (edges[:-1] + edges[1:]) / 2
    marginal = np.diff(norm.cdf(edges))
    marginal /= marginal.sum()
    indices = np.array(list(product(range(4), repeat=2)))
    probability = np.prod(marginal[indices], axis=1)
    dt = tenor / 2
    paths = s0 * np.exp(np.cumsum((rate - sigma**2 / 2)*dt
                                  + sigma*np.sqrt(dt)*z[indices], axis=1))
    payoff = np.maximum(paths.mean(axis=1)-strike, 0.)
    cap = payoff.max()
    unitary = amplitude_matrix(probability, payoff/cap)
    amplitude = float(np.sum(np.abs(unitary[1::2, 0])**2))
    target = float(np.exp(-rate*tenor)*probability@payoff)
    assert abs(np.exp(-rate*tenor)*cap*amplitude-target) < 1e-10
    # Independent Monte Carlo estimates a continuous two-date target.
    rng = np.random.default_rng(701)
    shocks = rng.normal(size=(100000, 2))
    mc_paths = s0*np.exp(np.cumsum((rate-sigma**2/2)*dt
                                  + sigma*np.sqrt(dt)*shocks, axis=1))
    discounted = np.exp(-rate*tenor)*np.maximum(mc_paths.mean(axis=1)-strike, 0.)
    shots = 4500
    observed = rng.binomial(shots, amplitude)/shots
    pd.DataFrame({"increment_1": indices[:, 0], "increment_2": indices[:, 1],
                  "probability": probability, "first_price": paths[:, 0],
                  "second_price": paths[:, 1], "payoff": payoff}).to_csv(
                      RESULTS/"asian_two_date_grid.csv", index=False)
    return dict(monitoring_dates=2, paths=16, index_qubits=4, total_qubits=5,
                grid_price=target, exact_ancilla_price=float(np.exp(-rate*tenor)*cap*amplitude),
                direct_shots=shots, direct_price=float(np.exp(-rate*tenor)*cap*observed),
                continuous_mc=float(discounted.mean()),
                continuous_mc_se=float(discounted.std(ddof=1)/np.sqrt(len(discounted))),
                probability_identity_error=float(abs(amplitude-probability@(payoff/cap))))


def factor_scenarios():
    """Illustrate dependence and deterministic factor stress on a daily scale."""
    returns = close_panel()[TICKERS].pct_change(fill_method=None).dropna().tail(252)
    mu = returns.mean().to_numpy()
    covariance = returns.cov().to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    weights = np.full(len(TICKERS), 1/len(TICKERS))
    # Fix eigenvector signs so a negative leading-factor shift reduces the
    # equally weighted basket. PCA eigenvector signs have no intrinsic meaning.
    for j in range(len(eigenvalues)):
        if weights@eigenvectors[:, j] < 0:
            eigenvectors[:, j] *= -1
    rng = np.random.default_rng(202609)
    shocks = rng.normal(size=(100000, len(TICKERS)))
    root = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))
    base = mu + shocks@root.T
    stress_shocks = shocks.copy()
    stress_shocks[:, 0] -= 2.
    stress = mu + stress_shocks@root.T
    independent = mu + shocks*np.sqrt(np.diag(covariance))
    records = []
    for name, scenarios in [("Independent marginal control", independent),
                             ("Correlated Gaussian", base),
                             ("Leading-factor stress", stress)]:
        loss = -scenarios@weights
        q = np.quantile(loss, .95, method="inverted_cdf")
        records.append(dict(scenario=name, var=float(q),
            es=float(q+np.mean(np.maximum(loss-q, 0))/.05), mean_loss=float(loss.mean())))
    pd.DataFrame(records).to_csv(RESULTS/"factor_scenarios.csv", index=False)
    return dict(observations=len(returns), generated_scenarios=100000,
                retained_factors=len(TICKERS), explained_first=float(eigenvalues[0]/eigenvalues.sum()),
                stressed_factor_shift=-2., scenarios=records)


def data_description():
    """Descriptive moments only; quoted-price returns exclude verified dividends."""
    panel = close_panel()[TICKERS]
    returns = panel.pct_change(fill_method=None)
    rows = []
    for ticker in TICKERS:
        series = returns[ticker].dropna()
        rows.append(dict(ticker=ticker, sector=SECTORS[ticker], prices=int(panel[ticker].notna().sum()),
            returns=len(series), mean_daily_pct=float(100*series.mean()),
            annual_vol_pct=float(100*np.sqrt(252)*series.std()),
            min_daily_pct=float(100*series.min()), max_daily_pct=float(100*series.max())))
    pd.DataFrame(rows).to_csv(RESULTS/"data_description.csv", index=False)
    return rows


def pqc_object_sizes():
    """FIPS 203/204/205 object lengths, excluding protocol overhead."""
    rows = [dict(algorithm="ML-KEM-768", public_key=1184, transmitted_object=1088,
                 object_type="Ciphertext"),
            dict(algorithm="ML-DSA-65", public_key=1952, transmitted_object=3309,
                 object_type="Signature"),
            dict(algorithm="SLH-DSA-SHA2-128s", public_key=32, transmitted_object=7856,
                 object_type="Signature")]
    pd.DataFrame(rows).to_csv(RESULTS/"pqc_object_sizes.csv", index=False)
    return dict(objects=rows, kem_key_and_ciphertext=1184+1088,
                dsa_key_and_signature=1952+3309,
                one_million_signatures_bytes=1000000*3309)


def run():
    import json
    pricing = json.loads((RESULTS/"pricing_risk_summary.json").read_text())
    output = dict(asian=asian_path_grid(pricing["sigma"]), factors=factor_scenarios(),
                  data=data_description(), pqc=pqc_object_sizes())
    save_json("teaching_summary.json", output)
    return output


if __name__ == "__main__":
    print(run())
