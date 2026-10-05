"""Run synthetic interfaces without external market data.

Inputs are hypothetical. Fixed-angle QAOA illustrates decoding and is not an
optimised investment strategy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio import encode, qaoa_probabilities
from quantum_learning import fidelity, states, two_axis_kernel
from teaching_interfaces import (
    bit_order_example, kernel_scores, local_register_probabilities,
    payoff_expectation, security_size_example, spectral_features,
)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "examples" / "synthetic"


def inputs():
    """Load declared assumptions rather than empirical estimates."""
    return (
        pd.read_csv(SYNTHETIC / "assets.csv"),
        pd.read_csv(SYNTHETIC / "payoffs.csv"),
        pd.read_csv(SYNTHETIC / "features.csv"),
    )


def register_example():
    probabilities = local_register_probabilities(n=8)
    ordering = bit_order_example()
    return {
        "wires": 8,
        "basis_states": len(probabilities),
        "probability_sum": float(probabilities.sum()),
        "PennyLane_order": ordering["PennyLane"].tolist(),
        "Qiskit_raw_order": ordering["Qiskit_raw"].tolist(),
    }


def pricing_example(payoffs, rate=0.03, maturity=0.25):
    """Recover the expectation before applying the discount factor once."""
    probability = payoffs["probability"].to_numpy()
    amounts = payoffs["payoff"].to_numpy()
    np.testing.assert_allclose(probability.sum(), 1.0)
    encoded = payoff_expectation(probability, amounts)
    classical = float(probability @ amounts)
    np.testing.assert_allclose(encoded, classical)
    return {
        "classical_expectation": classical,
        "encoded_expectation": float(encoded),
        "discounted_value": float(np.exp(-rate * maturity) * encoded),
    }


def portfolio_example(assets, shots=256, seed=11):
    """Compare sampled decoding with the exact feasible reference."""
    mu = assets["annual_mean"].to_numpy()
    covariance = np.diag(assets["annual_variance"].to_numpy())
    bits, base, cost, h, j, penalty = encode(mu, covariance, k=4)
    probabilities = qaoa_probabilities(np.tile([0.2, 0.1], 2), h, j)
    feasible = bits.sum(axis=1) == 4
    exact = int(np.flatnonzero(feasible)[np.argmin(base[feasible])])
    counts = np.random.default_rng(seed).multinomial(
        shots, probabilities / probabilities.sum()
    )
    observed = np.flatnonzero(feasible & (counts > 0))
    selected = int(observed[np.argmin(base[observed])]) if len(observed) else None
    names = assets["asset"].to_numpy()
    return {
        "feasible_baskets": int(feasible.sum()),
        "penalty": float(penalty),
        "shots": shots,
        "seed": seed,
        "feasible_probability": float(probabilities[feasible].sum()),
        "exact_holdings": names[bits[exact].astype(bool)].tolist(),
        "sampled_holdings": (
            names[bits[selected].astype(bool)].tolist() if selected is not None else []
        ),
        "sampled_objective_gap": (
            float(base[selected] - base[exact]) if selected is not None else None
        ),
        "penalised_expectation": float(probabilities @ cost),
        "interpretation": "Fixed angles; no parameter optimisation.",
    }


def kernel_example(features):
    """Preserve training order in both the fit and prediction matrices."""
    x = features[[f"angle_{j}" for j in range(4)]].to_numpy()
    labels = features["label"].to_numpy(dtype=int)
    fit_states, test_states = states(x[:8], layers=1), states(x[8:], layers=1)
    train = fidelity(fit_states, fit_states)
    cross = fidelity(test_states, fit_states)
    analytical = two_axis_kernel(x[:8], x[:8])
    np.testing.assert_allclose(train, analytical, atol=1e-12)
    return {
        "train_kernel": train,
        "cross_kernel": cross,
        "decision_scores": kernel_scores(x[:8], labels[:8], x[8:]),
        "closed_form_max_error": float(np.max(np.abs(train - analytical))),
    }


def measurement_example(train, cross, shots=1000, seed=11):
    """Sample all-zero events, then define both spectral feature maps."""
    rng = np.random.default_rng(seed)
    sampled_train = rng.binomial(shots, np.clip(train, 0, 1)) / shots
    sampled_cross = rng.binomial(shots, np.clip(cross, 0, 1)) / shots
    fit, test = spectral_features(sampled_train, sampled_cross)
    assert fit.shape[1] == test.shape[1]
    return {
        "shots_per_pair": shots,
        "train_entry_rmse": float(np.sqrt(np.mean((sampled_train - train) ** 2))),
        "feature_dimensions": fit.shape[1],
        "train_map_shape": list(fit.shape),
        "test_map_shape": list(test.shape),
    }


def run():
    assets, payoffs, features = inputs()
    kernel = kernel_example(features)
    return {
        "inputs": "Synthetic assumptions; no empirical market data",
        "register": register_example(),
        "pricing": pricing_example(payoffs),
        "portfolio": portfolio_example(assets),
        "kernel": {
            "train_shape": list(kernel["train_kernel"].shape),
            "cross_shape": list(kernel["cross_kernel"].shape),
            "closed_form_max_error": kernel["closed_form_max_error"],
            "decision_scores": kernel["decision_scores"].tolist(),
        },
        "measurement": measurement_example(
            kernel["train_kernel"], kernel["cross_kernel"]
        ),
        "pqc_sizes": security_size_example(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
