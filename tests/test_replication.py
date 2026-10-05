"""Independent small-domain checks and reconciliation of the saved evidence."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from qiskit.quantum_info import Statevector
from common import DATA,RESULTS,TICKERS,PORTFOLIO_K,market,close_panel
from portfolio import encode,qaoa_probabilities
from pricing_risk import amplitude_matrix
from quantum_learning import states,fidelity,product_state,product_kernel,two_axis_kernel,plot_results
from hardware_validation import qiskit_qaoa,noisy_kernel
from teaching_interfaces import (smoke_test, local_register_probabilities,
                                 bit_order_example, spectral_features)


def requires_files(*names):
    """Make unavailable empirical evidence explicit in the code-only release."""
    return unittest.skipUnless(all((ROOT / name).is_file() for name in names),
                               "Authorised inputs/saved results absent in code-only package")


class ReplicationTests(unittest.TestCase):
    @requires_files("results/qml_metrics.csv", "results/qml_weekly.csv", "results/qml_summary.json")
    def test_qml_figure_preserves_evidence(self):
        metrics = pd.read_csv(RESULTS / "qml_metrics.csv")
        weekly = pd.read_csv(RESULTS / "qml_weekly.csv")
        summary = json.loads((RESULTS / "qml_summary.json").read_text())
        originals = (metrics.copy(deep=True), weekly.copy(deep=True))
        with patch("matplotlib.figure.Figure.savefig"):
            fig = plot_results(metrics, weekly, summary["selected"])
        expected = metrics.loc[~metrics.method.str.startswith("VQC"), "auc"]
        np.testing.assert_allclose([bar.get_width() for bar in fig.axes[0].patches], expected)
        self.assertEqual(fig.axes[0].get_xlim(), (0, 1))
        self.assertIn("\n", fig.axes[0].get_yticklabels()[0].get_text())
        self.assertIn("\n", fig.axes[1].get_ylabel())
        names = metrics.loc[~metrics.method.str.startswith("VQC"), "method"].tolist()
        hatches = [bar.get_hatch() for bar in fig.axes[0].patches]
        self.assertEqual(hatches[names.index("Reuploading kernel")], "///")
        self.assertEqual(sum(bool(h) for h in hatches), 1)
        self.assertEqual([line.get_linestyle() for line in fig.axes[1].lines[:2]], ["--", "-"])
        self.assertEqual([line.get_marker() for line in fig.axes[1].lines[:2]], ["s", "o"])
        ic = weekly.pivot(index="week", columns="method", values="ic")
        ic.index = pd.to_datetime(ic.index)
        calendar = pd.date_range("2026-01-05", "2026-08-24", freq="W-MON")
        for line, name in zip(fig.axes[1].lines, ["RBF SVC", "Reuploading kernel"]):
            expected = ic[name].rolling(4).mean().reindex(calendar)
            np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)
        pd.testing.assert_frame_equal(metrics, originals[0])
        pd.testing.assert_frame_equal(weekly, originals[1])

    def test_local_register_and_bit_order(self):
        p = local_register_probabilities()
        self.assertEqual(p.shape, (256,))
        np.testing.assert_allclose(p, np.full(256, 1 / 256))
        result = bit_order_example()
        np.testing.assert_array_equal(result["PennyLane"], [0, 0, 1, 0])
        np.testing.assert_array_equal(result["Qiskit_raw"], [0, 1, 0, 0])

    def test_amplification_reflection_convention(self):
        p = np.array([.1, .2, .3, .4])
        values = np.array([0., .15, .5, 1.])
        a = amplitude_matrix(p, values)
        s0 = np.eye(8)
        s0[0, 0] = -1
        good = np.diag([1, -1] * 4)
        q = -a @ s0 @ a.conj().T @ good
        theta = np.arcsin(np.sqrt(p @ values))
        for power in [0, 1, 2, 4]:
            state = np.linalg.matrix_power(q, power) @ a[:, 0]
            self.assertAlmostEqual(np.sum(abs(state[1::2])**2),
                                   np.sin((2 * power + 1) * theta)**2, places=12)

    def test_spectral_training_and_test_map(self):
        matrix = np.array([[1., 1.1], [1.1, 1.]])
        fit, predicted = spectral_features(matrix, matrix)
        self.assertEqual(fit.shape, (2, 1))
        np.testing.assert_allclose(fit, predicted, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(fit @ fit.T).min(), -1e-12)

    def test_parameter_shift_convention(self):
        import pennylane as qml
        dev = qml.device("default.qubit", wires=1)
        @qml.qnode(dev)
        def f(angle):
            qml.RY(angle, wires=0)
            return qml.expval(qml.PauliZ(0))
        angle = .37
        shifted = (f(angle + np.pi / 2) - f(angle - np.pi / 2)) / 2
        self.assertAlmostEqual(float(shifted), -np.sin(angle), places=12)

    @requires_files("data/manifest.json", "data/daily_market.csv")
    def test_data_hash_and_dates(self):
        m=json.loads((DATA/"manifest.json").read_text())
        self.assertEqual(hashlib.sha256((DATA/"daily_market.csv").read_bytes()).hexdigest(),m["daily_sha256"])
        d=market();self.assertEqual(len(d),m['rows'])
        self.assertEqual(TICKERS,["AAPL","MSFT","AMZN","GOOGL","META","JPM","GS","XOM"])
        self.assertEqual(m["tickers"],TICKERS)
        self.assertEqual(PORTFOLIO_K,4)
        self.assertLessEqual(d.date.max(),pd.Timestamp("2026-08-31"))
        self.assertFalse(d.duplicated(["date","ticker"]).any())

    @requires_files("data/daily_market.csv")
    def test_early_close(self):
        d=market();early=d[d.date==pd.Timestamp("2024-07-03")]
        self.assertTrue((early.last_time<="12:55").all())

    @requires_files("data/daily_market.csv", "data/sessions.csv")
    def test_missing_session_is_not_bridged(self):
        r=close_panel().AAPL.pct_change(fill_method=None)
        self.assertTrue(pd.isna(r.loc["2026-03-27"]))
        self.assertTrue(pd.isna(r.loc["2026-03-30"]))

    @requires_files("data/weekly_learning_panel.csv")
    def test_label_horizons_and_balance(self):
        d=pd.read_csv(DATA/"weekly_learning_panel.csv",parse_dates=["week","end","label_week","label_end"])
        self.assertTrue((d.end<d.label_week).all())
        self.assertTrue(((d.label_week-d.week).dt.days==7).all())
        self.assertTrue((d.label_end<=pd.Timestamp("2026-08-31")).all())
        self.assertTrue((d.groupby("label_week").target.sum()==len(TICKERS)//2).all())

    def test_qubo_and_global_feasibility(self):
        n=len(TICKERS)
        rng=np.random.default_rng(10);m=rng.normal(size=(n,n));cov=m@m.T/100
        mu=rng.normal(size=n)/10
        bits,base,cost,h,j,penalty=encode(mu,cov)
        manual=np.array([5*(x/PORTFOLIO_K)@cov@(x/PORTFOLIO_K)-mu@(x/PORTFOLIO_K) for x in bits])
        np.testing.assert_allclose(base,manual,atol=1e-12)
        self.assertEqual(bits[np.argmin(cost)].sum(),PORTFOLIO_K)

    def test_sdk_equivalence(self):
        h=np.array([.1,.2,-.3]);j=np.triu(np.ones((3,3))*.15,1);theta=[.4,.7,.2,.1]
        q=Statevector.from_instruction(qiskit_qaoa(theta,h,j)).probabilities().reshape(2,2,2).transpose(2,1,0).ravel()
        np.testing.assert_allclose(q,qaoa_probabilities(theta,h,j),atol=1e-12)

    def test_payoff_probability_and_unitarity(self):
        p=np.array([.1,.2,.3,.4]);v=np.array([0.,.1,.5,1.]);a=amplitude_matrix(p,v)
        np.testing.assert_allclose(a.conj().T@a,np.eye(8),atol=1e-12)
        self.assertAlmostEqual(float(np.sum(abs(a[1::2,0])**2)),float(p@v),places=12)

    def test_analytical_kernels(self):
        x=np.random.default_rng(2).normal(size=(12,4))
        s=np.array([product_state(v) for v in x])
        np.testing.assert_allclose(fidelity(s,s),product_kernel(x,x),atol=1e-12)
        s=states(x,layers=1)
        np.testing.assert_allclose(fidelity(s,s),two_axis_kernel(x,x),atol=1e-12)

    def test_noise_reduction_identity(self):
        a=np.array([.1,.2,.3,.4]);b=-a
        self.assertAlmostEqual(noisy_kernel(a,b,2,0.,False),noisy_kernel(a,b,2,0.,True),places=12)

    @requires_files("results/qml_predictions.csv", "results/qml_metrics.csv", "results/qml_summary.json")
    def test_predictive_metrics(self):
        pred=pd.read_csv(RESULTS/"qml_predictions.csv")
        metrics=pd.read_csv(RESULTS/"qml_metrics.csv")
        for row in metrics.itertuples(): self.assertAlmostEqual(roc_auc_score(pred.target,pred[row.method]),row.auc,places=12)
        summary=json.loads((RESULTS/'qml_summary.json').read_text())
        self.assertEqual(len(pred),summary['test'])
        self.assertEqual(len(pred),len(TICKERS)*pred.label_week.nunique())

    @requires_files("results/portfolio_runs.csv", "data/daily_market.csv", "data/sessions.csv")
    def test_portfolio_objective_reconciliation(self):
        runs=pd.read_csv(RESULTS/"portfolio_runs.csv",parse_dates=["start"])
        r=close_panel()[TICKERS].pct_change(fill_method=None).dropna()
        for row in runs.itertuples():
            hist=r[r.index<row.start].tail(126)
            x=np.array([t in row.holdings.split(",") for t in TICKERS],float)
            w=x/PORTFOLIO_K;cost=5*w@(252*hist.cov().to_numpy())@w-(252*hist.mean().to_numpy())@w
            self.assertAlmostEqual(cost,row.objective,places=10)

    @requires_files("results/pricing_risk_summary.json", "results/pricing_repeated.csv")
    def test_saved_prices_and_tail_units(self):
        s=json.loads((RESULTS/"pricing_risk_summary.json").read_text())
        repeated=pd.read_csv(RESULTS/"pricing_repeated.csv")
        rmse=np.sqrt(np.mean((repeated.mlae-repeated.target)**2))
        self.assertAlmostEqual(rmse,s["mlae_rmse"],places=10)
        risk=s["risk"];self.assertGreater(risk["grid_es"],risk["grid_var"])
        self.assertEqual(s["preparation_calls"],4500)

    def test_printed_interfaces(self):
        self.assertIn("successfully", smoke_test())

    @requires_files("results/teaching_summary.json", "results/asian_two_date_grid.csv")
    def test_asian_lookup_matches_discrete_paths(self):
        output=json.loads((RESULTS/"teaching_summary.json").read_text())["asian"]
        paths=pd.read_csv(RESULTS/"asian_two_date_grid.csv")
        self.assertAlmostEqual(paths.probability.sum(),1.,places=12)
        price=np.exp(-.03*.25)*(paths.probability@paths.payoff)
        self.assertAlmostEqual(price,output["grid_price"],places=12)
        self.assertAlmostEqual(price,output["exact_ancilla_price"],places=12)
        self.assertEqual(output["monitoring_dates"],2)
        self.assertEqual(output["total_qubits"],5)

    @requires_files("results/teaching_summary.json")
    def test_factor_and_pqc_accounting(self):
        output=json.loads((RESULTS/"teaching_summary.json").read_text())
        factors=output["factors"]
        self.assertEqual(factors["retained_factors"],len(TICKERS))
        baseline,stress=factors["scenarios"][1:]
        self.assertGreater(stress["var"],baseline["var"])
        pqc=output["pqc"]
        self.assertEqual(pqc["kem_key_and_ciphertext"]+pqc["dsa_key_and_signature"],7533)
        self.assertEqual(pqc["one_million_signatures_bytes"],3309000000)

    @requires_files("results/qml_uncertainty.json", "results/qml_bootstrap_draws.npz",
                    "results/qml_predictions.csv", "results/qml_auc_intervals.csv")
    def test_saved_cluster_intervals(self):
        report = json.loads((RESULTS / "qml_uncertainty.json").read_text())
        pred = pd.read_csv(RESULTS / "qml_predictions.csv").sort_values(["label_week", "ticker"])
        self.assertEqual(hashlib.sha256((RESULTS / "qml_predictions.csv").read_bytes()).hexdigest(),
                         report["predictions_sha256"])
        groups = list(pred.groupby("label_week", sort=True))
        intervals = pd.read_csv(RESULTS / "qml_auc_intervals.csv")
        with np.load(RESULTS / "qml_bootstrap_draws.npz") as draws:
            for length in report["block_lengths"]:
                weights = draws[f"L{length}_week_counts"]
                auc = draws[f"L{length}_auc"]
                # Independently expand sampled stock rows for representative draws.
                for b in [0, 1, len(weights) - 1]:
                    expanded = pd.concat([g for (_, g), n in zip(groups, weights[b]) for _ in range(n)])
                    for j, method in enumerate(report["methods"]):
                        self.assertAlmostEqual(auc[b, j], roc_auc_score(expanded.target, expanded[method]), places=13)
                for j, method in enumerate(report["methods"]):
                    row = intervals[(intervals.method == method) & (intervals.block_weeks == length)].iloc[0]
                    np.testing.assert_allclose(np.quantile(auc[:, j], [.025, .975]),
                                               [row.auc_low, row.auc_high], atol=1e-14)


if __name__=="__main__": unittest.main(verbosity=2)
