"""Build portable, commented tutorial interfaces and optionally execute them.

Notebook outputs are generated artefacts. Financial logic remains in the named
modules so a notebook is never the only copy of an algorithm.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=ROOT/"notebooks"
SETUP='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image

# Resolve only the package itself; no author-specific absolute path is needed.
ROOT = Path.cwd().resolve()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "data" / "daily_market.csv").exists(), "Authorised daily inputs are required; see DATA_SCHEMA.md. Synthetic checks run without market data via src/teaching_interfaces.py."
sys.path.insert(0, str(ROOT / "src"))
from common import DATA, RESULTS, market
'''


def notebook(title,intro,steps,takeaways):
    nb=nbf.v4.new_notebook()
    nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                 "language_info":{"name":"python","version":"3.12.7"}}
    nb.cells=[nbf.v4.new_markdown_cell(f"# {title}\n\n## Goal\n\n{intro}"),
              nbf.v4.new_markdown_cell("## Setup\n\nRun this notebook from top to bottom. The experiment uses local simulation only; no cloud account, trading connection or API key is needed. Market observations stop on **31 August 2026**. The supplied data are quoted-price proxies, not verified total-return series."),
              nbf.v4.new_code_cell(SETUP)]
    for heading,explanation,code in steps:
        nb.cells.extend([nbf.v4.new_markdown_cell(f"## {heading}\n\n{explanation}"),nbf.v4.new_code_cell(code)])
    nb.cells.append(nbf.v4.new_markdown_cell("## Interpretation and Next Steps\n\n"+takeaways))
    return nb


def refresh_learning_figure():
    """Refresh only the stored Figure 3.1 image from frozen numerical results.

    Existing execution counts and fitted-model outputs are preserved. This is
    an artifact refresh, not evidence that the model-fitting cells were rerun.
    """
    path = NOTEBOOKS / "04_quantum_learning.ipynb"
    image = ROOT / "results" / "qml_comparison.png"
    nb = nbf.read(path, as_version=4)
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    changed = 0
    for cell in nb.cells:
        if cell.cell_type != "code" or "qml_comparison.png" not in cell.source:
            continue
        for output in cell.outputs:
            if "image/png" in output.get("data", {}):
                output.data["image/png"] = encoded
                changed += 1
    if changed != 1:
        raise ValueError(f"Expected one Figure 3.1 image, found {changed}")
    nb.metadata["figure_refresh"] = {
        "artifact": image.name,
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "scope": "Grayscale-safe plot from saved results; no model-fitting cell rerun",
    }
    nbf.validate(nb)
    nbf.write(nb, path)
    html, _ = HTMLExporter().from_notebook_node(nb)
    path.with_suffix(".html").write_text(html)
    print("Refreshed Figure 3.1 only; retained all numerical notebook outputs.")


def build(execute=False, only=None, kernel_name="python3"):
    NOTEBOOKS.mkdir(exist_ok=True)
    specs={
      "01_environment":notebook("Environment and Calendar-Aware Market Data",
        "Companion to Chapter 1. Establish what the data and execution environment represent before building a circuit. The fixed eight-stock cohort is AAPL, MSFT, AMZN, GOOGL, META, JPM, GS and XOM, spanning five broad sectors. It is not random, market-representative or a historical constituent reconstruction. Eight decision bits make all 256 basis strings inspectable; selecting four gives 70 feasible baskets. SPX is a descriptive price-index reference.",[
        ("1. Inspect the data contract","The manifest records Bloomberg as the author-identified provider, distinguishes requested dates from observed sessions, identifies missing files and states unresolved adjustment and redistribution questions. File hashes check identity, not economic authenticity.",
         'manifest = json.loads((DATA / "manifest.json").read_text())\ndisplay(pd.Series({k: manifest[k] for k in ["source", "observed_start", "observed_end", "rows", "expected_sessions", "missing_files_or_sessions", "price_adjustment", "redistribution"]}, name="Data contract").to_frame())'),
        ("2. Check chronology and coverage","Times were converted from the source archive's Shanghai convention to New York, including daylight saving. The window is 09:35--15:55, shortened to 12:55 on exchange-designated early closes. Missing observations are not filled.",
         'daily = market()\nassert daily.date.max() <= pd.Timestamp("2026-08-31")\nassert not daily.duplicated(["date", "ticker"]).any()\ndisplay(daily.groupby("ticker").agg(sessions=("date", "size"), first=("date", "min"), last=("date", "max")))\ndisplay(pd.read_csv(DATA / "data_issues.csv"))'),
        ("3. Verify a minimal circuit","PennyLane and Qiskit represent the same qubit mathematics. Here a Bell state checks preparation and probabilities. Exact simulator amplitudes are not an output that a physical QPU supplies directly.",
         'import pennylane as qml\nfrom qiskit import QuantumCircuit\nfrom qiskit.quantum_info import Statevector\ndev = qml.device("default.qubit", wires=2)\n@qml.qnode(dev)\ndef bell():\n    qml.Hadamard(wires=0)\n    qml.CNOT(wires=[0, 1])\n    return qml.probs(wires=[0, 1])\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\npl = np.asarray(bell())\nqk = Statevector.from_instruction(qc).probabilities()\nnp.testing.assert_allclose(pl, qk, atol=1e-12)\ndisplay(pd.DataFrame({"basis": ["00", "01", "10", "11"], "PennyLane": pl, "Qiskit": qk}))'),
        ("4. Allocate a local register and verify bit order", "Eight simulated wires store 256 complex amplitudes locally; Hadamards give 256 equal probabilities. An asymmetric two-qubit X test then makes the SDK ordering difference visible, unlike the reversal-symmetric Bell state. No hardware account is contacted.",
         'from teaching_interfaces import local_register_probabilities, bit_order_example\np = local_register_probabilities()\nprint("Eight-wire probability vector:", p.shape, "probability sum:", p.sum())\ndisplay(pd.DataFrame(bit_order_example()))'),
        ("5. Record software versions","Use the pinned environment when reproducing numerical outputs. Later SDK releases can change interfaces and compilation, even when the financial equations are unchanged.",
         'import importlib.metadata as metadata\nversions = {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "pennylane", "qiskit"]}\ndisplay(pd.Series(versions, name="Installed version").to_frame())')],
        "Read the displayed manifest for the executed sample counts. Missing quotes are not filled, and returns must join consecutive scheduled sessions. The fixed cohort and unverified adjustment convention limit investment inference. Classical PCA could reduce larger inputs; QPCA has a different data-access and readout model and is not required for this small case. The notebook is a teaching interface, not a prescribed software standard. To rebuild from another authorised archive, use the data-preparation module; use provider-managed authentication for any later hardware extension and keep literal tokens out of shared cells, scripts and outputs."),
      "02_portfolio":notebook("Portfolio Selection: Exact Search, Local Swap and QAOA",
        "Companion to Chapter 2, Section 2.2. Translate an equal-weight four-of-eight portfolio objective into QUBO coefficients and an Ising circuit. Separate exact simulator parameter training from finite-shot final decoding. Follow the equations before running the full case: means scale by 1/K, covariance by 1/K squared, and a penalty enforces K=4.",[
        ("1. Reproduce the monthly experiment","Each January--August 2026 decision uses 126 complete pre-decision return vectors. Risk aversion is 5. QAOA depths 1 and 2 each use three fixed seeds, at most 80 COBYLA evaluations and 2,048 final shots. This cell regenerates the results; it does not call a brokerage.",
         'from portfolio import run\nsummary = run()\ndisplay(pd.Series({k: summary[k] for k in ["dates", "runs", "optimal_runs", "failure_runs", "mean_feasibility", "mean_optimum_probability"]}, name="Results").to_frame())'),
        ("2. Compare financial decisions","The objective gap is computed against exact feasible enumeration. It is not a return gap. The selected basket is the best observed feasible string, without substituting an exact optimum after a failed quantum run.",
         'runs = pd.read_csv(RESULTS / "portfolio_runs.csv")\ndisplay(runs.groupby(["method", "depth"]).agg(mean_gap=("gap", "mean"), mean_feasibility=("feasible_probability", "mean"), mean_seconds=("seconds", "mean")))\ndisplay(runs[(runs.method == "QAOA") & (runs.gap > 1e-10)][["month", "depth", "seed", "gap", "holdings"]])'),
        ("3. Examine the sampling control","A tiny finite search space is easy to cover with many random samples. Compare against both unconstrained strings and uniform feasible baskets before interpreting a high best-of-many success rate.",
         'sampling = pd.read_csv(RESULTS / "portfolio_sampling.csv")\ndisplay(sampling.pivot(index="shots", columns="method", values="success").round(4))\ndisplay(Image(filename=str(RESULTS / "portfolio_comparison.png")))')],
        "Use the displayed summary rather than a hard-coded success claim. Exact search evaluates only 70 feasible baskets, and uniform feasible sampling is a strong control. Ask which step changes if the penalty, mixer or depth changes. A high best-of-many success rate does not imply a large one-shot optimum probability. The return calculation is a stylised fixed-cohort price outcome with a 20-basis-point round-trip deduction, not evidence of persistent alpha."),
      "03_pricing_risk":notebook("Payoff Amplitudes and Tail-Loss Queries",
        "Companion to Chapter 2, Sections 2.1 and 2.3. Build the probability and payoff registers for a hypothetical call, then reuse the bounded-payoff construction for a tail indicator and excess loss. Controlled RY angles are twice arcsin(sqrt(scaled payoff)); squaring amplitudes therefore recovers the desired payoff fraction.",[
        ("1. Reproduce pricing and risk calculations","Historical AAPL volatility calibrates an explicitly hypothetical non-dividend-paying model with S0=K=100, r=3% and T=0.25. Future payoffs are synthetic. MLAE uses genuine small-register amplification powers 0, 1, 2 and 4, with 250 shots each.",
         'from pricing_risk import run\nsummary = run()\ndisplay(pd.Series({k: summary[k] for k in ["calibration_end", "calibration_n", "sigma", "black_scholes", "mlae_target", "preparation_calls", "mlae_rmse", "direct_rmse"]}, name="Pricing").to_frame())'),
        ("2. Separate grid bias from measurement noise","The continuous analytical price and the discrete 16-bin target are different. The finite-query comparison measures errors around the discrete target. Finer bins can reduce representation bias but require larger preparation circuits.",
         'display(pd.read_csv(RESULTS / "pricing_discretisation.csv").round(6))\ndisplay(Image(filename=str(RESULTS / "pricing_errors.png")))'),
        ("3. Interpret tail quantities","The strict probability beyond a discrete 95% quantile need not equal 5%. Expected shortfall uses q + E[(L-q)+]/0.05, which accounts for the needed part of the atom at q. The quantile is fixed classically in this executed experiment.",
         'display(pd.Series(summary["risk"], name="Risk output").to_frame())\ndisplay(pd.read_csv(RESULTS / "risk_grid.csv").round(6))'),
        ("4. Extend to Asian paths and correlated factors", "Two four-bin Gaussian increments generate sixteen two-date paths. An actual five-qubit payoff lookup checks the discrete expectation; independent continuous Monte Carlo has the same two monitoring dates. The 32-date comparator prices a different monitoring contract. Factor scenarios retain all eight eigenvectors and compare dependence assumptions before any loss is encoded.",
         'from teaching_cases import run as run_extensions\nextensions = run_extensions()\ndisplay(pd.Series(extensions["asian"], name="Two-date Asian example").to_frame())\ndisplay(pd.read_csv(RESULTS / "factor_scenarios.csv").round(6))')],
        "Read the grid and measurement errors separately. More shots cannot eliminate coarse path-grid bias. The Asian circuit loads an enumerated payoff lookup; it does not implement scalable reversible path generation. In the factor example, a negative leading-factor shift is a chosen stress rather than a forecast. Try varying one representation setting while keeping the financial contract fixed."),
      "04_quantum_learning":notebook("Weekly Equity Learning and Quantum Kernel Controls",
        "Companion to Chapter 3. Compare classical models, a product fidelity kernel, a rotation/re-uploading family and a small variational classifier. The target is next-week membership of the top four in a fixed eight-stock cohort. Four economic features use four qubits; eight securities determine observations, not the feature-register width. Regime detection is explained in the chapter as a related extension, not reported as an executed experiment here.",[
        ("1. Execute chronological model selection","Features use the prior completed week. The year 2024 trains initial models; 2025 selects among pre-specified configurations. Final models refit the latest 52 eligible pre-2026 weeks. All 2026 results remain out of selection. Complete-window rules exclude six target weeks after the missing AAPL observation.",
         'from quantum_learning import run\nsummary = run()\ndisplay(pd.Series({k: summary[k] for k in ["training_2024", "validation_2025", "refit", "test", "test_weeks", "test_last_label_end"]}, name="Split sizes").to_frame())\ndisplay(pd.DataFrame(summary["selected"])[["method", "C", "scale", "layers", "auc"]])'),
        ("2. Examine the financial evidence","AUC pools observations while IC ranks the eight names within each week. No seed is selected on the final test. The depth-two model is a separate diagnostic, not a replacement for the validation-selected one-layer map.",
         'metrics = pd.read_csv(RESULTS / "qml_metrics.csv")\ndisplay(metrics[["method", "auc", "balanced_accuracy", "mean_ic", "mean_top2"]].round(4))\ndisplay(Image(filename=str(RESULTS / "qml_comparison.png")))'),
        ("3. Check classical equivalence", "A terminal fixed entangler cancels from a fidelity kernel. Both the product kernel and the selected one-layer two-axis kernel have explicit classical formulas. This checks the representation, not forecasting performance.",
         'display(pd.Series({"Product identity error": summary["product_identity_max_error"], "Two-axis identity error": summary["two_axis_identity_max_error"]}, name="Circuit checks").to_frame())'),
        ("4. Resample complete weeks, preserving model pairing", "The 224 stock-week rows contain only 28 complete weeks. Each resampled week carries all eight stocks, labels, returns and frozen model scores. Pooled AUC is recomputed, not averaged from within-week AUCs. Five thousand paired replicates use 1-, 2- and 4-week blocks. Longer blocks must be consecutive calendar weeks: they cannot wrap or bridge the excluded period. The same indices are used for every model. Model fitting and earlier selection are not repeated; edge weighting and the short non-stationary sample limit interpretation.",
         'uncertainty = json.loads((RESULTS / "qml_uncertainty.json").read_text())\nintervals = pd.read_csv(RESULTS / "qml_auc_intervals.csv")\ndisplay(intervals[intervals.block_weeks == 4].round(4))\ndisplay(pd.DataFrame(uncertainty["sensitivity"]).round(4))\ndisplay(pd.DataFrame(uncertainty["vqc_seed_summary"]).T.round(4))\nprint("VQC dispersion is sample SD across three specified initialisations, not a confidence interval.")')],
        "The selected kernel's point estimates exceed the RBF baseline. The four-week AUC-difference interval excludes zero, but the independent-week interval includes zero; report both rather than choose the favourable result. Paired IC intervals include zero. None establishes stable predictive or quantum computational advantage. The one-layer fidelity has a classical closed form. VQC mean and sample SD summarise all three fixed seeds, not a selected best seed or an ensemble. A falling training loss is not out-of-sample validation."),
      "05_hardware_validation":notebook("Cross-SDK, Shot and Noise Validation",
        "Companion to Chapter 4. Quantify what changes when a logical financial circuit is represented, compiled, measured or subjected to a specified noise model. Run the portfolio and learning notebooks first.",[
        ("1. Run the controlled implementation studies","The Qiskit topology study compares all-to-all and line connectivity. Kernel-shot experiments hold market inputs fixed. Mixed-state noise experiments use 24 fixed pairs, not a claimed physical device calibration.",
         'assert (RESULTS / "portfolio_parameters.json").exists()\nassert (RESULTS / "qml_kernel_inputs.npz").exists()\nfrom hardware_validation import run\nsummary = run()\nprint("Maximum SDK probability difference:", summary["max_sdk_probability_error"])\ndisplay(pd.read_csv(RESULTS / "hardware_compilation.csv").drop(columns="probability_error"))'),
        ("2. Assess measurement and spectral effects","Independent shot estimates can make a kernel indefinite. Positive spectral landmarks supply both a repaired training representation and a defined out-of-sample map. Seed standard deviations here are not uncertainty about the stock-market population.",
         'display(pd.DataFrame(summary["shot_summary"]).round(5))\ndisplay(Image(filename=str(RESULTS / "hardware_sensitivity.png")))'),
        ("3. Separate circuit simplification from sampling","The reduced overlap circuit removes terminal cancelling entanglers before inserting noise. Unit angular scale is used for these 24-pair diagnostics and matches the selected classifier's angular scale, but depths and noise conditions are controlled separately. This preserves ideal overlaps but lowers channel exposure. More shots would not remove this channel bias.",
         'noise = pd.read_csv(RESULTS / "hardware_noise_kernel.csv")\ndisplay(noise[noise.eta == .01][["depth", "cancelled", "rmse", "max_error"]].round(5))'),
        ("4. Calculate post-quantum object sizes", "PQC runs on classical computers and protects a different layer. This cell adds object lengths from FIPS 203--205. It does not implement cryptography, measure latency or assert that a protocol is secure. Distinguish KEM ciphertexts, signatures and public keys; certificates and framing are not included.",
         'from teaching_cases import pqc_object_sizes\nsizes = pqc_object_sizes()\ndisplay(pd.DataFrame(sizes["objects"]))\nprint("One KEM key/ciphertext plus one signature key/signature:", sizes["kem_key_and_ciphertext"] + sizes["dsa_key_and_signature"], "bytes")\nfrom teaching_interfaces import smoke_test\nprint(smoke_test())')],
        "Cross-SDK probabilities agree numerically after bit-order correction. Compare the eight-qubit row under both topologies to see the routing overhead. All circuit results remain local simulations. Physical-device validation needs target-specific compilation, calibration metadata, actual counts and timing. PQC object sizes are derived quantities, not a deployment test. The chapter supplies a proposed authenticated-result teaching exercise, clearly distinguished from what this notebook executes.")}
    for name,nb in specs.items():
        if only is not None and name != only:
            continue
        path=NOTEBOOKS/f"{name}.ipynb"
        nb.metadata.kernelspec.name = kernel_name
        nbf.validate(nb)
        if execute:
            client=NotebookClient(nb,timeout=600,kernel_name=kernel_name,resources={"metadata":{"path":str(NOTEBOOKS)}})
            client.execute()
        nbf.write(nb,path)
        if execute:
            html,_=HTMLExporter().from_notebook_node(nb)
            (NOTEBOOKS/f"{name}.html").write_text(html)
        print(f"{'Executed' if execute else 'Built'} {path.name}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true")
    p.add_argument("--refresh-learning-figure", action="store_true",
                   help="Replace the cached Figure 3.1 image without rerunning fits")
    p.add_argument("--kernel", default="python3", help="Registered Jupyter kernel in the tested environment")
    p.add_argument("--only", choices=["01_environment", "02_portfolio", "03_pricing_risk",
                                     "04_quantum_learning", "05_hardware_validation"])
    args=p.parse_args()
    if args.refresh_learning_figure:
        if args.execute or args.only:
            p.error("Figure refresh is separate from notebook construction/execution")
        refresh_learning_figure()
    else:
        build(args.execute, args.only, args.kernel)
