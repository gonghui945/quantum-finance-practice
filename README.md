# Quantum Computing for Financial Applications
## Practice, Workflows and Benchmarking

Computational companion by **Hui Gong and Francesca Medda**, UCL IFT Center for
Quantum Finance, University College London.

This repository accompanies the October 2026 manuscript *Quantum Computing for
Financial Applications: Practice, Workflows and Benchmarking*. It connects
financial definitions to quantum circuits, measurement, classical comparators
and the interpretation of results. Publication details will be updated when available.

**Start with [00_quickstart.ipynb](notebooks/00_quickstart.ipynb).** It runs locally
with the included synthetic CSVs. The five chapter notebooks require the separate
market inputs described in [DATA_SCHEMA.md](DATA_SCHEMA.md).
All quantum execution here uses local simulators.

## Notebooks and Chapters

| Notebook | Manuscript location | Main topics | Inputs |
| --- | --- | --- | --- |
| [00_quickstart.ipynb](notebooks/00_quickstart.ipynb) | Chapters 1-4; Appendix A.3 | Registers, payoff decoding, QAOA, kernels and shot diagnostics | Included synthetic CSVs |
| [01_environment.ipynb](notebooks/01_environment.ipynb) | Sections 1.2-1.5 | Data/model preparation, registers, SDK ordering and benchmarking | Authorised daily inputs |
| [03_pricing_risk.ipynb](notebooks/03_pricing_risk.ipynb) | Sections 2.1 and 2.3 | Payoff loading, amplitude estimation, Asian lookup and tail queries | Daily inputs for calibration |
| [02_portfolio.ipynb](notebooks/02_portfolio.ipynb) | Section 2.2 | QUBO/Ising encoding, QAOA and classical controls | Authorised daily inputs |
| [04_quantum_learning.ipynb](notebooks/04_quantum_learning.ipynb) | Chapter 3 | Kernels, variational learning and paired weekly uncertainty | Authorised daily inputs |
| [05_hardware_validation.ipynb](notebooks/05_hardware_validation.ipynb) | Chapter 4 | SDK agreement, shots, noise, compilation and PQC sizes | Portfolio and learning outputs |

Read pricing before portfolio to follow Chapter 2. Run notebook 05 after portfolio
and learning. File numbers identify notebooks rather than prescribe reading order.

## Installation and First Run

Use Python 3.12 in a dedicated environment. The tested reference is Python 3.12.7,
PennyLane 0.36.0 and Qiskit 2.2.3. These versions are pinned for reproduction.

```bash
git clone https://github.com/gonghui945/quantum-finance-practice.git
cd quantum-finance-practice
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pinned.txt
python src/quickstart.py
python -m unittest discover -s tests -v
python verify_manifest.py
```

On Windows, use `py -3.12 -m venv .venv` and activate
`.venv\Scripts\Activate.ps1` in PowerShell.

Open the notebooks in VS Code, Jupyter Notebook or JupyterLab using that environment.
To register its kernel:

```bash
python -m ipykernel install --user --name quantum-finance-practice --display-name "Quantum Finance Practice"
```

The quickstart uses eight hypothetical asset means and variances, four payoff
outcomes and twelve seeded feature vectors. No market download or account credential
is required. It also runs directly as `src/quickstart.py`.

Official guides: [PennyLane installation](https://pennylane.ai/install),
[PennyLane circuits](https://docs.pennylane.ai/en/stable/introduction/circuits.html)
and [Qiskit installation](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit).

## Files and Empirical Reproduction

| Folder/file | Role |
| --- | --- |
| `notebooks/` | Six tutorials with English explanations |
| `src/` | Commented modules and notebook builders |
| `tests/` | Encoding, probability, chronology and uncertainty checks |
| `examples/synthetic/` | Small reproducible CSV inputs |
| `requirements-pinned.txt` | Tested dependency versions |
| `DATA_SCHEMA.md` and `RECONSTRUCTION.md` | Market-data contract and transformation instructions |
| `MANIFEST.sha256` | File integrity record |
| `CITATION.cff` | Software citation metadata |

Algorithms live in `src/`; notebooks explain and exercise them.
Distributed notebooks have cleared outputs. Locally generated empirical
`data/` and `results/` are excluded by `.gitignore`.

The manuscript's illustrative equity cohort is AAPL, MSFT, AMZN, GOOGL, META, JPM,
GS and XOM. The window is **1 January 2024 to 31 August 2026**. Portfolio selection
chooses four of eight names; learning uses four features and four feature qubits.

The authors identify Bloomberg as the upstream provider. The public repository
includes [synthetic examples](examples/synthetic/README.md); original market
observations, predictions and fitted arrays require authorised access.
See [DATA_AND_PUBLICATION.md](DATA_AND_PUBLICATION.md).

Place required files in `data/` following [DATA_SCHEMA.md](DATA_SCHEMA.md).
[RECONSTRUCTION.md](RECONSTRUCTION.md) explains preprocessing and execution order.
An independently acquired dataset is an adapted experiment unless it matches the
original processed extract and conventions.

```bash
python src/run_cases.py --case all
python src/build_notebooks.py --execute --kernel quantum-finance-practice
```

These empirical cases take longer than the quickstart and regenerate local outputs.

## Reading the Evidence

The benchmarks distinguish task fidelity, approximation, statistical precision,
classical comparators, resources, stability, financial relevance and execution evidence.
Portfolio objective quality and sampling efficiency are reported separately.
Pricing distinguishes representation bias from measurement error. Kernel similarity
becomes a prediction through a fitted classifier. The selected one-layer kernel has a
classical closed form. PQC examples calculate standard object lengths; physical-device
extensions and cryptographic protocols are described as further work.

The original suite contains 25 tests. With this public package, 14 data-free checks
run and 11 empirical checks explicitly skip until their inputs/results are available.
The quickstart is additionally executed in full. Skipped tests identify missing evidence.

## Citation and Licensing

Use [CITATION.cff](CITATION.cff) for this computational companion.
The manuscript can currently be cited as:

> Gong, H. and Medda, F. (2026). *Quantum Computing for Financial Applications:
> Practice, Workflows and Benchmarking*. Manuscript in preparation.

[LICENSING.md](LICENSING.md) records the current code-licensing status.
A specific open-source licence has not yet been assigned. Third-party libraries
retain their own licences; data permissions are separate.

Questions can be raised through the repository's issue tracker.
