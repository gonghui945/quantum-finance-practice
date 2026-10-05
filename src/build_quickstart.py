"""Generate deterministic synthetic CSVs and a commented quickstart notebook."""
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build():
    data = ROOT / "examples" / "synthetic"
    data.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "asset": [f"Asset_{i + 1:02d}" for i in range(8)],
        "annual_mean": np.linspace(0.01, 0.10, 8),
        "annual_variance": np.full(8, 0.04),
    }).to_csv(data / "assets.csv", index=False, float_format="%.12g")
    pd.DataFrame({
        "outcome": np.arange(4),
        "probability": [0.1, 0.2, 0.3, 0.4],
        "payoff": [0.0, 1.0, 2.0, 3.0],
    }).to_csv(data / "payoffs.csv", index=False)
    features = pd.DataFrame(
        np.random.default_rng(11).normal(size=(12, 4)),
        columns=[f"angle_{i}" for i in range(4)],
    )
    features.insert(0, "observation", np.arange(12))
    features["label"] = np.arange(12) % 2
    features.to_csv(data / "features.csv", index=False, float_format="%.12g")

    cells = [
        nbf.v4.new_markdown_cell(
            "# Quantum Finance: Synthetic Quickstart\n\n"
            "Connect the conventions in Chapters 1-4 to small, executable examples. "
            "All inputs are synthetic CSVs included in the repository. The notebook "
            "runs locally using PennyLane and Qiskit.\n\n"
            "These examples exercise representation and decoding. The chapter "
            "notebooks provide the separate empirical workflows."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup and Input Assumptions\n\n"
            "Select the pinned Python environment, then run the cells in order. "
            "Eight hypothetical assets have assumed annualised means and diagonal "
            "variances. Four payoffs and twelve angular feature vectors illustrate "
            "pricing and learning. No account or token is needed."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            "# Support starting Jupyter from the root or notebooks folder.\n"
            "ROOT = Path.cwd().resolve()\n"
            "if not (ROOT / 'src').is_dir():\n"
            "    ROOT = ROOT.parent\n"
            "assert (ROOT / 'src').is_dir(), 'Open inside the repository.'\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from quickstart import (inputs, register_example, pricing_example,\n"
            "                        portfolio_example, kernel_example, measurement_example)\n"
            "from teaching_interfaces import security_size_example\n\n"
            "assets, payoffs, features = inputs()\n"
            "display(assets)\n"
            "display(payoffs)\n"
            "display(features.head())"
        ),
    ]
    steps = [
        (
            "1. Registers and Bit Order",
            "An eight-wire simulator stores 256 amplitudes. Hadamards give equal "
            "Born probabilities. An asymmetric two-wire Pauli X example shows "
            "why SDKs display different array orders. Record the wire-to-asset "
            "mapping before interpreting holdings.",
            "display(pd.Series(register_example(), name='Register checks').to_frame())",
        ),
        (
            "2. From Ancilla Probability to a Price",
            "Normalise a non-negative payoff by its cap and encode it with an "
            "ancilla rotation. Multiply the success probability by the cap to "
            "recover the expectation, then discount once. The declared classical "
            "expectation is 2.0; the encoded calculation must reproduce it.",
            "display(pd.Series(pricing_example(payoffs), name='Payoff decoding').to_frame())",
        ),
        (
            "3. From Bit Strings to Portfolio Holdings",
            "Four of eight assets give 70 feasible baskets. The QUBO contains "
            "a risk-return objective and cardinality penalty. Two QAOA layers "
            "use fixed angles, followed by 256 seeded samples. Compare the best "
            "observed feasible basket with exact enumeration. A penalised "
            "expectation and a decoded holding decision are different outputs.",
            "# Fixed angles illustrate decoding rather than fitted performance.\n"
            "display(pd.Series(portfolio_example(assets), name='Portfolio decoding').to_frame())",
        ),
        (
            "4. From Quantum Similarity to a Classifier",
            "Four angular inputs use four qubits. Eight observations form the "
            "training kernel; four form its prediction cross-kernel. One-layer "
            "fidelity agrees with a classical formula. A classical support-vector "
            "classifier combines the kernel with labels to produce decision "
            "scores. Similarity alone is not a financial forecast.",
            "kernel = kernel_example(features)\n"
            "display(pd.DataFrame(kernel['train_kernel']).round(4))\n"
            "display(pd.DataFrame({'synthetic_decision_score': kernel['decision_scores']}))\n"
            "print('Classical identity error:', kernel['closed_form_max_error'])",
        ),
        (
            "5. Finite-Shot Diagnostics",
            "An overlap is estimated by counting complete all-zero outcomes. "
            "This cell samples 1,000 binomial shots per pair. Sampling can yield "
            "a kernel requiring spectral repair; both training and prediction "
            "maps must be defined. The example simulates counting uncertainty.",
            "display(pd.Series(measurement_example(kernel['train_kernel'],\n"
            "                                     kernel['cross_kernel']),\n"
            "                  name='Measurement diagnostics').to_frame())",
        ),
        (
            "6. Post-Quantum Object Sizes",
            "PQC runs on classical computers and protects a different layer. "
            "The helper adds standard lengths of keys, a ciphertext and a "
            "signature. It performs size accounting rather than encryption "
            "or protocol authentication.",
            "display(pd.Series(security_size_example(), name='Derived object sizes').to_frame())",
        ),
    ]
    for title, explanation, code in steps:
        cells.extend([
            nbf.v4.new_markdown_cell(f"## {title}\n\n{explanation}"),
            nbf.v4.new_code_cell(code),
        ])
    cells.append(nbf.v4.new_markdown_cell(
        "## Continue with the Financial Cases\n\n"
        "Read 01_environment.ipynb for the market-data contract, then pricing, "
        "portfolio and learning. Hardware diagnostics require their saved outputs. "
        "DATA_SCHEMA.md and RECONSTRUCTION.md describe authorised empirical inputs.\n\n"
        "In this quickstart, vary one assumption at a time: payoff scale, "
        "portfolio settings or shot count. Keep the financial output explicit."
    ))
    path = ROOT / "notebooks" / "00_quickstart.ipynb"
    # Preserve the scaffold's notebook format when replacing its example cells.
    notebook = nbf.read(path, as_version=4) if path.exists() else nbf.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.7"},
    }
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print("Generated three synthetic CSVs and 00_quickstart.ipynb.")


if __name__ == "__main__":
    build()
