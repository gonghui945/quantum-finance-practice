# Synthetic Quickstart Inputs

These CSVs are generated from declared constants and NumPy random seed 11.
They contain no market observations or fitted empirical outputs.

| File | Contents | Role |
| --- | --- | --- |
| assets.csv | Eight hypothetical assets, annualised means and variances | Four-of-eight QUBO and QAOA |
| payoffs.csv | Four declared probabilities and non-negative payoffs | Ancilla decoding versus a classical weighted expectation |
| features.csv | Twelve four-dimensional angular vectors and alternating labels | Kernel matrices, classifier interfaces and shot diagnostics |

Covariance is diagonal and formed from the annual_variance column.
Asset identifiers are artificial; means are assumptions rather than estimates.
Features are already angular inputs. Alternating labels exercise the classifier
interface. The first eight rows fit the classifier and the last four exercise prediction.

Run the following from the repository root to regenerate the inputs and notebook:

```bash
python src/build_quickstart.py
```

Empirical reproduction uses the separate DATA_SCHEMA.md contract and chapter notebooks.
