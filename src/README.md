# Implementation Modules

| Module | Main purpose |
| --- | --- |
| quickstart.py | Runnable tour with the included synthetic CSVs |
| teaching_interfaces.py | Short circuit and decoding interfaces from Appendix A.3 |
| portfolio.py | QUBO, QAOA and classical portfolio controls |
| pricing_risk.py | Payoff encoding, amplitude estimation and tail queries |
| quantum_learning.py | Weekly features, kernel classifiers and variational learning |
| qml_uncertainty.py | Paired resampling of complete market weeks |
| hardware_validation.py | Local shot/noise, compilation and SDK diagnostics |
| teaching_cases.py | Asian payoff lookup, factor scenarios and PQC sizes |
| prepare_data.py | Adapter for an authorised source archive |
| common.py | Shared paths, plotting and serialization |
| run_cases.py | Command-line execution of the empirical workflows |
| build_notebooks.py | Build the five chapter notebooks |
| build_quickstart.py | Regenerate synthetic CSVs and the introductory notebook |

Run modules from the repository root. The quickstart works with the supplied
synthetic inputs. Empirical case modules expect the daily data contract in
DATA_SCHEMA.md. All quantum execution is local simulation.
