"""Run the local financial examples without a notebook or a cloud account.

Execute from the package root, for example: python src/run_cases.py --case all
Saved outputs are regenerated in results/. Hardware validation here means local
simulation, compilation and noise diagnostics, never a physical job submission.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "portfolio", "pricing", "learning", "hardware"], default="all")
    args = parser.parse_args()
    names = ["portfolio", "pricing", "learning", "hardware"] if args.case == "all" else [args.case]
    for name in names:
        print(f"Running local {name} workflow", flush=True)
        if name == "portfolio":
            from portfolio import run
        elif name == "pricing":
            from pricing_risk import run
        elif name == "learning":
            from quantum_learning import run
        else:
            from hardware_validation import run
        result = run()
        if name == "pricing":
            from teaching_cases import run as extensions
            extensions()
        # Saved JSON/CSV outputs contain the full scientific record.
        print(json.dumps({"case": name, "status": "completed", "summary_fields": sorted(result)}), flush=True)


if __name__ == "__main__":
    main()
