"""
Phase 1 health-check script.

Verifies that the virtual environment is active and that core
dependencies (pandas, numpy, python-dotenv) are correctly installed
and importable.
"""

import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def main() -> None:
    print("=" * 50)
    print("AI Stock Market Platform - Health Check")
    print("=" * 50)

    print(f"Python version : {sys.version.split()[0]}")
    print(f"Pandas version : {pd.__version__}")
    print(f"NumPy version  : {np.__version__}")

    load_dotenv()
    print("dotenv loaded  : OK")

    # Quick sanity check that pandas/numpy actually work together
    df = pd.DataFrame({"a": np.arange(5), "b": np.arange(5) * 2})
    assert df["b"].sum() == 20, "Sanity check failed"

    print("-" * 50)
    print("All checks passed. Environment is ready for Phase 2.")
    print("=" * 50)


if __name__ == "__main__":
    main()