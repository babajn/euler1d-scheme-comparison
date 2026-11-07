"""Run scalar and Euler numerical checks without executing notebooks."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scalar1d import run_tests
from euler1d_core import run_unit_tests


def main():
    for label, tests in (("Scalar", run_tests()), ("Euler", run_unit_tests(1.4))):
        for number, description in enumerate(tests, 1):
            print(f"{label} PASS {number}: {description}")
        print(f"{label}: {len(tests)} test groups passed.")


if __name__ == "__main__":
    main()
