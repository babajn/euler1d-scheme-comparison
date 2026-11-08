"""Repair known malformed LaTeX in Markdown cells, preserving numerical outputs.

Replacements are intentionally narrow: all code cells, execution counts,
plots, tables, animation output, and notebook metadata remain unchanged.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "01_advection_burgers.ipynb",
    "02_euler_classical.ipynb",
    "03_euler_riemann_solvers.ipynb",
    "04_spatial_reconstruction.ipynb",
    "05_time_integration.ipynb",
)
REPAIRS = {
    r"\frac C_{\mathrm{adv}}2": r"\frac{C_{\mathrm{adv}}}{2}",
    r"\bar\bar\rho": r"\bar\rho",
}

def check(markdown, name, cid):
    for bad in REPAIRS:
        if bad in markdown:
            raise ValueError(f"{name}, cell {cid}: malformed TeX {bad}")
    for char in markdown:
        if ord(char) < 32 and char not in "\n\t":
            raise ValueError(f"{name}, cell {cid}: invalid control character U+{ord(char):04X}")

def fix(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    original_non_markdown = [c for c in nb["cells"] if c["cell_type"] != "markdown"]
    original_metadata = nb.get("metadata")
    updated = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = cell["source"]
        before = "".join(source) if isinstance(source, list) else source
        after = before
        for old, new in REPAIRS.items():
            after = after.replace(old, new)
        check(after, path.name, cell.get("id"))
        if after != before:
            cell["source"] = after.splitlines(keepends=True) if isinstance(source, list) else after
            updated.append(cell.get("id"))
    assert original_non_markdown == [c for c in nb["cells"] if c["cell_type"] != "markdown"]
    assert original_metadata == nb.get("metadata")
    if updated:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path.name}: fixed {updated or 'none'}; execution output unchanged")
    return updated

def main():
    paths = [ROOT / "notebooks" / name for name in NAMES]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Expected five teaching notebooks")
    for path in paths:
        fix(path)

if __name__ == "__main__":
    main()
