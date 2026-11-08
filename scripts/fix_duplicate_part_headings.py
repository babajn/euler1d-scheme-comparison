"""Keep a single top-level chapter title in Euler textbook notebooks.

Only two Markdown headings are renamed. Numerical code, stored output,
execution counts, and notebook metadata are left unchanged.
"""
from pathlib import Path
import hashlib
import json

TARGETS = {
    "02_euler_classical.ipynb": (
        "classical-013", "# 第II部：古典的手法 — LFとLW", "## LFとLWの数値解法", "第II部"
    ),
    "03_euler_riemann_solvers.ipynb": (
        "modern-011", "# 第III部：現代の実用的手法 — 数値流束の比較", "## Roe・LLF・HLL・HLLCの数値流束比較", "第III部"
    ),
}

def cell_text(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source

def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

def process(path, target_id, old_title, new_title, chapter):
    nb = json.loads(path.read_text(encoding="utf-8"))
    other_before = [cell for cell in nb["cells"] if cell["cell_type"] != "markdown"]
    metadata_before = json.loads(json.dumps(nb["metadata"]))
    cells = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown" and cell.get("id") == target_id]
    if len(cells) != 1:
        raise ValueError(f"Expected one heading cell {target_id}: {path}")
    cell = cells[0]
    source = cell_text(cell)
    if old_title in source:
        if source.count(old_title) != 1:
            raise ValueError(f"Duplicate exact old title in {path}")
        new_source = source.replace(old_title, new_title)
        cell["source"] = new_source.splitlines(keepends=True) if isinstance(cell["source"], list) else new_source
        changed = True
    elif new_title in source:
        changed = False
    else:
        raise ValueError(f"Neither expected heading is present in {path}: {source[:150]}")
    headings = [line for c in nb["cells"] if c["cell_type"] == "markdown" for line in cell_text(c).splitlines() if line.startswith(f"# {chapter}")]
    if len(headings) != 1:
        raise AssertionError(f"Expected exactly one top-level {chapter} title: {headings}")
    other_after = [c for c in nb["cells"] if c["cell_type"] != "markdown"]
    assert digest(other_before) == digest(other_after), "Code or execution results changed"
    assert metadata_before == nb["metadata"], "Notebook metadata changed"
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{path.name}: {'fixed' if changed else 'already fixed'}; single {chapter} title; code and outputs preserved")

def main():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for name, args in TARGETS.items():
        process(root / name, *args)

if __name__ == "__main__":
    main()
