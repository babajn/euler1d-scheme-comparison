"""Apply instructional Markdown sections without changing code or stored outputs."""
from pathlib import Path
import hashlib
import json
import re

HEADER = re.compile(r"^<!-- (notebook|insert-before|append):([^>]+) -->\s*$", re.MULTILINE)


def text(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def apply_document(root, document):
    material = document.read_text(encoding="utf-8")
    markers = list(HEADER.finditer(material))
    if not markers or markers[0].group(1) != "notebook":
        raise ValueError(f"Missing notebook declaration: {document}")
    name = markers[0].group(2).strip()
    if Path(name).name != name or not name.endswith(".ipynb"):
        raise ValueError(f"Invalid notebook name: {name}")
    path = root / "notebooks" / name
    nb = json.loads(path.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(nb))
    codes_before = [c for c in nb["cells"] if c["cell_type"] == "code"]
    non_markdown_before = [c for c in nb["cells"] if c["cell_type"] != "markdown"]
    before_hash = digest(non_markdown_before)
    inserted = 0
    appended = 0
    for i, marker in enumerate(markers[1:], 1):
        action, arguments = marker.group(1), marker.group(2).strip()
        stop = markers[i + 1].start() if i + 1 < len(markers) else len(material)
        content = material[marker.end():stop].strip()
        anchor, *new_id = arguments.split(":")
        matches = [j for j, cell in enumerate(nb["cells"]) if cell.get("id") == anchor]
        if len(matches) != 1 or (action == "append" and nb["cells"][matches[0]]["cell_type"] != "markdown"):
            raise ValueError(f"Missing or ambiguous Markdown anchor {anchor} in {name}")
        if action == "insert-before":
            if len(new_id) != 1:
                raise ValueError(f"Missing new cell ID: {arguments}")
            cid = new_id[0]
            existing = [c for c in nb["cells"] if c.get("id") == cid]
            if existing:
                if len(existing) != 1 or text(existing[0]) != content:
                    raise ValueError(f"Existing instructional cell differs: {cid}")
            else:
                nb["cells"].insert(matches[0], {"cell_type": "markdown", "id": cid, "metadata": {}, "source": content})
                inserted += 1
        elif action == "append":
            cell = nb["cells"][matches[0]]
            current = text(cell)
            if content not in current:
                changed = current.rstrip() + "\n\n" + content
                cell["source"] = changed.splitlines(keepends=True) if isinstance(cell["source"], list) else changed
                appended += 1
        else:
            raise ValueError(f"Unknown operation: {action}")
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        before = text(cell)
        after = before.replace("$dU/dt=", r"$\frac{dU}{dt}=")
        after = after.replace(r"$y'=\lambda y$", r"$\frac{dy}{dt}=\lambda y$")
        if name == "03_euler_riemann_solvers.ipynb" and cell.get("id") == "modern-004":
            after = after.replace("一様格子上で区分一定の状態を用いる場合、以下のLF/LWの保存形実装は対応する古典的な差分更新になります。", "この部では区分一定の左右状態を作り、4種類の数値流束のいずれかを用いて同じ保存形で更新します。")
        if after != before:
            cell["source"] = after.splitlines(keepends=True) if isinstance(cell["source"], list) else after
    ids = [c.get("id") for c in nb["cells"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicated cell IDs: {name}")
    after_hash = digest([c for c in nb["cells"] if c["cell_type"] != "markdown"])
    if before_hash != after_hash:
        raise AssertionError(f"Code, outputs or non-Markdown cells changed: {name}")
    if original["metadata"] != nb["metadata"]:
        raise AssertionError(f"Notebook metadata changed: {name}")
    outputs = [o for c in codes_before for o in c.get("outputs", [])]
    report = {
        "notebook": name,
        "inserted_markdown_cells": inserted,
        "appended_markdown_cells": appended,
        "code_cells": len(codes_before),
        "executed_code_cells": sum(c.get("execution_count") is not None for c in codes_before),
        "png_outputs": sum("image/png" in o.get("data", {}) for o in outputs),
        "error_outputs": sum(o.get("output_type") == "error" for o in outputs),
        "non_markdown_sha256_before": before_hash,
        "non_markdown_sha256_after": after_hash,
        "code_and_outputs_preserved": True,
    }
    if nb != original:
        temp = path.with_suffix(".ipynb.tmp")
        temp.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    print(json.dumps(report, ensure_ascii=False))
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    documents = sorted((root / "docs/teaching").glob("0[1-5]_basics.md"))
    if len(documents) != 5:
        raise RuntimeError(f"Expected five teaching documents, found {len(documents)}")
    reports = [apply_document(root, doc) for doc in documents]
    destination = root / "validation/beginner_explanations.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({"scope": "Markdown only; no numerical rerun", "notebooks": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
