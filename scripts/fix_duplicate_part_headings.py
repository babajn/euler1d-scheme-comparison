"""Normalize the Markdown outline of Parts II and III without touching code/output."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1] / "notebooks"

def txt(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

def get(nb, cid):
    found = [c for c in nb["cells"] if c.get("id") == cid and c["cell_type"] == "markdown"]
    if len(found) != 1:
        raise ValueError(f"Expected one markdown cell {cid}")
    return found[0]

def edit(nb, cid, old, new):
    cell = get(nb, cid)
    s = txt(cell)
    if new in s:
        return
    possibilities = old if isinstance(old, tuple) else (old,)
    matched = [candidate for candidate in possibilities if candidate in s.splitlines()]
    if len(matched) != 1:
        raise ValueError(f"Unexpected heading in {cid}: {old}")
    s = s.replace(matched[0], new, 1)
    cell["source"] = s.splitlines(keepends=True) if isinstance(cell["source"], list) else s

def numbered(nb, cid, oldprefix, newprefix):
    s = txt(get(nb, cid))
    if any(line.startswith(newprefix) for line in s.splitlines()):
        return
    match = [line for line in s.splitlines() if line.startswith(oldprefix)]
    if len(match) != 1:
        raise ValueError(f"Missing numbered heading {oldprefix} in {cid}")
    edit(nb, cid, match[0], newprefix + match[0][len(oldprefix):])

def part_ii(nb):
    changes = [
        ("classical-001", "## 実行方法と設定", "## 0. 設定と実行方法"),
        ("basics-ii-waves", "### 特性速度は何の速さか", "### 1.4 特性速度は何の速さか"),
        ("classical-006", "### 共通の方程式と保存形", "## 2. 保存変数と計算条件\n\n### 2.1 共通の方程式と保存形"),
        ("basics-ii-variables", "### なぜ密度・速度・圧力を、そのまま三つ保存しないのか", "### 2.2 なぜ密度・速度・圧力を、そのまま三つ保存しないのか"),
        ("basics-ii-variables", "### 圧力は流束にも現れる", "### 2.3 圧力は流束にも現れる"),
        ("basics-ii-boundaries", "### 初期条件と境界条件を区別する", "### 2.4 初期条件と境界条件を区別する"),
        ("classical-007", "### 共通のテスト問題と初期条件", "### 2.5 共通のテスト問題と初期条件"),
        ("classical-008", "### 厳密解と比較の基準", "### 2.6 厳密解と比較の基準"),
        ("classical-009", "### 時間刻み・破綻・保存則", "### 2.7 時間刻み・破綻・保存則"),
        ("classical-010", "### 評価指標", "### 2.8 評価指標"),
        ("classical-011", "### 計算本体の自己検証", "### 2.9 計算本体の自己検証"),
        ("classical-013", ("# 第II部：古典的手法 — LFとLW", "## LFとLWの数値解法"), "## 3. LFとLWの数値解法"),
        ("basics-ii-lflw", "### LF・LWでは、どの操作を比較するのか", "### 3.1 LF・LWでは、どの操作を比較するのか"),
        ("classical-026", "### 第II部のまとめ", "## まとめ"),
    ]
    for cid, old, new in changes:
        edit(nb, cid, old, new)
    for k in range(6):
        numbered(nb, f"classical-{14+2*k:03d}", f"### II.{k+1} ", f"### 3.{k+2} ")
    index = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "basics-ii-variables")
    equation = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "classical-006")
    if index != equation + 1:
        cell = nb["cells"].pop(index)
        equation = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "classical-006")
        nb["cells"].insert(equation + 1, cell)

def part_iii(nb):
    changes = [
        ("modern-001", "## 実行方法と設定", "## 0. 設定と実行方法"),
        ("basics-iii-flux", "### そもそも、なぜ境界の流束が必要なのか", "## 1. 保存則と計算の準備\n\n### 1.1 そもそも、なぜ境界の流束が必要なのか"),
        ("modern-004", "### 共通の方程式と保存形", "### 1.2 共通の方程式と保存形"),
        ("modern-005", "### 共通のテスト問題と初期条件", "### 1.3 共通のテスト問題と初期条件"),
        ("modern-006", "### 厳密解と比較の基準", "### 1.4 厳密解と比較の基準"),
        ("modern-007", "### 時間刻み・破綻・保存則", "### 1.5 時間刻み・破綻・保存則"),
        ("modern-008", "### 評価指標", "### 1.6 評価指標"),
        ("modern-009", "### 計算本体の自己検証", "### 1.7 計算本体の自己検証"),
        ("basics-iii-riemann", "## リーマン問題からGodunov型スキームまで", "## 2. リーマン問題からGodunov型スキームまで"),
        ("modern-011", ("# 第III部：現代の実用的手法 — 数値流束の比較", "## Roe・LLF・HLL・HLLCの数値流束比較"), "## 3. 近似リーマンソルバーの数値流束比較"),
        ("basics-iii-solvers", "### 4種類を、まず考え方で区別する", "### 3.1 4種類を、まず考え方で区別する"),
        ("modern-012", "### Roe流束とエントロピー補正", "### 3.2 Roe流束とエントロピー補正"),
        ("basics-iii-hll", "### HLL・HLLCの分岐を読むための準備", "### 3.3 HLL・HLLCの分岐を読むための準備"),
        ("modern-013", "### HLLとHLLC：外側波速を共通にして比較する", "### 3.4 HLLとHLLC：外側波速を共通にして比較する"),
        ("modern-026", "### 第III部のまとめ", "## まとめ"),
    ]
    for cid, old, new in changes:
        edit(nb, cid, old, new)
    for k in range(1, 5):
        numbered(nb, "basics-iii-riemann", f"### {k}. ", f"### 2.{k} ")
    for k in range(6):
        numbered(nb, f"modern-{14+2*k:03d}", f"### III.{k+1} ", f"### 3.{k+5} ")

def verify(nb, part):
    headings = [line for c in nb["cells"] if c["cell_type"] == "markdown" for line in txt(c).splitlines() if re.match(r"^#{1,3} ", line)]
    first = [line for line in headings if line.startswith("# ")]
    assert len(first) == 1 and first[0].startswith(f"# 第{part}部"), first
    major, minor, active = [], {}, None
    for line in headings:
        if line.startswith("## "):
            m = re.match(r"^## (\d+)\. ", line)
            active = int(m.group(1)) if m else None
            if m:
                major.append(active)
        if line.startswith("### "):
            m = re.match(r"^### (\d+)\.(\d+) ", line)
            if m:
                a, b = map(int, m.groups())
                assert a == active, (a, active, line)
                minor.setdefault(a, []).append(b)
    assert major == [0, 1, 2, 3], major
    for a, values in minor.items():
        assert values == list(range(1, len(values)+1)), (a, values)
    assert not any(re.match(r"^### (II|III)\.\d+", line) for line in headings)
    return len(headings)

def main():
    for name, part, operation in [
        ("02_euler_classical.ipynb", "II", part_ii),
        ("03_euler_riemann_solvers.ipynb", "III", part_iii)
    ]:
        path = ROOT / name
        nb = json.loads(path.read_text(encoding="utf-8"))
        original_other = [c for c in nb["cells"] if c["cell_type"] != "markdown"]
        original_meta = json.dumps(nb["metadata"], sort_keys=True, ensure_ascii=False)
        ids = sorted(c.get("id") for c in nb["cells"] if c["cell_type"] == "markdown")
        operation(nb)
        count = verify(nb, part)
        assert original_other == [c for c in nb["cells"] if c["cell_type"] != "markdown"]
        assert original_meta == json.dumps(nb["metadata"], sort_keys=True, ensure_ascii=False)
        assert ids == sorted(c.get("id") for c in nb["cells"] if c["cell_type"] == "markdown")
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {count} headings; unique chapter title; sequential H2/H3; stored outputs unchanged")

if __name__ == "__main__":
    main()
