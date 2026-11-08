"""Express derivatives in instructional Markdown as explicit fractions.

Only Markdown cell sources are modified. Code, stored figures, tables,
animations, execution counts, and cell IDs are preserved.
"""
from pathlib import Path
import json
import re

REPLACEMENTS = (
    (r"\boldsymbol F(\boldsymbol U)_x", r"\frac{\partial\boldsymbol F(\boldsymbol U)}{\partial x}"),
    (r"\boldsymbol U_t", r"\frac{\partial\boldsymbol U}{\partial t}"),
    (r"\left(\frac{q^2}{2}\right)_x", r"\frac{\partial}{\partial x}\left(\frac{q^2}{2}\right)"),
    (r"f(q)_x", r"\frac{\partial f(q)}{\partial x}"),
    (r"q_{xx}", r"\frac{\partial^2 q}{\partial x^2}"),
    (r"\partial_tw_\pm", r"\frac{\partial w_\pm}{\partial t}"),
    (r"\partial_xw_\pm", r"\frac{\partial w_\pm}{\partial x}"),
    (r"\partial_tw_0", r"\frac{\partial w_0}{\partial t}"),
    (r"\partial_xw_0", r"\frac{\partial w_0}{\partial x}"),
    (r"Q_0'=q_0", r"\frac{dQ_0}{d\xi}=q_0"),
    (r"q_0'(\xi)", r"\frac{dq_0}{d\xi}"),
    (r"q_0'", r"\frac{dq_0}{d\xi}"),
    (r"-p_x/\rho", r"-\frac{1}{\rho}\frac{\partial p}{\partial x}"),
    (r"q_t", r"\frac{\partial q}{\partial t}"),
    (r"q_x", r"\frac{\partial q}{\partial x}"),
    (r"u_t", r"\frac{\partial u}{\partial t}"),
    (r"u_x", r"\frac{\partial u}{\partial x}"),
    (r"p_x", r"\frac{\partial p}{\partial x}"),
    (r"\rho_t", r"\frac{\partial\rho}{\partial t}"),
    (r"\rho_x", r"\frac{\partial\rho}{\partial x}"),
    (r"W_t", r"\frac{\partial W}{\partial t}"),
    (r"W_x", r"\frac{\partial W}{\partial x}"),
    (r"\eta_t", r"\frac{\partial\eta}{\partial t}"),
    (r"\psi_x", r"\frac{\partial\psi}{\partial x}"),
    (r"\partial_t", r"\frac{\partial}{\partial t}"),
    (r"\partial_x", r"\frac{\partial}{\partial x}"),
    (r"D/Dt", r"\frac{D}{Dt}"),
    (r"Dp/Dt", r"\frac{Dp}{Dt}"),
    (r"\frac{\frac{\partial p}{\partial x}}{\rho}", r"\frac{1}{\rho}\frac{\partial p}{\partial x}"),
    (r"\min \frac{dq_0}{d\xi}", r"\min_{\xi} \frac{dq_0}{d\xi}"),
    (r"dx/dt", r"\frac{dx}{dt}"),
    (r"dq/dt", r"\frac{dq}{dt}"),
    (r"dt/dx", r"\frac{dt}{dx}"),
)

# The temporal convergence-order symbol p_t is not a derivative.
SHORTHAND = re.compile(r"(?<![A-Za-z])(q_t|q_x|q_\{xx\}|u_t|u_x|p_x|\\rho_t|\\rho_x|W_t|W_x|\\eta_t|\\psi_x|\\partial_[tx]|\\boldsymbol U_t|f\(q\)_x|\\boldsymbol F\(\\boldsymbol U\)_x)")


def convert(markdown):
    for old, new in REPLACEMENTS:
        markdown = markdown.replace(old, new)
    return markdown


def normalize(path):
    with path.open(encoding="utf-8") as stream:
        notebook = json.load(stream)
    changed = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = cell["source"]
        before = "".join(source) if isinstance(source, list) else source
        after = convert(before)
        match = SHORTHAND.search(after)
        if match:
            raise ValueError(f"Unconverted derivative in {path}: {match.group(0)}")
        if after != before:
            changed += 1
            cell["source"] = after.splitlines(keepends=True) if isinstance(source, list) else after
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def main():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "notebooks").glob("0[1-5]_*.ipynb"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five notebooks; found {len(paths)}")
    for path in paths:
        print(f"{path.name}: changed {normalize(path)} Markdown cells")


if __name__ == "__main__":
    main()
