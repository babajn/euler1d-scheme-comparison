"""Three-part teaching organization; evolution uses the unchanged euler1d_core.

Plotting and diagnostics are retained from the original comparison notebook.
The three parts classify teaching purposes, not mutually exclusive algorithms.
"""
import numpy as np
from euler1d_core import ALL_SOLVERS, RIEMANN_SOLVERS, PROBLEMS, simulate, metrics, exact_pointwise, exact_cell_average, cons_to_prim, project_reference

import csv
import html
import json
import platform
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# 岡部・伊藤パレット。方式ごとの色はすべての図で共通にする。
COLORS = {"LF": "#0072B2", "LW": "#D55E00", "Roe": "#009E73", "LLF": "#CC79A7", "HLL": "#E69F00", "HLLC": "#56B4E9"}
LINES = {"LF": "-", "LW": "-", "Roe": "--", "LLF": "-.", "HLL": ":", "HLLC": "--"}
MARKERS = {"LF": "s", "LW": "^", "Roe": "o", "LLF": "D", "HLL": "v", "HLLC": "x"}
plt.rcParams.update({"figure.dpi": 115, "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8.5, "mathtext.fontset": "dejavusans"})


def show_table(rows, columns=None, digits=4):
    """pandasを要求しない簡単なHTML表。データ数が多い表も横スクロールできる。"""
    if not rows:
        print("No rows.")
        return
    columns = list(rows[0]) if columns is None else list(columns)
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{value:.{digits}g}"
        return str(value)
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(format_value(row.get(c, '')))}</td>" for c in columns) + "</tr>" for row in rows)
    style = "<style>.euler-table{border-collapse:collapse;font-size:12px}.euler-table th,.euler-table td{padding:5px 9px;border-bottom:1px solid #ccc;text-align:right}.euler-table th{background:#eee;color:#111}</style>"
    display(HTML(style + '<div style="overflow-x:auto"><table class="euler-table"><thead><tr>' + header + "</tr></thead><tbody>" + body + "</tbody></table></div>"))


def print_failures(results):
    for run in results:
        if run.status != "ok":
            print(f"{run.label}: stopped at last accepted t={run.t:.8g}; rejected trial ended at t={run.fail_time:.8g}.")
            print("  " + run.fail_reason)


def field_values(W, field, gamma):
    if field == "rho":
        return W[..., 0], r"Density $\rho$"
    if field == "u":
        return W[..., 1], r"Velocity $u$"
    if field == "p":
        return W[..., 2], r"Pressure $p$"
    if field == "e":
        return W[..., 2] / ((gamma - 1.0) * W[..., 0]), r"Specific internal energy $e$"
    raise ValueError("Choose rho, u, p, or e.")


def plot_profiles(results, fields=("rho",), reference=None, xlim=None, logarithmic=False, title_suffix=""):
    """各物理量を独立した図に描く。失敗した計算を最終時刻の比較図に混ぜない。"""
    complete = [r for r in results if r.status == "ok"]
    if not complete:
        print("No completed run to plot.")
        print_failures(results)
        return
    base = complete[0]
    if any(abs(r.t - base.t) > 1.0e-12 or r.problem != base.problem or r.gamma != base.gamma for r in complete):
        raise ValueError("All plotted solutions must have the same problem, time, and gamma.")
    for field in fields:
        fig = plt.figure(figsize=(9.8, 4.5))
        ax = fig.gca()
        if base.problem.key != "shu_osher":
            xp = np.linspace(*base.problem.bounds, 2401)
            Wp = exact_pointwise(base.problem, xp, base.t, base.gamma)
            yp, ylabel = field_values(Wp, field, base.gamma)
            ax.plot(xp, yp, color="#000000", lw=1.3, label="Exact (pointwise)", zorder=5)
        elif reference is not None:
            if reference.status != "ok" or abs(reference.t - base.t) > 1.0e-12:
                raise ValueError("The reference must be complete at the plotted time.")
            yp, ylabel = field_values(cons_to_prim(reference.U, reference.gamma), field, reference.gamma)
            ax.plot(reference.x, yp, color="#000000", lw=1.2, label=f"Numerical ref. N={reference.n}", zorder=5)
        for index, run in enumerate(complete):
            y, ylabel = field_values(cons_to_prim(run.U, run.gamma), field, run.gamma)
            stride = max(1, run.n // 22)
            ax.plot(run.x, y, color=COLORS[run.solver], ls=("-" if run.order == 2 else LINES[run.solver]), lw=1.25, marker=MARKERS[run.solver], ms=3.1, markevery=(index % stride, stride), mfc="none", label=run.label)
        ax.set(xlabel="$x$", ylabel=ylabel, xlim=base.problem.bounds if xlim is None else xlim)
        ax.set_title(f"{base.problem.title}: t={base.t:g}, N={base.n}" + ("\n" + title_suffix if title_suffix else ""))
        if logarithmic:
            ax.set_yscale("log")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    print_failures(results)


def plot_convergence(results, title):
    fig = plt.figure(figsize=(9.5, 4.7))
    ax = fig.gca()
    for solver in ALL_SOLVERS:
        selected = sorted([r for r in results if r.solver == solver and r.status == "ok"], key=lambda r: r.n)
        if not selected:
            continue
        n = np.array([r.n for r in selected])
        error = np.array([metrics(r)["L1_rho"] for r in selected])
        ax.loglog(n, error, color=COLORS[solver], marker=MARKERS[solver], ls=LINES[solver], ms=5, mfc="none", label=selected[0].label)
    ax.set(xlabel="Number of cells $N$", ylabel=r"Mean $L_1$ error in density", title=title)
    ax.grid(which="both", alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def convergence_rows(results):
    rows = []
    for solver in ALL_SOLVERS:
        subset = sorted([r for r in results if r.solver == solver], key=lambda r: r.n)
        previous = None
        for r in subset:
            row = dict(metrics(r), method=r.label)
            row["q_observed"] = np.nan if previous is None or row["status"] != "ok" else np.log(previous["L1_rho"] / row["L1_rho"]) / np.log(row["n"] / previous["n"])
            rows.append(row)
            previous = row
    return rows


def plot_pressure_minimum(results):
    fig = plt.figure(figsize=(9.5, 4.7))
    ax = fig.gca()
    for run in results:
        ax.plot(run.history[:, 0], run.history[:, 2], color=COLORS[run.solver], ls=LINES[run.solver], label=run.label)
        if np.isfinite(run.fail_min_p):
            ax.plot(run.fail_time, run.fail_min_p, marker="x", ms=9, color=COLORS[run.solver], ls="none")
    ax.axhline(0.0, color="#000000", lw=0.8)
    ax.set_yscale("symlog", linthresh=1.0e-3)
    ax.set(xlabel="Time", ylabel=r"Minimum pressure $p_{\min}$", title="Pressure positivity: lines = accepted states; crosses = rejected trials")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    plt.show()
    plt.close(fig)

PARTS = {
    "classical": {"solvers": ("LF", "LW"), "order": 1, "title": "Part I: classical methods"},
    "modern": {"solvers": RIEMANN_SOLVERS, "order": 1, "title": "Part II: practical fluxes, first-order baseline"},
    "high_accuracy": {"solvers": RIEMANN_SOLVERS, "order": 2, "title": "Part III: MUSCL + SSP-RK2"},
}
RUN_CACHE = {}


def get_run(key, solver, cfg, n=None, order=1, **overrides):
    """Cache only identical numerical settings; plotting never changes integration."""
    problem = PROBLEMS[key]
    parameters = dict(n=cfg["n_basic"] if n is None else n, gamma=cfg["gamma"], cfl=cfg["cfl"], order=order, limiter=cfg["limiter"], wave_speeds=cfg["wave_speeds"], roe_entropy=cfg["roe_entropy"])
    parameters.update(overrides)
    cache_key = (problem, solver, tuple(sorted(parameters.items())))
    if cache_key not in RUN_CACHE:
        RUN_CACHE[cache_key] = simulate(problem, solver, **parameters)
    return RUN_CACHE[cache_key]


def run_part(part, key, cfg, n=None):
    group = PARTS[part]
    results = [get_run(key, solver, cfg, n=n, order=group["order"]) for solver in group["solvers"]]
    print(group["title"] + " | " + key + ": " + ", ".join(r.label + "=" + r.status for r in results))
    return results


def summarize(results, reference=None, columns=None):
    columns = ("method", "n", "order", "status", "t_reached", "L1_rho", "rho_min", "p_min", "balance_max") if columns is None else columns
    rows = [dict(metrics(r, reference=reference), method=r.label) for r in results]
    show_table(rows, columns)
    return rows


def run_convergence(part, cfg):
    group = PARTS[part]
    results = [get_run("entropy", solver, cfg, n=n, order=group["order"]) for solver in group["solvers"] for n in cfg["n_convergence"]]
    plot_convergence(results, group["title"] + ": smooth entropy wave")
    show_table(convergence_rows(results), ("method", "order", "n", "L1_rho", "q_observed"))
    return results


def shu_reference(cfg):
    """Two numerical references, retained from the original notebook, not exact."""
    n1, n2 = cfg["n_reference"]
    if n2 <= n1 or n2 % n1 or n1 % cfg["n_shu_osher"]:
        raise ValueError("Use nested integer grids: comparison N -> reference N1 -> N2.")
    refs = [get_run("shu_osher", "HLLC", cfg, n=n, order=2, limiter=cfg["reference_limiter"]) for n in (n1, n2)]
    if any(r.status != "ok" for r in refs):
        raise RuntimeError("An incomplete numerical reference must not be used.")
    coarse, fine = refs
    delta = np.abs(coarse.U[:, 0] - project_reference(fine, coarse.edges)[:, 0])
    show_table([{"N1": n1, "N2": n2, "mean_abs_density_difference": float(delta.mean())}])
    print("This is a two-resolution numerical comparison, not an exact solution or a convergence proof.")
    return fine


def export_results(directory, cfg, reference=None):
    """Export raw solver IDs separately from display names and teaching parts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for i, run in enumerate(RUN_CACHE.values()):
        ref = reference if run.problem.key == "shu_osher" and reference is not None and reference.n > run.n and reference.n % run.n == 0 else None
        row = dict(metrics(run, ref), method=run.label)
        row["part"] = "classical" if run.solver in PARTS["classical"]["solvers"] else ("high_accuracy" if run.order == 2 else "modern")
        row["reference"] = "none: incomplete" if run.status != "ok" else ("exact cell averages" if run.problem.key != "shu_osher" else (f"numerical N={ref.n}" if ref is not None else "reference run"))
        rows.append(row)
        for key, value in {"x": run.x, "edges": run.edges, "U0": run.U0, "U": run.U, "history": run.history, "balance": run.balance, "boundary_exchange": run.boundary_exchange}.items():
            arrays[f"run_{i:03d}_{key}"] = value
    if not rows:
        return rows
    with (directory / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "states.npz", **arrays)
    (directory / "settings.json").write_text(json.dumps({"settings": cfg, "parts": PARTS, "python": sys.version, "numpy": np.__version__}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} distinct runs to {directory.resolve()}")
    return rows


class FrameRecorder:
    """Interpolate conserved variables for display only; no feedback to the integrator."""
    def __init__(self, times):
        self.times = np.asarray(times, dtype=float)
        if self.times.ndim != 1 or len(self.times) < 2 or self.times[0] != 0.0 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("Frame times must start at zero and increase strictly.")
        self.states = None
        self.left_time = np.full(len(self.times), np.nan)
        self.right_time = np.full(len(self.times), np.nan)
        self.weight = np.full(len(self.times), np.nan)
        self.previous_time = None
        self.previous_state = None
        self.next_frame = 0

    def __call__(self, t, U):
        if self.states is None:
            self.states = np.full((len(self.times),) + U.shape, np.nan)
            self.states[0] = U
            self.left_time[0] = self.right_time[0] = t
            self.weight[0] = 0.0
            self.next_frame = 1
        else:
            if t <= self.previous_time:
                raise ValueError("Accepted solver times must increase strictly.")
            while self.next_frame < len(self.times) and self.times[self.next_frame] <= t:
                j = self.next_frame
                target = self.times[j]
                w = (target - self.previous_time) / (t - self.previous_time)
                if not (0.0 <= w <= 1.0):
                    raise RuntimeError("A movie frame would require extrapolation.")
                # Preserve endpoint states exactly, including the final numerical solution.
                self.states[j] = U if target == t else (self.previous_state if target == self.previous_time else (1.0 - w) * self.previous_state + w * U)
                self.left_time[j], self.right_time[j], self.weight[j] = self.previous_time, t, w
                self.next_frame += 1
        self.previous_time = float(t)
        self.previous_state = U.copy()

    @property
    def valid(self):
        return np.isfinite(self.weight)




def make_part_animation(part, key, cfg, field="rho", n_frames=61):
    """Record existing integrators without dt clipping; return a single-axis movie.

    Shu-Osher movies remain in the original animation notebook; this helper uses
    exact reference solutions only. The final states match simulate without an observer.
    """
    from matplotlib.animation import FuncAnimation
    if key == "shu_osher":
        raise ValueError("Use the original animation notebook for Shu-Osher numerical-reference movies.")
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2.")
    group, problem = PARTS[part], PROBLEMS[key]
    times = np.linspace(0.0, problem.final_time, n_frames)
    runs, recorders = [], []
    for solver in group["solvers"]:
        recorder = FrameRecorder(times)
        run = simulate(problem, solver, n=cfg["n_basic"], gamma=cfg["gamma"], cfl=cfg["cfl"], order=group["order"], limiter=cfg["limiter"], wave_speeds=cfg["wave_speeds"], roe_entropy=cfg["roe_entropy"], observer=recorder)
        baseline = get_run(key, solver, cfg, order=group["order"])
        assert np.array_equal(run.U, baseline.U) and run.steps == baseline.steps and run.status == baseline.status
        assert run.t == baseline.t and np.array_equal(run.boundary_exchange, baseline.boundary_exchange)
        if run.status == "ok":
            assert recorder.valid.all() and np.array_equal(recorder.states[-1], run.U)
        runs.append(run)
        recorders.append(recorder)
    reference_U = np.stack([exact_cell_average(problem, runs[0].edges, t, cfg["gamma"]) for t in times])
    reference, ylabel = field_values(cons_to_prim(reference_U, cfg["gamma"]), field, cfg["gamma"])
    values = [field_values(cons_to_prim(rec.states, cfg["gamma"], check=False), field, cfg["gamma"])[0] for rec in recorders]
    finite = np.concatenate([reference.ravel()] + [v[r.valid].ravel() for v, r in zip(values, recorders)])
    lo, hi = float(finite.min()), float(finite.max())
    logscale = key == "double_rarefaction" and field in ("rho", "p")
    margin = 0.08 * max(hi - lo, 0.1 * max(abs(lo), abs(hi), 1.0))
    fig = plt.figure(figsize=(10.8, 5.5))
    ax = fig.add_axes((0.09, 0.24, 0.62, 0.60))
    ax.set(xlabel="$x$", ylabel=ylabel, xlim=problem.bounds, ylim=(lo / 1.5, hi * 1.5) if logscale else (lo - margin, hi + margin))
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.22)
    ref_line, = ax.plot(runs[0].x, reference[0], color="#000000", lw=1.6, label="Exact (cell-averaged U)")
    lines = []
    for run, value in zip(runs, values):
        line, = ax.plot(run.x, value[0], color=COLORS[run.solver], ls=LINES[run.solver], marker=MARKERS[run.solver], markevery=max(1, run.n // 20), ms=3, mfc="none", label=run.label)
        lines.append(line)
    ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1.0), frameon=False, fontsize=9)
    fig.text(0.09, 0.945, group["title"], fontsize=13)
    clock = fig.text(0.09, 0.89, "", fontsize=11)
    status = fig.text(0.09, 0.10, "", fontsize=9)
    fig.text(0.09, 0.045, "Display interpolation only; unchanged solver steps. Stopped solutions disappear.", fontsize=9)
    def update(j):
        clock.set_text(f"{problem.title} | N={cfg['n_basic']} | CFL={cfg['cfl']:g} | t={times[j]:.5f}")
        ref_line.set_ydata(reference[j])
        notes = []
        for run, rec, line, value in zip(runs, recorders, lines, values):
            line.set_ydata(value[j] if rec.valid[j] else np.full(run.n, np.nan))
            notes.append(run.label + (": active" if rec.valid[j] else f": STOPPED (last accepted t={run.t:.5g})"))
        status.set_text("\n".join(["; ".join(notes[:2]), "; ".join(notes[2:])]))
        return [ref_line, clock, status] + lines
    update(0)
    movie = FuncAnimation(fig, update, frames=len(times), interval=50, repeat=True, blit=False)
    return fig, movie
