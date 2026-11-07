"""Independent spatial/time experiments, plots, movies, and reproducible result export."""
from dataclasses import replace
from pathlib import Path
import csv
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import euler1d_core as core
import euler1d_mol as mol
from euler1d_study import show_table, field_values, FrameRecorder

SPATIAL_METHODS = {
    "constant": ("constant", "minmod"),
    "muscl_minmod": ("muscl", "minmod"),
    "muscl_mc": ("muscl", "mc"),
    "linear_unlimited": ("muscl", "unlimited"),
}
SPATIAL_LABELS = {"constant": "Piecewise constant", "muscl_minmod": "MUSCL-minmod", "muscl_mc": "MUSCL-MC", "linear_unlimited": "Linear, unlimited"}
# Okabe-Ito palette; each chart compares variants for a single numerical flux.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
STYLES = ("--", "-", "-.", ":")
MARKERS = ("o", "s", "^", "D")
MOVIE_PROBLEMS = tuple(key for key in core.PROBLEMS if key != "shu_osher")
CACHE = {}


def get_run(key, solver, cfg, reconstruction="constant", limiter="minmod", time_integrator="ssprk2", n=None, cfl=None, t_end=None, fixed_steps=None, observer=None):
    p = core.PROBLEMS[key]
    kw = dict(n=cfg["n_basic"] if n is None else n, gamma=cfg["gamma"], cfl=cfg["cfl"] if cfl is None else cfl, reconstruction=reconstruction, limiter=limiter, time_integrator=time_integrator, wave_speeds=cfg["wave_speeds"], roe_entropy=cfg["roe_entropy"], t_end=t_end, fixed_steps=fixed_steps)
    token = (p, solver, tuple(kw.items()))
    if observer is not None:
        return mol.simulate(p, solver, observer=observer, **kw)
    if token not in CACHE:
        CACHE[token] = mol.simulate(p, solver, **kw)
    return CACHE[token]


def variants(cfg, vary):
    if vary == "space":
        return [(name, dict(reconstruction=SPATIAL_METHODS[name][0], limiter=SPATIAL_METHODS[name][1], time_integrator=cfg["time_integrator"])) for name in cfg["spatial_methods"]]
    if vary == "time":
        return [(ti, dict(reconstruction=cfg["reconstruction"], limiter=cfg["limiter"], time_integrator=ti)) for ti in cfg["time_integrators"]]
    raise ValueError("vary must be space or time.")


def variant_label(run, vary):
    if vary == "time":
        return mol.TIME_LABELS[run.time_integrator]
    if run.reconstruction == "constant":
        return SPATIAL_LABELS["constant"]
    return "Linear, unlimited" if run.limiter == "unlimited" else "MUSCL-" + ("MC" if run.limiter == "mc" else run.limiter)


def group_runs(key, solver, cfg, vary, **kwargs):
    return [get_run(key, solver, cfg, **spec, **kwargs) for _, spec in variants(cfg, vary)]


def summarize(runs, reference=None):
    rows = [mol.metrics(r, reference) for r in runs]
    columns = ("solver", "reconstruction", "limiter", "time_integrator", "status", "t_reached", "L1_rho", "p_min", "balance_max", "n_rhs", "seconds")
    show_table(rows, columns)
    for r in runs:
        if r.status != "ok":
            print(r.label + ": " + r.fail_reason)
    return rows


def plot_group(runs, vary, fields=("rho",), reference=None):
    """One figure per field and flux; never overlay different completion times."""
    p, target = runs[0].problem, runs[0].target_time
    if any(r.problem != p or r.target_time != target or r.gamma != runs[0].gamma for r in runs):
        raise ValueError("Compare the same problem, gamma, and target time.")
    for field in fields:
        fig = plt.figure(figsize=(10.8, 4.9))
        ax = fig.add_axes((0.09, 0.16, 0.64, 0.69))
        if p.key != "shu_osher":
            xp = np.linspace(*p.bounds, 1601)
            exact, ylabel = field_values(core.exact_pointwise(p, xp, target, runs[0].gamma), field, runs[0].gamma)
            ax.plot(xp, exact, color="black", lw=1.6, label="Exact (pointwise)")
        elif reference is not None:
            if reference.status != "ok" or reference.t != target:
                raise ValueError("Incomplete reference or different reference time.")
            exact, ylabel = field_values(core.cons_to_prim(reference.U, reference.gamma), field, reference.gamma)
            ax.plot(reference.x, exact, color="black", lw=1.3, label=f"Numerical ref. N={reference.n}")
        for j, r in enumerate(runs):
            y, ylabel = field_values(core.cons_to_prim(r.U, r.gamma), field, r.gamma)
            label = variant_label(r, vary)
            if r.status != "ok":
                ax.plot([], [], color=COLORS[j], ls=STYLES[j], label=label + f" (stopped t={r.t:.3g})")
                continue
            stride = max(1, r.n // 24)
            ax.plot(r.x, y, color=COLORS[j], ls=STYLES[j], marker=MARKERS[j], ms=3, mfc="none", markevery=(j % stride, stride), lw=1.3, label=label)
        if p.key == "double_rarefaction" and field in ("rho", "p"):
            ax.set_yscale("log")
        ax.set(xlabel="$x$", ylabel=ylabel, xlim=p.bounds, title=f"{mol.flux_label(runs[0].solver, runs[0].wave_speeds)} | {p.title}\nN={runs[0].n}, t={target:g}")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1.0), frameon=False, fontsize=9)
        plt.show()
        plt.close(fig)


def spatial_convergence(cfg):
    rows = []
    for solver in cfg["fluxes"]:
        fig = plt.figure(figsize=(9.6, 4.8))
        ax = fig.add_axes((0.1, 0.17, 0.59, 0.67))
        for j, (name, spec) in enumerate(variants(cfg, "space")):
            sequence = [get_run("entropy", solver, cfg, n=n, **spec) for n in cfg["n_convergence"]]
            errors = [mol.metrics(r)["L1_rho"] for r in sequence]
            ax.loglog(cfg["n_convergence"], errors, color=COLORS[j], ls=STYLES[j], marker=MARKERS[j], label=SPATIAL_LABELS[name])
            previous = None
            for r, error in zip(sequence, errors):
                q = np.nan if previous is None else np.log(previous[1] / error) / np.log(r.n / previous[0])
                rows.append(dict(flux=mol.flux_label(solver, cfg["wave_speeds"]), reconstruction=name, n=r.n, L1_rho=error, q_observed=q, time_integrator=r.time_integrator))
                previous = r.n, error
        ax.set(xlabel="Number of cells N", ylabel="Mean density L1 error", title=mol.flux_label(solver, cfg["wave_speeds"]) + " | spatial refinement")
        ax.grid(which="both", alpha=0.22)
        ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1), frameon=False, fontsize=9)
        plt.show()
        plt.close(fig)
    show_table(rows)
    return rows


def spatial_time_check(cfg):
    """Estimate time sensitivity without claiming a pure spatial error measurement."""
    rows = []
    for solver in cfg["fluxes"]:
        for name, spec in variants(cfg, "space"):
            a, b = [get_run("entropy", solver, cfg, cfl=c, **spec) for c in (cfg["cfl"], cfg["cfl"] / 2)]
            delta = float(np.mean(np.abs(a.U[:, 0] - b.U[:, 0]))) if a.status == b.status == "ok" else np.nan
            error = mol.metrics(b)["L1_rho"]
            rows.append(dict(flux=mol.flux_label(solver, cfg["wave_speeds"]), reconstruction=name, density_change=delta, fine_dt_total_error=error, change_over_error=delta / error))
    show_table(rows)
    return rows


def temporal_convergence(cfg):
    """Refine time only against two same-grid SSP-RK3 references, one pair per flux."""
    rows, reference_rows = [], []
    counts = tuple(cfg["temporal_steps"])
    refs = tuple(cfg["reference_steps"])
    if any(not isinstance(k, int) or k < 1 for k in counts + refs) or any(b <= a for a, b in zip(counts, counts[1:])) or not counts[-1] < refs[0] < refs[1]:
        raise ValueError("Use increasing positive temporal_steps and two larger reference_steps.")
    kw = dict(n=cfg["temporal_n"], t_end=cfg["temporal_t_end"], cfl=cfg["fixed_cfl_limit"], reconstruction=cfg["temporal_reconstruction"], limiter=cfg["temporal_limiter"])
    for solver in cfg["fluxes"]:
        coarse, fine = [get_run("entropy", solver, cfg, time_integrator="ssprk3", fixed_steps=k, **kw) for k in refs]
        if coarse.status != "ok" or fine.status != "ok":
            raise RuntimeError("A time reference failed. Increase reference_steps; do not use incomplete references.")
        gap = float(np.mean(np.abs(coarse.U[:, 0] - fine.U[:, 0])))
        best_error = np.inf
        fig = plt.figure(figsize=(9.6, 4.8))
        ax = fig.add_axes((0.1, 0.17, 0.61, 0.67))
        for j, ti in enumerate(cfg["time_integrators"]):
            previous, errors = None, []
            for k in counts:
                r = get_run("entropy", solver, cfg, time_integrator=ti, fixed_steps=k, **kw)
                error = float(np.mean(np.abs(r.U[:, 0] - fine.U[:, 0]))) if r.status == "ok" else np.nan
                total_error = mol.metrics(r)["L1_rho"]
                dt = cfg["temporal_t_end"] / k
                q = np.nan if previous is None or not np.isfinite(error) or error <= 0 else np.log(previous[1] / error) / np.log(previous[0] / dt)
                rows.append(dict(flux=mol.flux_label(solver, cfg["wave_speeds"]), time_integrator=ti, n=r.n, dt=dt, steps=k, status=r.status, E_time=error, E_total=total_error, q_time=q, n_rhs=r.n_rhs, seconds=r.elapsed, max_cfl=r.max_cfl))
                errors.append(error)
                previous = dt, error
                if np.isfinite(error):
                    best_error = min(best_error, error)
            ax.loglog(cfg["temporal_t_end"] / np.asarray(counts), errors, color=COLORS[j], ls=STYLES[j], marker=MARKERS[j], label=mol.TIME_LABELS[ti])
        ratio = gap / best_error if best_error > 0 and np.isfinite(best_error) else np.nan
        reference_rows.append(dict(flux=mol.flux_label(solver, cfg["wave_speeds"]), reference_steps_coarse=refs[0], reference_steps_fine=refs[1], reference_gap=gap, minimum_measured_error=best_error, gap_over_error=ratio, reference_resolved=bool(np.isfinite(ratio) and ratio < 0.05)))
        if not np.isfinite(ratio) or ratio >= 0.05:
            warnings.warn("Reference gap is not small relative to the measured temporal error: " + solver)
        ax.set(xlabel="Time step", ylabel="Density L1 error vs same-grid reference", title=mol.flux_label(solver, cfg["wave_speeds"]) + " | temporal refinement")
        ax.grid(which="both", alpha=0.22)
        ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1), frameon=False)
        plt.show()
        plt.close(fig)
    show_table(reference_rows)
    show_table(rows, ("flux", "time_integrator", "steps", "dt", "E_time", "q_time", "E_total", "status", "n_rhs"))
    return rows, reference_rows


def cost_table(time_rows, tolerances=(1e-3, 1e-5, 1e-7)):
    """Minimum work among the computed samples, not an optimization or universal ranking."""
    rows = []
    for flux in dict.fromkeys(r["flux"] for r in time_rows):
        for tol in tolerances:
            for ti in mol.TIME_INTEGRATORS:
                valid = [r for r in time_rows if r["flux"] == flux and r["time_integrator"] == ti and r["status"] == "ok" and r["E_time"] <= tol]
                best = min(valid, key=lambda r: r["n_rhs"]) if valid else None
                rows.append(dict(flux=flux, tolerance=tol, time_integrator=ti, n_rhs=best["n_rhs"] if best else None, seconds=best["seconds"] if best else None, achieved_error=best["E_time"] if best else None))
    show_table(rows)
    return rows


def comparison_movie(key, solver, cfg, vary):
    """Only display interpolates the accepted states. Stopped runs disappear."""
    if key not in MOVIE_PROBLEMS:
        raise ValueError("movie_problem choices: " + ", ".join(MOVIE_PROBLEMS))
    if cfg["movie_frames"] < 2:
        raise ValueError("movie_frames must be >=2.")
    target = core.PROBLEMS[key].final_time
    times = np.linspace(0.0, target, cfg["movie_frames"])
    runs, recorders = [], []
    for _, spec in variants(cfg, vary):
        recorder = FrameRecorder(times)
        run = get_run(key, solver, cfg, observer=recorder, **spec)
        baseline = get_run(key, solver, cfg, **spec)
        np.testing.assert_array_equal(run.U, baseline.U)
        assert run.steps == baseline.steps and run.t == baseline.t and np.array_equal(run.boundary_exchange, baseline.boundary_exchange)
        runs.append(run)
        recorders.append(recorder)
    field = cfg["movie_field"]
    exact = np.stack([core.exact_cell_average(runs[0].problem, runs[0].edges, t, cfg["gamma"]) for t in times])
    reference, ylabel = field_values(core.cons_to_prim(exact, cfg["gamma"]), field, cfg["gamma"])
    values = [field_values(core.cons_to_prim(rec.states, cfg["gamma"], check=False), field, cfg["gamma"])[0] for rec in recorders]
    finite = np.concatenate([reference.ravel()] + [v[r.valid].ravel() for v, r in zip(values, recorders)])
    lo, hi = finite.min(), finite.max()
    margin = 0.08 * max(hi - lo, 0.1)
    log = key == "double_rarefaction" and field in ("rho", "p")
    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_axes((0.09, 0.29, 0.62, 0.52))
    ax.set(xlabel="$x$", ylabel=ylabel, xlim=runs[0].problem.bounds, ylim=(lo / 1.5, hi * 1.5) if log else (lo - margin, hi + margin))
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.22)
    ref_line, = ax.plot(runs[0].x, reference[0], color="black", lw=1.5, label="Exact (cell-averaged U)")
    lines = [ax.plot(r.x, val[0], color=COLORS[j], ls=STYLES[j], lw=1.4, label=variant_label(r, vary))[0] for j, (r, val) in enumerate(zip(runs, values))]
    ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1), frameon=False, fontsize=9)
    fig.text(0.09, 0.95, mol.flux_label(solver, cfg["wave_speeds"]) + " | " + runs[0].problem.title, fontsize=13)
    clock = fig.text(0.09, 0.89, "", fontsize=11)
    status = fig.text(0.09, 0.14, "", fontsize=9, va="top")
    fig.text(0.09, 0.025, "Display interpolation only. Fixed axes; stopped solutions disappear.", fontsize=9)
    def update(j):
        clock.set_text(f"N={runs[0].n} | t={times[j]:.5f} / {target:g}")
        ref_line.set_ydata(reference[j])
        notes = []
        for line, r, rec, val in zip(lines, runs, recorders, values):
            line.set_ydata(val[j] if rec.valid[j] else np.full(r.n, np.nan))
            notes.append(variant_label(r, vary) + (": active" if rec.valid[j] else f": STOPPED (last t={r.t:.4g})"))
        status.set_text("\n".join(notes))
        return [ref_line, clock, status] + lines
    update(0)
    return fig, FuncAnimation(fig, update, frames=len(times), interval=60, repeat=True, blit=False)


def export_results(directory, cfg, tables=None, reference=None):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for i, r in enumerate(CACHE.values()):
        rows.append(dict(run_id=i, **mol.metrics(r, reference if r.problem.key == "shu_osher" else None)))
        for name in ("x", "edges", "U0", "U", "history", "dt_history", "boundary_exchange", "balance"):
            arrays[f"run_{i:04d}_{name}"] = getattr(r, name)
    def write_csv(name, data):
        if data:
            with (folder / name).open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    write_csv("metrics.csv", rows)
    for name, data in (tables or {}).items():
        write_csv(name + ".csv", data)
    np.savez_compressed(folder / "states.npz", **arrays)
    (folder / "settings.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} runs: {folder}")
    return rows


def validate_config(cfg, vary):
    for name in ("fluxes", "movie_fluxes"):
        seq = cfg[name]
        if not seq or len(set(seq)) != len(seq) or any(s not in core.RIEMANN_SOLVERS for s in seq):
            raise ValueError(name + " must be a nonempty selection from Roe, LLF, HLL, HLLC.")
    if cfg["movie_problem"] not in MOVIE_PROBLEMS or cfg["movie_field"] not in ("rho", "u", "p", "e"):
        raise ValueError("Choose movie_problem and movie_field from the documented table.")
    if not isinstance(cfg["movie_frames"], int) or cfg["movie_frames"] < 2:
        raise ValueError("movie_frames must be an integer >=2.")
    if vary == "space":
        seq = cfg["spatial_methods"]
        if not seq or len(set(seq)) != len(seq) or any(s not in SPATIAL_METHODS for s in seq):
            raise ValueError("Invalid spatial_methods.")
        if cfg["time_integrator"] not in mol.TIME_INTEGRATORS:
            raise ValueError("Invalid time_integrator.")
    else:
        seq = cfg["time_integrators"]
        if not seq or len(set(seq)) != len(seq) or any(s not in mol.TIME_INTEGRATORS for s in seq):
            raise ValueError("Invalid time_integrators.")
        for prefix in ("", "temporal_"):
            if cfg[prefix + "reconstruction"] not in mol.RECONSTRUCTIONS or cfg[prefix + "limiter"] not in mol.LIMITERS:
                raise ValueError("Invalid reconstruction or limiter.")
