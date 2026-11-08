"""Small, explicit experiments for conservation, Riemann fluxes and time error."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import euler1d_core as core
import euler1d_mol as mol
import scalar1d as sc
import tvd_study as tvd
from euler1d_study import show_table

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
STYLES = ("-", "--", "-.", ":")


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def front_position(x, q, level):
    """Linear interpolation of a single descending crossing; never use total mass."""
    indices = np.flatnonzero((q[:-1] >= level) & (q[1:] < level))
    if len(indices) != 1:
        raise ValueError(f"Expected one descending crossing, found {len(indices)}")
    i = indices[0]
    return float(x[i] + (level - q[i]) * (x[i + 1] - x[i]) / (q[i + 1] - q[i]))


def nonconservative_comparison(cfl=0.4, sizes=(100, 400, 1600), t_end=0.4):
    """Positive Burgers shocks: conservative Godunov vs the advective-form counterexample."""
    if not 0 < cfl <= 1 or t_end <= 0 or t_end >= 0.6:
        raise ValueError("Require 0<CFL<=1 and 0<T<0.6 to keep both shocks inside the domain.")
    rows, finest = [], []
    for left, right in ((1.0, 0.0), (2.0, 1.0)):
        p = replace(sc.CASES["burgers_shock"], left=left, right=right, final_time=t_end)
        speed_exact = 0.5 * (left + right)
        for n in sizes:
            run = sc.simulate(p, "Godunov", n=int(n), cfl=cfl)
            if run["status"] != "ok":
                raise RuntimeError(run["reason"])
            x, q = run["x"], run["q0"].copy()
            dx, t = float(run["edges"][1] - run["edges"][0]), 0.0
            while t < t_end:
                dt = min(cfl * dx / np.max(q), t_end - t)
                q -= dt / dx * q * (q - np.r_[left, q[:-1]])
                t = min(t_end, t + dt)
            for label, values in (("Godunov", run["q"]), ("Nonconservative", q)):
                pos = front_position(x, values, 0.5 * (left + right))
                mass_change = dx * np.sum(values - run["q0"])
                expected_change = t_end * 0.5 * (left**2 - right**2)
                rows.append(dict(left=left, right=right, N=int(n), method=label, front_x=pos, measured_speed=(pos-p.x0)/t_end, exact_speed=speed_exact, balance=mass_change-expected_change))
            if n == sizes[-1]:
                finest.append((p, x, run["q"], q))
    show_table(rows)
    for p, x, conservative, nonconservative in finest:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        xp = np.linspace(*p.bounds, 2001)
        ax.plot(xp, sc.exact_pointwise(p, xp, t_end), color="black", lw=1.2, label="Entropy solution")
        ax.plot(x, conservative, color=COLORS[0], label="Godunov")
        ax.plot(x, nonconservative, color=COLORS[1], ls="--", label="Nonconservative")
        ax.set(xlim=(-0.15, 0.8), xlabel="$x$", ylabel="$q$", title=f"Burgers shock ({p.left:g}, {p.right:g}), N={len(x)}, t={t_end:g}")
        ax.legend(); fig.tight_layout(); plt.show(); plt.close(fig)
    return rows


def roe_entropy_comparison(cfg):
    p = core.PROBLEMS["transonic"]
    runs = [core.simulate(p, "Roe", n=cfg["n_basic"], gamma=cfg["gamma"], cfl=cfg["cfl"], order=1, roe_entropy=e, wave_speeds=cfg["wave_speeds"]) for e in (0.0, 0.1)]
    structure = core.riemann_structure(p.left, p.right, cfg["gamma"])
    fan = p.x0 + p.final_time * np.asarray(structure["left_speeds"])
    rows = []
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    xp = np.linspace(*p.bounds, 2001)
    ax.plot(xp, core.exact_pointwise(p, xp, p.final_time, cfg["gamma"])[:, 0], color="black", label="Exact (pointwise)")
    for j, r in enumerate(runs):
        if r.status != "ok":
            raise RuntimeError(r.fail_reason)
        rho = r.U[:, 0]
        inside = (r.x[:-1] >= fan[0]) & (r.x[1:] <= fan[1])
        jump = float(np.max(np.abs(np.diff(rho)[inside])))
        rows.append(dict(entropy_fix=r.roe_entropy, status=r.status, fan_left=float(fan[0]), fan_right=float(fan[1]), max_adjacent_jump_in_fan=jump, E_total=core.metrics(r)["L1_rho"], rho_min=float(rho.min())))
        ax.plot(r.x, rho, color=COLORS[j], ls=STYLES[j], marker=("o", "s")[j], markevery=4, ms=3, mfc="none", label=f"Roe, entropy fix = {r.roe_entropy:g}")
    ax.set(xlim=(max(p.bounds[0], fan[0]-0.08), min(p.bounds[1], fan[1]+0.08)), xlabel="$x$", ylabel="Density", title=f"Transonic rarefaction, N={cfg['n_basic']}")
    ax.legend(); fig.tight_layout(); plt.show(); plt.close(fig)
    show_table(rows)
    return runs, rows


def sod_zoom(runs):
    p = core.PROBLEMS["sod"]
    for bounds, title in (((0.6, 0.75), "Contact region"), ((0.82, 0.88), "Shock region")):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        xp = np.linspace(*bounds, 1001)
        ax.plot(xp, core.exact_pointwise(p, xp, p.final_time, runs[0].gamma)[:, 0], color="black", label="Exact (pointwise)")
        for j, r in enumerate(runs):
            if r.status == "ok":
                ax.plot(r.x, r.U[:, 0], color=COLORS[j], ls=STYLES[j], marker=("o", "s", "^", "D")[j], ms=3, mfc="none", label=r.label)
        ax.set(xlim=bounds, xlabel="$x$", ylabel="Density", title=f"Sod: {title}, N={runs[0].n}")
        ax.legend(); fig.tight_layout(); plt.show(); plt.close(fig)


def tv_shape_diagnostics(n=100, cfl=0.4, t_end=0.5):
    edges = np.linspace(0, 1, n+1)
    q0 = sc.exact_cell_average(sc.CASES["advection_box"], edges, 0.0)
    rows = []
    for ti in tvd.INTEGRATORS:
        r = tvd.simulate(q0, "minmod", ti, cfl, t_end)
        states = r["states"]
        qmin, qmax = states.min(axis=1), states.max(axis=1)
        excess = tvd.total_variation(states) - 2*(qmax-qmin)
        rows.append(dict(integrator=ti, TV_final=float(r["tv"][-1]), q_min_final=float(qmin[-1]), q_max_final=float(qmax[-1]), max_TV_minus_2range=float(excess.max()), min_TV_minus_2range=float(excess.min()), mixed_length_final=float(np.count_nonzero((states[-1]>0.1)&(states[-1]<0.9))/n)))
    show_table(rows)
    return rows


def forward_euler_interpretation(cfg):
    """Compare the measured error and layer width, without ranking methods universally."""
    import euler1d_experiments as ex
    rows = []
    for key in ("entropy", "contact", "sod"):
        for flux in cfg["fluxes"]:
            for run in ex.group_runs(key, flux, cfg, "time"):
                m = mol.metrics(run)
                rows.append(dict(problem=key, flux=mol.flux_label(flux), integrator=run.time_integrator, status=run.status, E_total=m["L1_rho"], mixed_length=m["mixed_length"], overshoot=m["rho_overshoot"], undershoot=m["rho_undershoot"]))
    show_table(rows)
    return rows


def courant_amplitudes(n=200, distance=1.0):
    """Analytic amplification for a resolved sine; shorten only the final step."""
    theta, rows = 2*np.pi/n, []
    for c in (0.4, 0.17):
        full = int(np.floor(distance*n/c))
        remainder = distance*n - full*c
        for name in ("Upwind", "LF"):
            def gain(cc):
                return abs(1-cc*(1-np.exp(-1j*theta))) if name == "Upwind" else abs(np.cos(theta)-1j*cc*np.sin(theta))
            amp = gain(c)**full
            if remainder > 1e-10:
                amp *= gain(remainder)
            rows.append(dict(method=name, C_adv=c, N=n, distance=distance, amplitude_ratio=float(amp)))
    show_table(rows)
    return rows


def run_tests():
    checks = []
    a, b = np.array([0.,0.,1.,1.]), np.array([.25,.75,.25,.75])
    assert tvd.total_variation(a) == tvd.total_variation(b) == 2
    assert tvd.total_variation(b) > 2*np.ptp(b)
    checks.append("TV does not determine extrema or layer width")
    rng = np.random.default_rng(8)
    for _ in range(100):
        q = rng.normal(size=40)
        assert tvd.total_variation(q) >= 2*np.ptp(q)-1e-13
    checks.append("periodic TV lower bound")
    q=np.array([2.,2.,1.,1.]); x=np.arange(4.)
    assert front_position(x,q,1.5)==1.5
    checks.append("front location interpolates a descending crossing")
    return checks
