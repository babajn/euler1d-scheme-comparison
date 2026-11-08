"""Periodic scalar TVD diagnostics; independent of the Euler production updates.

The variable is a scalar cell average on [0, 1). No clipping, filtering,
step rejection or artificial positivity repair is applied in these experiments.
"""
import numpy as np
from euler1d_core import minmod

RECONSTRUCTIONS = ("constant", "minmod", "mc", "unlimited")
INTEGRATORS = ("forward_euler", "ssprk2", "ssprk3")


def total_variation(q):
    """Sum adjacent differences on the last axis, INCLUDING the periodic wrap."""
    q = np.asarray(q, dtype=float)
    if q.ndim < 1 or q.shape[-1] < 2 or not np.all(np.isfinite(q)):
        raise ValueError("Expected finite scalar states with at least two cells.")
    return np.sum(np.abs(q - np.roll(q, 1, axis=-1)), axis=-1)


def slopes(q, reconstruction):
    """Cell-width-scaled increments, not derivatives."""
    dm, dp = q - np.roll(q, 1), np.roll(q, -1) - q
    if reconstruction == "constant":
        return np.zeros_like(q)
    if reconstruction == "minmod":
        return minmod(dm, dp)
    if reconstruction == "mc":
        return minmod(0.5 * (dm + dp), minmod(2.0 * dm, 2.0 * dp))
    if reconstruction == "unlimited":
        return 0.5 * (dm + dp)
    raise ValueError(f"Choose reconstruction from {RECONSTRUCTIONS}.")


def euler_step(q, courant, reconstruction):
    """Upwind flux with a signed Courant number and forward Euler time step."""
    s = slopes(q, reconstruction)
    left, right = q + 0.5 * s, np.roll(q - 0.5 * s, -1)
    flux_dt_dx = np.maximum(courant, 0.0) * left + np.minimum(courant, 0.0) * right
    return q - (flux_dt_dx - np.roll(flux_dt_dx, 1))


def step(q, courant, reconstruction, integrator):
    if integrator not in INTEGRATORS:
        raise ValueError(f"Choose integrator from {INTEGRATORS}.")
    q1 = euler_step(q, courant, reconstruction)
    if integrator == "forward_euler":
        return q1
    y2 = euler_step(q1, courant, reconstruction)
    if integrator == "ssprk2":
        return 0.5 * q + 0.5 * y2
    q2 = 0.75 * q + 0.25 * y2
    return q / 3.0 + (2.0 / 3.0) * euler_step(q2, courant, reconstruction)


def simulate(q0, reconstruction="minmod", integrator="ssprk2", cfl=0.4, t_end=0.25, speed=1.0):
    """Fixed equal steps; CFL>1 allowed ONLY for deliberate counterexamples."""
    q = np.asarray(q0, dtype=float).copy()
    if q.ndim != 1 or len(q) < 4 or not np.all(np.isfinite(q)):
        raise ValueError("q0 must be a finite 1D array with at least four cells.")
    if reconstruction not in RECONSTRUCTIONS or integrator not in INTEGRATORS:
        raise ValueError("Unknown reconstruction or integrator.")
    if not np.isfinite([cfl, t_end, speed]).all() or cfl <= 0 or t_end <= 0 or speed == 0:
        raise ValueError("Require finite CFL>0, t_end>0 and nonzero speed.")
    nsteps = max(1, int(np.ceil(t_end * abs(speed) * len(q) / cfl - 1e-12)))
    courant = t_end * speed * len(q) / nsteps
    states = [q.copy()]
    for _ in range(nsteps):
        q = step(q, courant, reconstruction, integrator)
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("Nonfinite counterexample; shorten t_end or lower CFL.")
        states.append(q.copy())
    states = np.asarray(states)
    return {"times": np.linspace(0.0, t_end, nsteps + 1), "states": states, "tv": total_variation(states), "CFL": abs(courant), "reconstruction": reconstruction, "integrator": integrator}


def diagnostics(times, states, label):
    """A per-step test is distinct from comparison against initial TV."""
    times, states = np.asarray(times), np.asarray(states)
    if states.ndim != 2 or len(times) != len(states) or len(times) < 2:
        raise ValueError("Pass every accepted step, not interpolated movie frames.")
    tv = total_variation(states)
    tol = 1e-11 * max(1.0, float(tv.max()))
    increment = np.diff(tv)
    return {"method": label, "TV_initial": float(tv[0]), "TV_final": float(tv[-1]), "max_step_increase": max(0.0, float(increment.max())), "steps_with_increase": int(np.count_nonzero(increment > tol)), "TV_check_tol": tol, "max_mean_drift": float(np.max(np.abs(states.mean(axis=1) - states[0].mean()))), "q_min": float(states.min()), "q_max": float(states.max())}


def plot_histories(histories, title):
    """histories: (label, accepted times, states); one axis per figure."""
    import matplotlib.pyplot as plt
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    styles = ("-", "--", "-.", ":")
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for i, (label, times, states) in enumerate(histories):
        tv = total_variation(states)
        if tv[0] == 0:
            raise ValueError("Normalized TV needs a nonconstant initial state.")
        ax.plot(times, tv / tv[0], label=label, color=colors[i % len(colors)], ls=styles[i % len(styles)], lw=1.6)
    ax.axhline(1.0, color="#000000", lw=0.7, ls=":")
    ax.set(xlabel="Time", ylabel=r"$TV(q)/TV(q^0)$", title=title)
    ax.grid(alpha=0.2)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    lo, hi = ax.get_ylim()
    if hi - lo < 0.1:
        ax.set_ylim(min(lo, 0.95), max(hi, 1.05))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_tests():
    checks = []
    np.testing.assert_allclose(total_variation(np.array([0., 0., 1., 1.])), 2.0)
    checks.append("periodic endpoint is included")
    rng = np.random.default_rng(20261108)
    for speed in (-1.0, 1.0):
        q = rng.normal(size=32)
        np.testing.assert_allclose(step(q, speed, "constant", "forward_euler"), np.roll(q, int(speed)), atol=1e-14)
    checks.append("signed CFL=1 shift")
    # Random states supplement, but never replace, the analytic sufficient-condition proof.
    for rec in ("constant", "minmod", "mc"):
        limit = 1.0 if rec == "constant" else 0.5
        for ti in INTEGRATORS:
            for c in (-limit, -0.4, 0.4, limit):
                for _ in range(20):
                    q = rng.normal(size=64)
                    q1 = step(q, c, rec, ti)
                    assert total_variation(q1) <= total_variation(q) + 2e-12
                    np.testing.assert_allclose(q1.sum(), q.sum(), atol=2e-13)
    checks.append("TVD sufficient bounds and conservation in 720 random one-step tests")
    q = np.r_[np.zeros(10), np.ones(10), np.zeros(20)]
    assert total_variation(step(q, 0.4, "unlimited", "ssprk2")) > total_variation(q) + 1e-4
    for ti in INTEGRATORS:
        assert total_variation(step(q, 1.2, "constant", ti)) > total_variation(q) + 1e-4
    checks.append("unlimited and oversized-CFL counterexamples")
    return checks


def export_diagnostics(rows, path):
    from pathlib import Path
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
