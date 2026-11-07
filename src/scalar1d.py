"""Conservative finite-volume exercises for linear advection and inviscid Burgers.

Unknowns are cell averages. Exact references are averaged on the same cells.
The uncorrected Roe flux is provided only as an entropy-condition counterexample.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import numpy as np


@dataclass(frozen=True)
class ScalarProblem:
    key: str
    title: str
    equation: str
    profile: str
    bounds: tuple = (0.0, 1.0)
    boundary: str = "periodic"
    final_time: float = 1.0
    speed: float = 1.0
    mean: float = 1.0
    amplitude: float = 0.2
    left: float = 1.0
    right: float = 0.0
    x0: float = 0.0


CASES = {
    "advection_sine": ScalarProblem("advection_sine", "Linear advection: sine wave", "advection", "sine"),
    "advection_box": ScalarProblem("advection_box", "Linear advection: square pulse", "advection", "box", final_time=0.25),
    "burgers_sine": ScalarProblem("burgers_sine", "Burgers: smooth compression", "burgers", "sine", final_time=0.4),
    "burgers_shock": ScalarProblem("burgers_shock", "Burgers: shock (1, 0)", "burgers", "riemann", (-1.0, 1.0), "outflow", 0.4),
    "burgers_rarefaction": ScalarProblem("burgers_rarefaction", "Burgers: rarefaction (-1, 1)", "burgers", "riemann", (-1.0, 1.0), "outflow", 0.4, left=-1.0, right=1.0),
}
SCHEMES = {"advection": ("Upwind", "LF", "LW"), "burgers": ("Godunov", "LF", "LW", "Roe_no_efix")}
LABELS = {"Upwind": "Upwind (1st order)", "LF": "LF", "LW": "LW (Richtmyer)", "Godunov": "Godunov (1st order)", "Roe_no_efix": "Roe (no entropy fix)"}
COLORS = {"Upwind": "#009E73", "LF": "#0072B2", "LW": "#D55E00", "Godunov": "#CC79A7", "Roe_no_efix": "#E69F00"}
STYLES = {"Upwind": "--", "LF": "-", "LW": "-.", "Godunov": "--", "Roe_no_efix": ":"}


def breaking_time(problem):
    """First loss of smoothness for the periodic sinusoidal Burgers data."""
    if problem.equation != "burgers" or problem.profile != "sine" or problem.amplitude == 0.0:
        return np.inf
    return (problem.bounds[1] - problem.bounds[0]) / (2.0 * np.pi * abs(problem.amplitude))


def physical_flux(q, equation, speed=1.0):
    q = np.asarray(q, dtype=float)
    if equation == "advection":
        return speed * q
    if equation == "burgers":
        return 0.5 * q**2
    raise ValueError("Unknown scalar equation.")


def numerical_flux(left, right, scheme, equation, dt, dx, speed=1.0):
    """One shared flux per interface; LF and LLF must not be confused."""
    if equation not in SCHEMES or scheme not in SCHEMES[equation]:
        raise ValueError("Scheme is not defined for this equation.")
    if not (np.isfinite(dt) and np.isfinite(dx) and dt > 0.0 and dx > 0.0):
        raise ValueError("Positive finite dt and dx are required.")
    left, right = np.broadcast_arrays(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    fl, fr = physical_flux(left, equation, speed), physical_flux(right, equation, speed)
    if scheme == "LF":
        return 0.5 * (fl + fr) - 0.5 * (dx / dt) * (right - left)
    if scheme == "LW":
        half = 0.5 * (left + right) - 0.5 * (dt / dx) * (fr - fl)
        return physical_flux(half, equation, speed)
    if scheme == "Upwind":
        return np.where(speed >= 0.0, fl, fr)
    shock_speed = 0.5 * (left + right)
    if scheme == "Roe_no_efix":
        return 0.5 * (fl + fr) - 0.5 * abs(shock_speed) * (right - left)
    # Convex Burgers flux: rarefaction selects the state at x/t=0.
    rare_state = np.maximum(left, np.minimum(0.0, right))
    shock_state = np.where(shock_speed >= 0.0, left, right)
    state = np.where(left <= right, rare_state, shock_state)
    return 0.5 * state**2


def _feet(x, t, problem):
    """Invert x=xi+t*q0(xi) only while that map is one-to-one."""
    if t >= breaking_time(problem):
        raise ValueError("The characteristic inverse is not a post-shock entropy solution.")
    x = np.asarray(x, dtype=float)
    a, b = problem.bounds
    k = 2.0 * np.pi / (b - a)
    lo = x - t * (problem.mean + abs(problem.amplitude))
    hi = x - t * (problem.mean - abs(problem.amplitude))
    for _ in range(65):
        mid = 0.5 * (lo + hi)
        value = mid + t * (problem.mean + problem.amplitude * np.sin(k * (mid - a))) - x
        lo, hi = np.where(value <= 0.0, mid, lo), np.where(value > 0.0, mid, hi)
    return 0.5 * (lo + hi)


def exact_pointwise(problem, x, t):
    """Return None after breaking for sine data; no false 'exact' continuation."""
    x = np.asarray(x, dtype=float)
    a, b = problem.bounds
    length, k = b - a, 2.0 * np.pi / (b - a)
    if problem.profile == "sine":
        if problem.equation == "advection":
            feet = x - problem.speed * t
        elif t >= breaking_time(problem):
            return None
        else:
            feet = _feet(x, t, problem)
        return problem.mean + problem.amplitude * np.sin(k * (feet - a))
    if problem.profile == "box" and problem.equation == "advection":
        y = np.mod((x - a - problem.speed * t) / length, 1.0)
        return ((y >= 0.25) & (y < 0.45)).astype(float)
    if problem.profile == "riemann" and problem.equation == "burgers":
        ql, qr = problem.left, problem.right
        if t == 0.0:
            return np.where(x < problem.x0, ql, qr)
        if ql >= qr:
            return np.where(x < problem.x0 + 0.5 * (ql + qr) * t, ql, qr)
        return np.clip((x - problem.x0) / t, ql, qr)
    raise ValueError("Unsupported profile/equation combination.")


def exact_cell_average(problem, edges, t):
    """Analytic cell integrals, with bisection only for pre-shock sine feet."""
    edges = np.asarray(edges, dtype=float)
    dx = np.diff(edges)
    if np.any(dx <= 0.0) or t < 0.0:
        raise ValueError("Increasing edges and nonnegative time are required.")
    a, b = problem.bounds
    length, k = b - a, 2.0 * np.pi / (b - a)
    if problem.profile == "sine":
        if problem.equation == "advection":
            feet = edges - problem.speed * t
            primitive = problem.mean * feet - problem.amplitude * np.cos(k * (feet - a)) / k
        elif t >= breaking_time(problem):
            return None
        else:
            feet = _feet(edges, t, problem)
            q0 = problem.mean + problem.amplitude * np.sin(k * (feet - a))
            primitive = problem.mean * feet - problem.amplitude * np.cos(k * (feet - a)) / k + 0.5 * t * q0**2
        return np.diff(primitive) / dx
    if problem.profile == "box" and problem.equation == "advection":
        y = (edges - a - problem.speed * t) / length
        primitive = length * (0.2 * np.floor(y) + np.clip(y - np.floor(y) - 0.25, 0.0, 0.2))
        return np.diff(primitive) / dx
    if problem.profile == "riemann" and problem.equation == "burgers":
        ql, qr = problem.left, problem.right
        if t == 0.0 or ql >= qr:
            position = problem.x0 + (0.5 * (ql + qr) * t if ql >= qr else 0.0)
            fraction = np.clip((position - edges[:-1]) / dx, 0.0, 1.0)
            return qr + (ql - qr) * fraction
        xl, xr = problem.x0 + ql * t, problem.x0 + qr * t
        left_length = np.clip(xl - edges[:-1], 0.0, dx)
        right_length = np.clip(edges[1:] - xr, 0.0, dx)
        lo, hi = np.maximum(edges[:-1], xl), np.minimum(edges[1:], xr)
        fan = np.where(hi > lo, ((hi - problem.x0)**2 - (lo - problem.x0)**2) / (2.0 * t), 0.0)
        return (ql * left_length + fan + qr * right_length) / dx
    raise ValueError("Unsupported profile/equation combination.")


def simulate(problem, scheme, n=200, cfl=0.4, t_end=None, record=False, max_steps=200000):
    """Explicit conservative update. No clipping, retry, or solver fallback."""
    if not isinstance(n, (int, np.integer)) or n < 4 or not 0.0 < cfl <= 1.0:
        raise ValueError("Require integer n>=4 and 0<CFL<=1.")
    if problem.equation not in SCHEMES or scheme not in SCHEMES[problem.equation]:
        raise ValueError("Invalid equation or scheme.")
    if problem.boundary not in ("periodic", "outflow"):
        raise ValueError("Boundary must be periodic or outflow.")
    target = problem.final_time if t_end is None else float(t_end)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("Invalid final time.")
    edges = np.linspace(*problem.bounds, n + 1)
    dx = float(edges[1] - edges[0])
    q0 = exact_cell_average(problem, edges, 0.0)
    q, t, exchange, steps = q0.copy(), 0.0, 0.0, 0
    times, history, states = [t], [[t, dx * q.sum(), 0.0]], [q.copy()] if record else []
    status, reason = "ok", ""
    while t < target:
        if target - t <= 16.0 * np.finfo(float).eps * max(1.0, target):
            t = target
            times[-1], history[-1][0] = target, target
            break
        if steps >= max_steps:
            status, reason = "failed", "Maximum step count exceeded."
            break
        speed = abs(problem.speed) if problem.equation == "advection" else float(abs(q).max())
        dt = min(cfl * dx / speed, target - t) if speed > 0.0 else target - t
        if not np.isfinite(dt) or dt <= 0.0 or t + dt == t:
            status, reason = "failed", "Invalid or unresolvable time step."
            break
        ghost = np.pad(q, 1, mode="wrap" if problem.boundary == "periodic" else "edge")
        with np.errstate(over="ignore", invalid="ignore"):
            f = numerical_flux(ghost[:-1], ghost[1:], scheme, problem.equation, dt, dx, problem.speed)
            trial = q - (dt / dx) * np.diff(f)
        if not np.all(np.isfinite(trial)) or np.max(abs(trial)) > 1.0e6:
            status, reason = "failed", "Nonfinite state or numerical blow-up (>1e6); trial rejected."
            break
        q = trial
        exchange += dt * (f[-1] - f[0])
        t, steps = min(target, t + dt), steps + 1
        times.append(t)
        history.append([t, dx * q.sum(), dx * (q.sum() - q0.sum()) + exchange])
        if record:
            states.append(q.copy())
    return {"problem": problem, "scheme": scheme, "n": n, "cfl": cfl, "edges": edges, "x": 0.5 * (edges[:-1] + edges[1:]), "q0": q0, "q": q, "t": t, "target": target, "steps": steps, "status": status, "reason": reason, "exchange": exchange, "history": np.asarray(history), "times": np.asarray(times), "states": np.asarray(states) if record else None}


def metrics(run):
    p, q, edges = run["problem"], run["q"], run["edges"]
    exact = exact_cell_average(p, edges, run["t"]) if run["status"] == "ok" else None
    dx = np.diff(edges)
    reference_range = (p.mean - abs(p.amplitude), p.mean + abs(p.amplitude)) if p.profile == "sine" else ((0.0, 1.0) if p.profile == "box" else tuple(sorted((p.left, p.right))))
    tv = np.sum(abs(np.diff(q))) + (abs(q[-1] - q[0]) if p.boundary == "periodic" else 0.0)
    return {"case": p.key, "method": LABELS[run["scheme"]], "N": run["n"], "CFL": run["cfl"], "t": run["t"], "status": run["status"], "L1": float(np.sum(abs(q - exact) * dx) / np.sum(dx)) if exact is not None else np.nan, "balance": float(run["history"][-1, 2]), "TV": float(tv), "undershoot": max(0.0, reference_range[0] - float(q.min())), "overshoot": max(0.0, float(q.max()) - reference_range[1])}


def shock_position(run):
    """Mass-equivalent position; useful for conservation, not a profile sharpness test."""
    p = run["problem"]
    if p.profile != "riemann" or p.left <= p.right:
        raise ValueError("Requires a compressive Burgers Riemann problem.")
    a, b = p.bounds
    return a + (np.dot(run["q"], np.diff(run["edges"])) - p.right * (b - a)) / (p.left - p.right)


def sample_state(run, t):
    """Display interpolation only; never used to advance the solution."""
    if run["states"] is None:
        raise ValueError("Run with record=True before requesting animation frames.")
    if t > run["t"] + 1.0e-13:
        return np.full(run["n"], np.nan)
    j = int(np.searchsorted(run["times"], t, side="right"))
    if j == 0:
        return run["states"][0]
    if j >= len(run["times"]):
        return run["states"][-1]
    w = (t - run["times"][j - 1]) / (run["times"][j] - run["times"][j - 1])
    return (1.0 - w) * run["states"][j - 1] + w * run["states"][j]


def plot_profiles(runs, title=None):
    import matplotlib.pyplot as plt
    good = [r for r in runs if r["status"] == "ok"]
    if not good:
        raise ValueError("No completed runs to compare.")
    base, p = good[0], good[0]["problem"]
    if any(r["problem"] != p or abs(r["t"] - base["t"]) > 1.0e-12 for r in good):
        raise ValueError("Compare identical problems at a common physical time.")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    xp = np.linspace(*p.bounds, 2401)
    ref = exact_pointwise(p, xp, base["t"])
    if ref is not None:
        ax.plot(xp, ref, color="black", lw=1.5, label="Exact (pointwise)")
    for r in good:
        key = r["scheme"]
        ax.plot(r["x"], r["q"], color=COLORS[key], ls=STYLES[key], lw=1.4, label=LABELS[key])
    ax.set(xlabel="$x$", ylabel="$q$", xlim=p.bounds, title=title or f"{p.title}: t={base['t']:g}, N={base['n']}")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    for r in runs:
        if r["status"] != "ok":
            print(LABELS[r["scheme"]], "stopped at", r["t"], r["reason"])


def animate(runs, frames=41, fps=15):
    """Single-axis animation. Reference disappears after sine-wave breaking."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    if frames < 2 or fps <= 0:
        raise ValueError("Require frames>=2 and fps>0.")
    p, target = runs[0]["problem"], runs[0]["target"]
    if any(r["problem"] != p or r["target"] != target for r in runs):
        raise ValueError("Animation requires the same problem and target time.")
    times = np.linspace(0.0, target, frames)
    xp = np.linspace(*p.bounds, 1001)
    values = [np.stack([sample_state(r, t) for t in times]) for r in runs]
    refs = [exact_pointwise(p, xp, t) for t in times]
    all_values = np.concatenate([v[np.isfinite(v)] for v in values] + [r for r in refs if r is not None])
    lo, hi = float(all_values.min()), float(all_values.max())
    pad = 0.1 * max(hi - lo, 0.1)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    line_ref, = ax.plot(xp, refs[0], color="black", lw=1.4, label="Exact where available")
    lines = [ax.plot(r["x"], v[0], color=COLORS[r["scheme"]], ls=STYLES[r["scheme"]], label=LABELS[r["scheme"]])[0] for r, v in zip(runs, values)]
    ax.set(xlabel="$x$", ylabel="$q$", xlim=p.bounds, ylim=(lo - pad, hi + pad))
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    title = ax.set_title("")
    note = fig.text(0.09, 0.01, "", fontsize=9)
    def update(j):
        title.set_text(f"{p.title}: t={times[j]:.4f}")
        line_ref.set_ydata(refs[j] if refs[j] is not None else np.full_like(xp, np.nan))
        for line, v in zip(lines, values):
            line.set_ydata(v[j])
        stopped = [LABELS[r["scheme"]] for r in runs if times[j] > r["t"] + 1.0e-13]
        text = "Exact sine reference unavailable after shock formation." if refs[j] is None else "Display interpolation only; integration time steps are unchanged."
        note.set_text(text + (" STOPPED: " + ", ".join(stopped) if stopped else ""))
        return [line_ref, title, note] + lines
    update(0)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    movie = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps)
    output = HTML(movie.to_jshtml())
    plt.close(fig)
    return output


def export_results(runs, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [metrics(r) for r in runs]
    if not rows:
        return
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {f"run_{i:03d}_{k}": r[k] for i, r in enumerate(runs) for k in ("edges", "q0", "q", "history")}
    np.savez_compressed(directory / "states.npz", **arrays)
    settings = [{"case": r["problem"].__dict__, "scheme": r["scheme"], "N": r["n"], "CFL": r["cfl"], "target": r["target"], "status": r["status"], "reason": r["reason"]} for r in runs]
    (directory / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def run_tests():
    """Small reproducible numerical tests; raises AssertionError on failure."""
    checks = []
    z = np.array([-2.0, -0.2, 0.0, 1.0, 2.0])
    for eq, methods in SCHEMES.items():
        for scheme in methods:
            np.testing.assert_allclose(numerical_flux(z, z, scheme, eq, 0.1, 1.0), physical_flux(z, eq), atol=1e-14)
    checks.append("consistent flux F(q,q)=f(q)")
    l, r = np.array([1., -2., -1., 1., 2., -1.]), np.array([2., -1., 1., 0., 1., -2.])
    np.testing.assert_allclose(numerical_flux(l, r, "Godunov", "burgers", 0.1, 1.), [0.5, 0.5, 0., 0.5, 2., 2.])
    checks.append("Burgers Godunov shock/rarefaction branches")
    q = np.random.default_rng(49).normal(size=40)
    for a in (-1.0, 1.0):
        g = np.pad(q, 1, mode="wrap")
        for scheme in SCHEMES["advection"]:
            f = numerical_flux(g[:-1], g[1:], scheme, "advection", 0.025, 0.025, a)
            np.testing.assert_allclose(q - np.diff(f), np.roll(q, int(a)), atol=2e-14)
    checks.append("CFL=1 exact cell shift in both directions")
    for a in (-1.0, 1.0):
        p = replace(CASES["advection_box"], speed=a)
        qex = exact_cell_average(p, np.linspace(0, 1, 81), 0.73)
        np.testing.assert_allclose(qex.sum() / 80, 0.2, atol=2e-14)
    checks.append("periodic exact square-pulse cell integrals")
    for key, scheme in (("advection_sine", "Upwind"), ("burgers_shock", "Godunov")):
        run = simulate(CASES[key], scheme, n=40)
        assert abs(metrics(run)["balance"]) < 5e-13
    checks.append("periodic and open-boundary conservation budgets")
    p = CASES["burgers_shock"]
    r = simulate(p, "Godunov", n=80)
    np.testing.assert_allclose(shock_position(r), p.x0 + (p.left + p.right) * r["t"] / 2, atol=5e-13)
    checks.append("Rankine-Hugoniot shock position from conserved integral")
    p = replace(p, right=-1.0)
    r = simulate(p, "Godunov", n=80)
    np.testing.assert_allclose(r["q"], r["q0"], atol=1e-14)
    checks.append("stationary compressive shock is retained")
    p = CASES["burgers_rarefaction"]
    bad, good = simulate(p, "Roe_no_efix", n=80), simulate(p, "Godunov", n=80)
    np.testing.assert_allclose(bad["q"], bad["q0"], atol=1e-14)
    assert metrics(good)["L1"] < metrics(bad)["L1"] / 2
    checks.append("expansion-shock counterexample vs entropy solution")
    p = CASES["burgers_sine"]
    av = exact_cell_average(p, np.linspace(0, 1, 121), 0.4)
    np.testing.assert_allclose(av.mean(), p.mean, atol=2e-14)
    assert exact_cell_average(p, np.linspace(0, 1, 121), 1.01 * breaking_time(p)) is None
    checks.append("smooth Burgers integral and post-breaking reference guard")
    plain, recorded = simulate(p, "Godunov", n=40), simulate(p, "Godunov", n=40, record=True)
    np.testing.assert_array_equal(plain["q"], recorded["q"])
    assert plain["steps"] == recorded["steps"] and plain["exchange"] == recorded["exchange"]
    checks.append("recording does not change time integration")
    for scheme, lower, upper in (("Upwind", 0.8, 1.2), ("LF", 0.8, 1.2), ("LW", 1.8, 2.2)):
        errors = [metrics(simulate(CASES["advection_sine"], scheme, n=n))["L1"] for n in (100, 200)]
        order = np.log2(errors[0] / errors[1])
        assert lower < order < upper, (scheme, order)
    checks.append("advection measured convergence orders")
    return checks
