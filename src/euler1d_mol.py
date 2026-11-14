"""Method-of-lines Euler solver: flux, reconstruction, and time method are independent.

Uses the fluxes, initial conditions, exact solutions, and boundary conditions in
 euler1d_core. LF/LW are not method-of-lines fluxes here and are deliberately excluded.
"""
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import euler1d_core as core

RECONSTRUCTIONS = ("constant", "muscl")
LIMITERS = ("minmod", "mc", "unlimited")
TIME_INTEGRATORS = ("forward_euler", "ssprk2", "ssprk3")
TIME_LABELS = {"forward_euler": "Forward Euler", "ssprk2": "SSP-RK2", "ssprk3": "SSP-RK3"}
TIME_ORDERS = {"forward_euler": 1, "ssprk2": 2, "ssprk3": 3}


def reconstruction_label(reconstruction, limiter):
    return "Piecewise constant" if reconstruction == "constant" else "MUSCL (" + limiter + ")"


def flux_label(solver, wave_speeds="einfeldt"):
    if solver == "LLF":
        return "LLF (Rusanov)"
    return "HLL (HLLE)" if solver == "HLL" and wave_speeds == "einfeldt" else solver


@dataclass
class MOLResult(core.RunResult):
    reconstruction: str = "constant"
    time_integrator: str = "forward_euler"
    n_rhs: int = 0
    max_cfl: float = 0.0
    dt_history: np.ndarray = field(default_factory=lambda: np.empty(0))
    fixed_steps: Optional[int] = None

    @property
    def time_order(self):
        return TIME_ORDERS[self.time_integrator]

    @property
    def label(self):
        return " | ".join((flux_label(self.solver, self.wave_speeds), reconstruction_label(self.reconstruction, self.limiter), TIME_LABELS[self.time_integrator]))


def face_flux(U, problem, solver, gamma, reconstruction, limiter, wave_speeds, roe_entropy):
    """Recompute ghost cells and interface states at every RK right-hand-side call."""
    UL, UR = core.interface_states(U, problem, gamma, 1 if reconstruction == "constant" else 2, limiter)
    return core.numerical_flux(UL, UR, solver, gamma, wave_speeds=wave_speeds, roe_entropy=roe_entropy)


def spatial_operator(U, problem, solver="HLLC", gamma=1.4, reconstruction="constant", limiter="minmod", wave_speeds="einfeldt", roe_entropy=0.1):
    """Return dU/dt and the net boundary influx per unit time; no dt enters L(U)."""
    if solver not in core.RIEMANN_SOLVERS or reconstruction not in RECONSTRUCTIONS or limiter not in LIMITERS:
        raise ValueError("Choose a supported Riemann flux, reconstruction, and limiter.")
    dx = (problem.bounds[1] - problem.bounds[0]) / len(U)
    F = face_flux(U, problem, solver, gamma, reconstruction, limiter, wave_speeds, roe_entropy)
    return -(F[1:] - F[:-1]) / dx, F[0] - F[-1]


def advance_mol(U, problem, solver, dt, dx, gamma, reconstruction, limiter, time_integrator, wave_speeds, roe_entropy, counter=None):
    """Return the new state and the boundary flux integrated with the same RK weights."""
    if time_integrator not in TIME_INTEGRATORS:
        raise ValueError("Unknown time_integrator.")
    def euler_step(V):
        if counter is not None:
            counter[0] += 1
        F = face_flux(V, problem, solver, gamma, reconstruction, limiter, wave_speeds, roe_entropy)
        return V - (dt / dx) * (F[1:] - F[:-1]), F[0] - F[-1]
    U1, B1 = euler_step(U)
    core.cons_to_prim(U1, gamma, stage="RK stage 1")
    if time_integrator == "forward_euler":
        return U1, dt * B1
    Y2, B2 = euler_step(U1)
    if time_integrator == "ssprk2":
        Unew = 0.5 * U + 0.5 * Y2
        core.cons_to_prim(Unew, gamma, stage="SSP-RK2 stage 2")
        return Unew, 0.5 * dt * (B1 + B2)
    U2 = 0.75 * U + 0.25 * Y2
    core.cons_to_prim(U2, gamma, stage="SSP-RK3 stage 2")
    Y3, B3 = euler_step(U2)
    Unew = U / 3.0 + (2.0 / 3.0) * Y3
    core.cons_to_prim(Unew, gamma, stage="SSP-RK3 stage 3")
    return Unew, dt * (B1 / 6.0 + B2 / 6.0 + (2.0 / 3.0) * B3)


def simulate(problem, solver="HLLC", n=200, gamma=1.4, cfl=0.4, reconstruction="constant", limiter="minmod", time_integrator="forward_euler", wave_speeds="einfeldt", roe_entropy=0.1, t_end=None, fixed_steps=None, max_steps=200000, history_stride=10, observer=None):
    """Adaptive CFL steps or exactly fixed_steps equal steps. No retry or floors.

    In fixed-step mode, cfl is an upper bound checked at accepted step beginnings;
    a violation stops the run rather than silently changing dt. The bound is a
    diagnostic restriction, not a positivity or stability proof for all combinations.
    """
    if solver not in core.RIEMANN_SOLVERS:
        raise ValueError("Use Roe, LLF, HLL, or HLLC. Classical LF/LW remain in euler1d_core.")
    if reconstruction not in RECONSTRUCTIONS or limiter not in LIMITERS or time_integrator not in TIME_INTEGRATORS:
        raise ValueError("Invalid reconstruction, limiter, or time_integrator.")
    if wave_speeds not in ("einfeldt", "davis", "symmetric"):
        raise ValueError("Invalid wave_speeds.")
    if not np.isfinite(gamma) or gamma <= 1 or not 0 < cfl <= 0.9 or not np.isfinite(roe_entropy) or roe_entropy < 0:
        raise ValueError("Require finite gamma>1, 0<CFL<=0.9, and entropy coefficient>=0.")
    if fixed_steps is not None and (isinstance(fixed_steps, bool) or not isinstance(fixed_steps, (int, np.integer)) or fixed_steps < 1):
        raise ValueError("fixed_steps must be None or a positive integer.")
    if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1 for k in (history_stride, max_steps)):
        raise ValueError("history_stride and max_steps must be positive integers.")
    target = problem.final_time if t_end is None else float(t_end)
    if not np.isfinite(target) or target < 0:
        raise ValueError("t_end must be finite and nonnegative.")
    edges, x, dx = core.grid_for(problem, n)
    U0 = core.initial_cell_average(problem, edges, gamma)
    U = U0.copy()
    initial = dx * U.sum(axis=0)
    exchanged = np.zeros(3)
    W = core.cons_to_prim(U, gamma)
    history = [[0.0, W[:, 0].min(), W[:, 2].min(), 0.0]]
    dts, counter, failure = [], [0], {}
    t, steps, status, cfl_seen = 0.0, 0, "ok", 0.0
    started = time.perf_counter()
    if observer is not None:
        observer(t, U)
    while (steps < fixed_steps if fixed_steps is not None else t < target) and target > 0:
        if steps >= max_steps:
            status, failure = "max_steps", {"fail_reason": "Maximum step count reached."}
            break
        speed = core.max_signal_speed(U, gamma)
        if fixed_steps is not None:
            dt = target / fixed_steps
            measured_cfl = dt * speed / dx
            if measured_cfl > cfl * (1.0 + 2e-12):
                status = "cfl_limit"
                failure = {"fail_time": t + dt, "fail_dt": dt, "fail_reason": f"Fixed dt gives CFL={measured_cfl:.6g} > allowed {cfl:g}. Increase fixed_steps."}
                break
        else:
            dt_cfl, remaining = cfl * dx / speed, target - t
            dt = dt_cfl if remaining > 2 * dt_cfl else remaining / max(1, int(np.ceil(remaining / dt_cfl)))
            measured_cfl = dt * speed / dx
        if not np.isfinite(dt) or dt <= 0 or t + dt == t:
            status, failure = "failed", {"fail_time": t, "fail_dt": dt, "fail_reason": "Time step cannot advance time."}
            break
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                trial, boundary = advance_mol(U, problem, solver, dt, dx, gamma, reconstruction, limiter, time_integrator, wave_speeds, roe_entropy, counter)
        except core.NumericalFailure as exc:
            status = "failed"
            failure = {"fail_time": t + dt, "fail_dt": dt, "fail_min_rho": exc.min_rho, "fail_min_p": exc.min_p, "fail_reason": str(exc)}
            break
        except FloatingPointError as exc:
            status, failure = "failed", {"fail_time": t + dt, "fail_dt": dt, "fail_reason": str(exc)}
            break
        U = trial
        exchanged += boundary
        steps += 1
        t = min(target, steps * dt) if fixed_steps is not None else min(target, t + dt)
        if fixed_steps is not None and steps == fixed_steps:
            t = target
        dts.append(dt)
        cfl_seen = max(cfl_seen, measured_cfl)
        if observer is not None:
            observer(t, U)
        if steps % history_stride == 0 or t == target:
            W = core.cons_to_prim(U, gamma)
            balance = dx * U.sum(axis=0) - initial - exchanged
            history.append([t, W[:, 0].min(), W[:, 2].min(), np.max(np.abs(balance))])
    elapsed = time.perf_counter() - started
    balance = dx * U.sum(axis=0) - initial - exchanged
    if history[-1][0] != t:
        W = core.cons_to_prim(U, gamma)
        history.append([t, W[:, 0].min(), W[:, 2].min(), np.max(np.abs(balance))])
    order = 1 if reconstruction == "constant" else 2
    return MOLResult(problem, solver, n, order, gamma, cfl, limiter, wave_speeds, roe_entropy, edges, x, U0, U, t, target, steps, elapsed, status, initial, exchanged, balance, np.asarray(history), **failure, reconstruction=reconstruction, time_integrator=time_integrator, n_rhs=counter[0], max_cfl=cfl_seen, dt_history=np.asarray(dts), fixed_steps=fixed_steps)


def metrics(run, reference=None):
    row = core.metrics(run, reference)
    row.update(method=run.label, reconstruction=run.reconstruction, time_integrator=run.time_integrator, n_rhs=run.n_rhs, max_cfl=run.max_cfl, fixed_steps=run.fixed_steps)
    row["dt_min"] = float(run.dt_history.min()) if run.steps else np.nan
    row["dt_max"] = float(run.dt_history.max()) if run.steps else np.nan
    return row


def run_tests():
    """Small deterministic checks; not a proof for arbitrary initial conditions."""
    checks = []
    for solver in core.RIEMANN_SOLVERS:
        for order, rec, ti in ((1, "constant", "forward_euler"), (2, "muscl", "ssprk2")):
            a = core.simulate(core.PROBLEMS["sod"], solver, n=24, order=order)
            b = simulate(core.PROBLEMS["sod"], solver, n=24, reconstruction=rec, time_integrator=ti)
            np.testing.assert_array_equal(a.U, b.U)
            np.testing.assert_array_equal(a.boundary_exchange, b.boundary_exchange)
            assert a.steps == b.steps and a.t == b.t and a.status == b.status
    checks.append("exact agreement with piecewise-constant/FE and MUSCL/SSP-RK2 baseline")
    W = np.tile([1.0, 0.3, 1.0], (16, 1))
    U = core.prim_to_cons(W)
    p = core.PROBLEMS["entropy"]
    for solver in core.RIEMANN_SOLVERS:
        for rec in RECONSTRUCTIONS:
            for ti in TIME_INTEGRATORS:
                out, exchange = advance_mol(U, p, solver, 0.005, 1 / 16, 1.4, rec, "mc", ti, "einfeldt", 0.1)
                np.testing.assert_allclose(out, U, rtol=0.0, atol=1e-14)
                np.testing.assert_allclose(exchange, 0.0, atol=1e-14)
    checks.append("constant state: all flux/reconstruction/time combinations")
    for solver in core.RIEMANN_SOLVERS:
        r = simulate(core.PROBLEMS["sod"], solver, n=24, reconstruction="muscl", time_integrator="ssprk3")
        assert r.status == "ok" and np.max(np.abs(r.balance)) < 3e-13
        assert r.n_rhs == 3 * r.steps
    checks.append("SSP-RK3 open-boundary budget and RHS count")
    for ti in TIME_INTEGRATORS:
        r = simulate(p, "HLLC", n=24, time_integrator=ti, t_end=0.05, fixed_steps=20)
        assert r.status == "ok" and r.steps == 20 and r.t == 0.05
        np.testing.assert_array_equal(r.dt_history, np.full(20, 0.05 / 20))
        assert r.n_rhs == TIME_ORDERS[ti] * 20
        assert np.max(np.abs(r.balance)) < 3e-13
    checks.append("fixed equal steps, counts, and periodic conservation")
    records = []
    a = simulate(p, "LLF", n=24, reconstruction="muscl", time_integrator="ssprk3", t_end=0.05)
    b = simulate(p, "LLF", n=24, reconstruction="muscl", time_integrator="ssprk3", t_end=0.05, observer=lambda t, U: records.append((t, U.copy())))
    np.testing.assert_array_equal(a.U, b.U)
    assert a.steps == b.steps and np.array_equal(a.boundary_exchange, b.boundary_exchange)
    assert len(records) == b.steps + 1
    checks.append("observer does not change integration")
    r = simulate(core.PROBLEMS["double_rarefaction"], "Roe", n=40, time_integrator="forward_euler")
    assert r.status == "failed" and r.t < r.target_time
    core.cons_to_prim(r.U)
    checks.append("failed trial leaves an admissible last accepted state")
    r = simulate(p, "HLLC", n=40, fixed_steps=1)
    assert r.status == "cfl_limit" and r.steps == 0
    checks.append("unsafe fixed dt is rejected, not changed")
    # HLLC transports the constant-p, constant-u entropy mode as scalar upwinding.
    theta, dt, dx = 2 * np.pi / 32, 0.004, 1 / 32
    edges, x, _ = core.grid_for(p, 32)
    start = core.initial_cell_average(p, edges, 1.4)
    z = dt * (np.exp(-1j * theta) - 1) / dx
    for ti, degree in TIME_ORDERS.items():
        out, _ = advance_mol(start, p, "HLLC", dt, dx, 1.4, "constant", "minmod", ti, "einfeldt", 0.1)
        R = 1 + z + (z**2 / 2 if degree >= 2 else 0) + (z**3 / 6 if degree == 3 else 0)
        expected = 1 + 0.2 * np.sinc(1 / 32) * np.imag(np.exp(2j * np.pi * x) * R)
        np.testing.assert_allclose(out[:, 0], expected, rtol=0.0, atol=8e-15)
    checks.append("FE/SSP-RK2/SSP-RK3 stability polynomials on an Euler entropy mode")
    return checks
