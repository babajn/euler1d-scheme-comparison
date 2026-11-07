# Standalone module extracted from notebooks/euler1d_stage1_light.ipynb
# Numerical algorithm is unchanged from the notebook.
from euler1d_core import ALL_SOLVERS, PROBLEMS, simulate, exact_cell_average, cons_to_prim

"""Record common-time frames without changing a solver's integration steps."""
from pathlib import Path
import hashlib
import json
import numpy as np



def _signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def frame_cache_parameters(key, cfg):
    """Physical and sampling settings. Rendering-only changes do not rerun the simulation."""
    p = PROBLEMS[key]
    names = ("gamma", "cfl", "wave_speeds", "roe_entropy", "n_frames", "n_reference", "reference_limiter", "verify_observer")
    result = {name: cfg[name] for name in names}
    result.update({"format_version": 1, "problem": vars(p), "n": cfg["n_shu_osher"] if key == "shu_osher" else cfg["n_basic"]})
    # Change cache_revision after editing the physical solver in the notebook.
    result["cache_revision"] = cfg.get("cache_revision", "original-solver-with-observer-v1")
    return json.loads(json.dumps(result))


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


def run_summary(run):
    return {"solver": run.solver, "label": run.label, "order": run.order, "n": run.n, "status": run.status,
            "t_reached": run.t, "t_target": run.target_time, "steps": run.steps, "seconds": run.elapsed,
            "fail_time": run.fail_time, "fail_dt": run.fail_dt, "fail_min_rho": run.fail_min_rho,
            "fail_min_p": run.fail_min_p, "fail_reason": run.fail_reason,
            "balance_max": float(np.max(np.abs(run.balance)))}


def _run_and_record(problem, solver, n, times, cfg, order=1, limiter="minmod"):
    recorder = FrameRecorder(times)
    kwargs = dict(n=n, gamma=cfg["gamma"], cfl=cfg["cfl"], order=order, limiter=limiter, wave_speeds=cfg["wave_speeds"], roe_entropy=cfg["roe_entropy"])
    run = simulate(problem, solver, observer=recorder, **kwargs)
    summary = run_summary(run)
    good = recorder.valid
    if not np.array_equal(recorder.states[0], run.U0):
        raise AssertionError("Initial movie frame differs from the solver initial state.")
    if np.any(good):
        cons_to_prim(recorder.states[good], cfg["gamma"])
        assert np.all(recorder.times[good] <= run.t)
    if run.status == "ok":
        assert good.all() and np.array_equal(recorder.states[-1], run.U)
    else:
        assert np.isnan(recorder.states[~good]).all()
    if cfg["verify_observer"] and order == 1:
        baseline = simulate(problem, solver, **kwargs)
        same = (np.array_equal(run.U, baseline.U) and run.steps == baseline.steps and run.status == baseline.status and run.t == baseline.t and np.array_equal(run.boundary_exchange, baseline.boundary_exchange))
        if not same:
            raise AssertionError("The observer changed the numerical evolution.")
        summary["observer_identical"] = True
    return run, recorder, summary


def save_animation_data(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"times": data["times"], "x": data["x"], "edges": data["edges"], "reference": data["reference"]}
    for solver in ALL_SOLVERS:
        arrays["U_" + solver] = data["states"][solver]
        arrays["valid_" + solver] = data["valid"][solver]
        arrays["left_" + solver], arrays["right_" + solver], arrays["weight_" + solver] = data["brackets"][solver]
    metadata = {k: data[k] for k in ("key", "title", "gamma", "cfl", "n", "parameters", "data_id", "runs", "reference_label", "reference_check")}
    arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **arrays)


def load_animation_data(path):
    with np.load(path, allow_pickle=False) as z:
        data = json.loads(str(z["metadata"]))
        data.update({name: z[name].copy() for name in ("times", "x", "edges", "reference")})
        data["states"] = {s: z["U_" + s].copy() for s in ALL_SOLVERS}
        data["valid"] = {s: z["valid_" + s].copy() for s in ALL_SOLVERS}
        data["brackets"] = {s: tuple(z[p + s].copy() for p in ("left_", "right_", "weight_")) for s in ALL_SOLVERS}
    return data


def get_animation_data(key, cfg=ANIM_CFG):
    """Six Stage-1 schemes, common output times, and an exact or numerical reference."""
    parameters = frame_cache_parameters(key, cfg)
    cache = Path(cfg["output_dir"]) / "data" / (key + ".npz")
    if cfg["reuse_data"] and cache.exists():
        cached = load_animation_data(cache)
        if cached["parameters"] == parameters:
            print(f"{key}: read cached common-time frames")
            return cached
    p, n = PROBLEMS[key], parameters["n"]
    times = np.linspace(0.0, p.final_time, int(cfg["n_frames"]))
    data = {"key": key, "title": p.title, "gamma": cfg["gamma"], "cfl": cfg["cfl"], "n": n,
            "times": times, "parameters": parameters, "data_id": _signature(parameters),
            "runs": {}, "states": {}, "valid": {}, "brackets": {}, "reference_check": None}
    for solver in ALL_SOLVERS:
        run, recorder, summary = _run_and_record(p, solver, n, times, cfg)
        data["runs"][solver], data["states"][solver], data["valid"][solver] = summary, recorder.states, recorder.valid
        data["brackets"][solver] = (recorder.left_time, recorder.right_time, recorder.weight)
        data["x"], data["edges"] = run.x, run.edges
    if key != "shu_osher":
        data["reference"] = np.stack([exact_cell_average(p, data["edges"], t, cfg["gamma"]) for t in times])
        data["reference_label"] = "Exact (cell-averaged U)"
    else:
        nr1, nr2 = map(int, cfg["n_reference"])
        if not (nr2 > nr1 >= n and nr2 % nr1 == 0 and nr1 % n == 0):
            raise ValueError("Require n_reference[1] > n_reference[0] >= n_shu_osher and integer refinement ratios.")
        averaged = []
        for nr in (nr1, nr2):
            print(f"{key}: numerical reference, HLLC + MUSCL/SSP-RK2, N={nr}")
            ref_path = cache.parent / f"shu_osher_reference_{nr}_{data['data_id'][:12]}.npz"
            if cfg["reuse_data"] and ref_path.exists():
                with np.load(ref_path, allow_pickle=False) as saved:
                    projected = saved["projected"].copy()
            else:
                run, recorder, summary = _run_and_record(p, "HLLC", nr, times, cfg, order=2, limiter=cfg["reference_limiter"])
                if run.status != "ok":
                    raise RuntimeError("A numerical reference failed: " + run.fail_reason)
                projected = recorder.states.reshape(len(times), n, nr // n, 3).mean(axis=2)
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(ref_path, projected=projected)
            averaged.append(projected)
        data["reference"] = averaged[-1]
        gap = np.mean(np.abs(averaged[1][..., 0] - averaged[0][..., 0]), axis=1)
        data["reference_label"] = f"Numerical ref. N={nr2}\n(projected to N={n})"
        data["reference_check"] = {"n_reference": [nr1, nr2], "final_density_L1_gap": float(gap[-1]), "max_movie_density_L1_gap": float(gap.max()), "note": "Two-resolution numerical comparison, not an exact solution or a convergence proof."}
    cons_to_prim(data["reference"], cfg["gamma"])
    save_animation_data(data, cache)
    print(f"{key}: " + ", ".join(s + "=" + data["runs"][s]["status"] for s in ALL_SOLVERS))
    return data