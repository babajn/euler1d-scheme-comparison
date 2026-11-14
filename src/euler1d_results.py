"""Track experiment membership independently of numerical-result reuse."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ExperimentCache(dict):
    """A result may be reused by several experiments, identified by cfg snapshots."""
    def __init__(self):
        super().__init__()
        self._experiments = {}

    def record(self, cfg, key):
        signature = json.dumps(cfg, sort_keys=True, default=json_value)
        self._experiments.setdefault(signature, {})[key] = None

    def runs_for(self, cfg):
        signature = json.dumps(cfg, sort_keys=True, default=json_value)
        return [self[key] for key in self._experiments.get(signature, {}) if key in self]

    def clear(self):
        super().clear()
        self._experiments.clear()


def run_settings(run):
    """Physical and numerical settings for one saved array group."""
    settings = {name: getattr(run, name) for name in
                ("solver", "n", "order", "gamma", "cfl", "limiter", "wave_speeds", "roe_entropy", "target_time")}
    settings["problem"] = asdict(run.problem)
    for name in ("reconstruction", "time_integrator", "fixed_steps"):
        if hasattr(run, name):
            settings[name] = getattr(run, name)
    return settings


def run_metadata(run_id, prefix, run, reference=None):
    if run.status != "ok":
        error_reference = "none: incomplete"
    elif run.problem.key != "shu_osher":
        error_reference = "exact cell averages"
    else:
        error_reference = "numerical" if reference is not None else "none"
    return dict(run_id=run_id, array_prefix=prefix, settings=run_settings(run),
                status=run.status, t_reached=run.t, steps=run.steps,
                error_reference=error_reference,
                reference_settings=run_settings(reference) if reference is not None else None)
