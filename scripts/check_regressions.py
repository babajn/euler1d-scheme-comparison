"""Regression checks for experiment export, invalid inputs and periodic references.

Uses the standard library test runner and temporary output directories only.
Notebook definitions are executed in memory; saved outputs are never rewritten.
"""
from pathlib import Path
from dataclasses import replace
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import euler1d_core as core
import euler1d_mol as mol

CORE_IDS = {"f208f769", "4fcea455", "4ad541c0", "e5516255", "df3fc6fa",
            "91d6c64e", "6297cce0", "42a4838f", "597be8ec"}


def notebook(name):
    return json.loads((ROOT / "notebooks" / (name + ".ipynb")).read_text())


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import euler1d_experiments
        import euler1d_study
        cls.ex, cls.study = euler1d_experiments, euler1d_study
        cls.embedded = []
        for name in ("euler1d_stage1_light", "euler1d_complete_light"):
            env = {"__name__": name}
            for cell in notebook(name)["cells"]:
                if cell.get("id") in CORE_IDS:
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), env)
            cls.embedded.append(env)

    def setUp(self):
        self.ex.CACHE.clear()
        self.study.RUN_CACHE.clear()
        self.cfg = dict(n_basic=24, gamma=1.4, cfl=0.4, limiter="minmod",
                        wave_speeds="einfeldt", roe_entropy=0.1)

    def test_nonfinite_inputs_rejected(self):
        solvers = [(core.simulate, core.PROBLEMS), (mol.simulate, core.PROBLEMS)]
        solvers += [(env["simulate"], env["PROBLEMS"]) for env in self.embedded]
        for simulate, problems in solvers:
            for name in ("t_end", "gamma", "cfl", "roe_entropy"):
                for value in (np.nan, np.inf, -np.inf):
                    with self.subTest(module=simulate.__module__, parameter=name, value=value):
                        with self.assertRaises(ValueError):
                            simulate(problems["entropy"], "Roe", n=16, **{name: value})
            with self.assertRaises(ValueError):
                simulate(replace(problems["entropy"], final_time=np.nan), "HLLC", n=16)

    def test_entropy_reference_matches_cell_integral(self):
        nodes, weights = np.polynomial.legendre.leggauss(12)
        for bounds in ((0., 2.), (10., 11.), (-3., 1.)):
            for speed in (-1.3, 1.0):
                p = replace(core.PROBLEMS["entropy"], bounds=bounds, velocity=speed)
                edges, x, dx = core.grid_for(p, 80)
                for t in (0., 2.7):
                    points = x[:, None] + dx * nodes / 2
                    integral = np.einsum("q,iqk->ik", weights,
                                         core.prim_to_cons(core.exact_pointwise(p, points, t))) / 2
                    np.testing.assert_allclose(core.exact_cell_average(p, edges, t), integral, atol=2e-13, rtol=2e-13)

    def test_contact_translation_scaling_and_wrap(self):
        unit_edges = np.linspace(0., 1., 101)
        for bounds in ((0., 2.), (10., 11.), (-3., 1.)):
            origin, end = bounds
            length = end - origin
            for speed in (-1., 1.):
                p = replace(core.PROBLEMS["contact"], bounds=bounds, velocity=speed)
                unit = replace(core.PROBLEMS["contact"], velocity=speed)
                edges, x, dx = core.grid_for(p, 100)
                for t_unit in (0., 2.37):
                    actual = core.exact_cell_average(p, edges, t_unit * length)
                    expected = core.exact_cell_average(unit, unit_edges, t_unit)
                    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0.)
                    self.assertAlmostEqual(float(np.dot(actual[:, 0] - 1., np.diff(edges))), 0.2 * length, places=12)
                    np.testing.assert_allclose(core.exact_pointwise(p, x, t_unit * length),
                                               core.exact_pointwise(unit, (x - origin) / length, t_unit), atol=1e-13)

    def test_entropy_amplitude_uses_domain_wavelength(self):
        p = replace(core.PROBLEMS["entropy"], bounds=(10., 12.))
        run = core.simulate(p, "HLLC", n=80, t_end=0.)
        theta = 2 * np.pi * (run.x - 10.) / 2.
        rho = 1 + 0.15 * np.sinc(1 / 80) * np.sin(theta + 0.2)
        run.U = core.prim_to_cons(np.column_stack((rho, np.ones(80), np.ones(80))))
        row = core.metrics(run)
        self.assertAlmostEqual(row["amp_ratio"], 0.75, places=12)
        self.assertAlmostEqual(row["phase_rad"], 0.2, places=12)

    def test_export_isolates_settings_and_records_gamma(self):
        for module in (self.ex, self.study):
            cfg = dict(self.cfg)
            old_cfg = dict(cfg)
            old = module.get_run("entropy", "HLLC", cfg, t_end=.01)
            cfg["gamma"] = 5 / 3  # Mutating the same dictionary must also be safe.
            new = module.get_run("entropy", "HLLC", cfg, t_end=.01)
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                rows = module.export_results(folder / "new", cfg)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["gamma"], 5 / 3)
                metadata = json.loads((folder / "new/runs.json").read_text())
                self.assertEqual(metadata[0]["settings"]["gamma"], 5 / 3)
                self.assertEqual(metadata[0]["settings"]["problem"]["bounds"], [0., 1.])
                with np.load(folder / "new/states.npz", allow_pickle=False) as states:
                    np.testing.assert_array_equal(states[metadata[0]["array_prefix"] + "_U"], new.U)
                old_rows = module.export_results(folder / "old", old_cfg)
                self.assertEqual(len(old_rows), 1)
                self.assertEqual(old_rows[0]["gamma"], old.gamma)

    def test_cache_reuse_and_intentional_parameter_sweeps(self):
        parent = dict(self.cfg)
        first = self.ex.get_run("entropy", "HLLC", parent, t_end=.01)
        self.ex.get_run("entropy", "HLLC", parent, n=48, cfl=.2, t_end=.01)
        derived = dict(parent, temporal_limiter="mc")
        self.ex.get_run("entropy", "HLLC", derived, reconstruction="muscl", limiter="mc",
                        t_end=.01, experiment_cfg=parent)
        other = dict(parent, movie_frames=5)
        reused = self.ex.get_run("entropy", "HLLC", other, t_end=.01)
        self.assertIs(first, reused)
        with tempfile.TemporaryDirectory() as directory:
            rows = self.ex.export_results(Path(directory) / "parent", parent)
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["n"] for r in rows}, {24, 48})
            self.assertEqual(len(self.ex.export_results(Path(directory) / "other", other)), 1)

    def test_grid_change_does_not_project_old_runs(self):
        old_cfg, new_cfg = dict(self.cfg, n_basic=16), dict(self.cfg, n_basic=24)
        reference = core.simulate(core.PROBLEMS["shu_osher"], "HLLC", n=72, order=2,
                                  limiter="mc", t_end=.03)
        for module in (self.ex, self.study):
            module.get_run("shu_osher", "HLLC", old_cfg, t_end=.03)
            module.get_run("shu_osher", "HLLC", new_cfg, t_end=.03)
            with tempfile.TemporaryDirectory() as directory:
                rows = module.export_results(directory, new_cfg, reference=reference)
                self.assertEqual([r["n"] for r in rows], [24])
                metadata = json.loads((Path(directory) / "runs.json").read_text())
                self.assertEqual(metadata[0]["reference_settings"]["n"], 72)
                with np.load(Path(directory) / "states.npz", allow_pickle=False) as states:
                    np.testing.assert_array_equal(states["reference_U"], reference.U)

    def test_empty_experiment_does_not_write_results(self):
        for module in (self.ex, self.study):
            module.get_run("entropy", "HLLC", self.cfg, t_end=.01)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "absent"
                with self.assertRaises(ValueError):
                    module.export_results(output, dict(self.cfg, gamma=5 / 3))
                self.assertFalse(output.exists())

    def test_notebook_settings_reset_tables_and_reference(self):
        for name, cid in (("04_spatial_reconstruction", "spatial-005"), ("05_time_integration", "temporal-005")):
            self.ex.get_run("entropy", "HLLC", self.cfg, t_end=.01)
            env = dict(ex=self.ex, ROOT=ROOT, TABLES={"old": [1]}, reference=object())
            cell = next(c for c in notebook(name)["cells"] if c.get("id") == cid)
            exec("".join(cell["source"]), env)
            self.assertFalse(self.ex.CACHE)
            self.assertEqual(env["TABLES"], {})
            self.assertIsNone(env["reference"])

    def test_standalone_reconstruction_reports_actual_minima(self):
        for env in self.embedded:
            run = env["simulate"](env["PROBLEMS"]["sod"], "HLLC", n=200, order=2, limiter="unlimited")
            self.assertEqual(run.status, "failed")
            self.assertEqual(run.fail_min_rho, -0.09375)
            self.assertEqual(run.fail_min_p, -0.125)
            self.assertEqual(run.t, 0.)
            np.testing.assert_array_equal(run.U, run.U0)

    def test_embedded_core_definitions_match_source(self):
        def definitions(source):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        expected = definitions((ROOT / "src/euler1d_core.py").read_text())
        for name in ("euler1d_stage1_light", "euler1d_complete_light"):
            actual = {}
            for cell in notebook(name)["cells"]:
                if cell.get("id") in CORE_IDS:
                    actual.update(definitions("".join(cell["source"])))
            for key, value in expected.items():
                self.assertEqual(actual.get(key), value, (name, key))
        # Export helpers are also embedded to keep complete_light self-contained.
        expected_helpers = definitions((ROOT / "src/euler1d_results.py").read_text())
        cell = next(c for c in notebook("euler1d_complete_light")["cells"] if c.get("id") == "8af3f8f2")
        actual_helpers = definitions("".join(cell["source"]))
        for key, value in expected_helpers.items():
            self.assertEqual(actual_helpers.get(key), value, key)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="euler1d-regressions-") as directory:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(directory) / "cache"))
        os.environ["MPLBACKEND"] = "Agg"
        with contextlib.redirect_stdout(io.StringIO()):
            unittest.main(verbosity=2)
