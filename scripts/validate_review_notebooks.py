"""Validate instructional structure, execution output and review experiments."""
from pathlib import Path
import csv
import json
import re
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import scalar1d, euler1d_core, euler1d_mol, tvd_study, review_experiments
    test_sets = {
        'scalar': scalar1d.run_tests(),
        'euler': euler1d_core.run_unit_tests(),
        'mol': euler1d_mol.run_tests(),
        'tvd': tvd_study.run_tests(),
        'review': review_experiments.run_tests(),
    }
    result = {'test_groups': {key: len(value) for key, value in test_sets.items()}, 'notebooks': [], 'reference_checks': []}
    paths = sorted((ROOT / 'notebooks').glob('0[1-5]_*.ipynb'))
    assert len(paths) == 5
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        major, minors, active, titles = [], {}, None, []
        for cell in nb.cells:
            if cell.cell_type != 'markdown':
                continue
            fenced = False
            for line in cell.source.splitlines():
                if line.startswith('```'):
                    fenced = not fenced
                if fenced:
                    continue
                if line.startswith('# '):
                    titles.append(line)
                if line.startswith('## '):
                    m = re.match(r'## (\d+)\. ', line)
                    active = int(m.group(1)) if m else None
                    if m:
                        major.append(active)
                if line.startswith('### '):
                    m = re.match(r'### (\d+)\.(\d+) ', line)
                    if active is not None:
                        assert m and int(m.group(1)) == active, (path.name, line)
                        minors.setdefault(active, []).append(int(m.group(2)))
            for link in re.findall(r'\]\(([^)\s]+\.ipynb)(?:#[^)]*)?\)', cell.source):
                assert (path.parent / link).exists(), (path.name, link)
        assert len(titles) == 1
        assert major == list(range(len(major))), major
        for seq in minors.values():
            assert seq == list(range(1, len(seq)+1)), seq
        code = [c for c in nb.cells if c.cell_type == 'code']
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type == 'error' for c in code for o in c.outputs)
        png = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
        animations = sum('function Animation' in o.get('data', {}).get('text/html', '') for c in code for o in c.outputs)
        assert png and animations == 1
        assert any('"run_movies": True' in c.source for c in code)
        assert any(c.id.startswith('rev-') for c in nb.cells)
        result['notebooks'].append(dict(notebook=path.name, code_cells=len(code), png_outputs=png, animations=animations, errors=0, unique_part_title=True, consecutive_headings=True))
    for limiter in ('unlimited', 'minmod', 'mc'):
        path = ROOT / 'five_part_outputs' / 'temporal' / f'reference_muscl_{limiter}.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            ratio = float(row['gap_over_error'])
            assert ratio < 0.05, (limiter, ratio)
            result['reference_checks'].append(dict(limiter=limiter, ratio=ratio))
    failure = euler1d_mol.simulate(euler1d_core.PROBLEMS['sod'], 'HLLC', n=200, reconstruction='muscl', limiter='unlimited', time_integrator='ssprk2')
    if failure.status == 'failed' and 'MUSCL reconstruction' in failure.fail_reason:
        import numpy as np
        assert np.isfinite(failure.fail_min_rho) and np.isfinite(failure.fail_min_p)
        result['reconstruction_failure'] = dict(reason=failure.fail_reason, rho_min=failure.fail_min_rho, p_min=failure.fail_min_p)
    result['scope'] = 'All five notebooks: default settings, saved outputs, new standard exercises. Optional high-resolution Shu-Osher not executed.'
    (ROOT / 'validation').mkdir(exist_ok=True)
    (ROOT / 'validation/review_v3_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
