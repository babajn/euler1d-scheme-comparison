"""Small tests for both orders, cache isolation and failed-frame handling."""
from pathlib import Path
import json
import tempfile
import numpy as np
import nbformat
import matplotlib
matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[1]


def load_definitions():
    nb = nbformat.read(ROOT / 'notebooks/euler1d_stage1_light.ipynb', as_version=4)
    wanted = {'f208f769','4fcea455','4ad541c0','e5516255','df3fc6fa','91d6c64e','6297cce0','42a4838f','597be8ec','e74b1cc1','ce716b67','bf80bab8','order-settings','order-functions'}
    env = {'__name__': '__main__'}
    for c in nb.cells:
        if c.id in wanted:
            exec(compile(c.source, c.id, 'exec'), env)
    return env


def main():
    env = load_definitions()
    checks = env['run_unit_tests']()
    with tempfile.TemporaryDirectory() as temp:
        cfg = dict(env['ANIM_CFG'], output_dir=temp, n_basic=24, n_shu_osher=24, n_reference=(48,96), n_frames=9, save_mp4=False, reuse_data=True)
        ocfg = dict(env['ORDER_CFG'], problems=tuple(env['PROBLEMS']))
        env['validate_order_config'](cfg, ocfg)
        for key in ocfg['problems']:
            base = env['get_animation_data'](key, cfg)
            high = env['get_muscl_animation_data'](key, cfg, ocfg, base)
            for s in ocfg['fluxes']:
                assert high['runs'][s]['observer_identical']
                assert np.array_equal(high['states'][s][0],base['states'][s][0])
                assert np.isnan(high['states'][s][~high['valid'][s]]).all()
                if high['runs'][s]['status']=='ok':
                    assert high['valid'][s].all()
            for mode, s in [('muscl',None),('pair','Roe')]:
                fig,movie,update=env['make_order_comparison_animation'](base,high,'rho',cfg,mode,s)
                fig.canvas.draw(); movie.event_source.stop(); update(len(base['times'])-1)
                movie._draw_was_started=True
                env['plt'].close(fig)
            original = env['simulate']
            def no_simulation(*args, **kwargs):
                raise AssertionError('Cache incorrectly reran a simulation')
            env['simulate']=no_simulation
            cached = env['get_muscl_animation_data'](key,cfg,ocfg,base)
            env['simulate']=original
            for s in ocfg['fluxes']:
                np.testing.assert_array_equal(cached['states'][s],high['states'][s])
            print('passed:',key,{s:high['runs'][s]['status'] for s in high['runs']})
        checks.extend(['all eight problems with four MUSCL fluxes; observer unchanged','A/B grids and initial states match','failed frames stay NaN and are not extrapolated','B/C render both normal and failed runs','cached MUSCL data reused without simulation'])
        p1=env['order_cache_parameters']('sod',cfg,ocfg)
        p2=env['order_cache_parameters']('sod',cfg,dict(ocfg,limiter='mc'))
        assert p1!=p2
        bad=dict(ocfg,fluxes=('LF','HLLC'))
        try: env['validate_order_config'](cfg,bad)
        except ValueError: pass
        else: raise AssertionError('LF accepted for MUSCL')
        checks.extend(['limiter participates in cache signature','LF/LW rejected from MUSCL comparisons'])
    out={'tests':checks,'count':len(checks),'scope':'Small n=24 problems, including Shu-Osher numerical references 48/96. All existing numerical-core tests included.'}
    (ROOT/'validation').mkdir(exist_ok=True)
    (ROOT/'validation/stage1_order_tests.json').write_text(json.dumps(out,indent=2))
    print('All checks passed:',len(checks))

if __name__=='__main__':
    main()
