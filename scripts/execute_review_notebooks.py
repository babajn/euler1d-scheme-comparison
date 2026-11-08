"""Execute numbered notebooks in fresh kernels; keep all actual outputs."""
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]


def run(path):
    start=time.perf_counter();print('START',path.name,flush=True)
    nb=nbformat.read(path,as_version=4)
    first=next(c for c in nb.cells if c.cell_type=='code')
    source=first.source
    first.source='get_ipython().run_line_magic("matplotlib", "inline")\n'+source
    try:
        NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    except Exception:
        nbformat.write(nb,ROOT/'validation'/('FAILED_'+path.name));raise
    finally:first.source=source
    code=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    assert not errors and all(c.execution_count is not None for c in code)
    nbformat.validate(nb);nbformat.write(nb,path)
    report={'notebook':path.name,'code_cells':len(code),'errors':len(errors),'png_outputs':sum('image/png' in o.get('data',{}) for c in code for o in c.outputs),'animations':sum('text/html' in o.get('data',{}) and 'function Animation' in o.get('data',{}).get('text/html','') for c in code for o in c.outputs),'seconds':time.perf_counter()-start}
    print('DONE',report,flush=True);return report


if __name__=='__main__':
    (ROOT/'validation').mkdir(exist_ok=True)
    paths=sorted((ROOT/'notebooks').glob('0[1-5]_*.ipynb'))
    assert len(paths)==5
    with ThreadPoolExecutor(max_workers=3) as pool: reports=list(pool.map(run,paths))
    (ROOT/'validation/review_execution.json').write_text(json.dumps(reports,indent=2))
