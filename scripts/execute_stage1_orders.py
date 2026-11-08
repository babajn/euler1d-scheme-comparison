"""Execute standard settings in a fresh kernel and retain all outputs."""
from pathlib import Path
import json
import time
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'notebooks/euler1d_stage1_light.ipynb'
nb=nbformat.read(p,as_version=4)
start=time.perf_counter()
first=next(c for c in nb.cells if c.cell_type=='code')
original=first.source
first.source='get_ipython().run_line_magic("matplotlib", "inline")\n'+original

def on_cell_start(cell,cell_index,**kwargs):
    if cell.cell_type=='code': print(f'cell {cell_index}: {cell.id} | {time.perf_counter()-start:.1f}s',flush=True)

client=NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False,on_cell_start=on_cell_start)
client.execute()
first.source=original
nbformat.validate(nb)
nbformat.write(nb,p)
cs=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cs)
assert not [o for c in cs for o in c.outputs if o.output_type=='error']
outs=[o for c in cs for o in c.outputs]
vids=sum('video/mp4' in ''.join(o.get('data',{}).get('text/html','')) for o in outs)
pngs=sum('image/png' in o.get('data',{}) for o in outs)
assert vids==29,(vids,pngs)
assert pngs==15,(vids,pngs)
report={'notebook':p.name,'code_cells':len(cs),'embedded_mp4_videos':vids,'png_final_snapshots':pngs,'errors':0,'seconds':time.perf_counter()-start,'settings':'Unmodified defaults. A: 7 problems/14 videos, B: 3 problems/3 videos, C: 3 problems x 4 solvers/12 videos. A Shu-Osher reference N=1600,3200 executed.'}
(ROOT/'validation').mkdir(exist_ok=True)
(ROOT/'validation/stage1_order_execution.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
