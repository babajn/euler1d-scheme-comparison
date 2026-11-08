"""Apply reviewed teaching sections and checks; never fabricate execution output."""
from pathlib import Path
import json
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("01_advection_burgers", "02_euler_classical", "03_euler_riemann_solvers", "04_spatial_reconstruction", "05_time_integration")
BLOCK = re.compile(r"^<!-- (after|before):([^:]+):([^:]+):(markdown|code) -->\s*$", re.M)


def cell(nb, cid):
    found = [c for c in nb.cells if c.id == cid]
    if len(found) != 1:
        raise ValueError(f"Expected one cell: {cid}, got {len(found)}")
    return found[0]


def insert_blocks(nb, path):
    source = path.read_text(encoding="utf-8")
    matches = list(BLOCK.finditer(source))
    for i, m in enumerate(matches):
        mode, target, cid, kind = m.groups()
        text = source[m.end():matches[i+1].start() if i+1<len(matches) else len(source)].strip()
        existing = [c for c in nb.cells if c.id == cid]
        if existing:
            if existing[0].source != text:
                existing[0].source = text
                if kind == "code":
                    existing[0].outputs = []; existing[0].execution_count = None
            continue
        new = nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
        new.id = cid
        j = next(j for j,c in enumerate(nb.cells) if c.id == target)
        nb.cells.insert(j + (mode == "after"), new)


def move_after(nb, cid, target):
    c = cell(nb,cid)
    nb.cells.remove(c)
    j = next(j for j,c in enumerate(nb.cells) if c.id == target)
    nb.cells.insert(j+1,c)


def cleanup(nb, part):
    for c in nb.cells:
        c.source = c.source.replace('four_part_outputs', 'five_part_outputs')
        c.source = c.source.replace('[III 数値流束]', '[III 近似リーマンソルバー]')
        if c.cell_type != "markdown":
            continue
        s = c.source
        s = re.sub(r'<a id="part-[^"\n]+"></a>\s*', '', s)
        s = s.replace('## スカラー保存則から流体スキームを学ぶ', '**スカラー保存則から流体スキームを学ぶ**')
        s = s.replace('## この部で最初に理解すること', '**この部で最初に理解すること**')
        s = s.replace(r'\boldsymbol F^*', r'\widehat{\boldsymbol F}').replace(r'F^*', r'\widehat F')
        s = s.replace(r'\varepsilon_1', r'E_{\mathrm{total}}').replace(r'\epsilon_1', r'E_{\mathrm{total}}')
        s = s.replace('§2.2a', '「全変動とTVD条件」').replace('§2.4a', '「全変動の時間変化を測る演習」')
        s = s.replace('第IV部§2.2', '第IV部「このスカラー演習に対するTVDの十分条件」')
        s = s.replace('§2.2', '「TVDの確認演習」') if c.id.startswith('rev-') else s
        if part == 0:
            s = re.sub(r'(?<![A-Za-z_])C(?![A-Za-z_])', lambda m: r'C_{\mathrm{adv}}', s)
            s = s.replace(r'\frac C_{\mathrm{adv}}2', r'\frac{C_{\mathrm{adv}}}{2}')
        if part in (1,2):
            s = s.replace('RoeとHLLCがほぼ重なること自体も、このテストでは重要な結果です。','')
            s = s.replace('を測定します。SSP-RK2の境界流束も段階の重み$1/2,1/2$で積算します。','を測定します。この部では、実際の1ステップ更新に使う境界流束から保存収支を求めます。')
        if part == 1:
            s = s.replace('表示名は `LLF (Rusanov)`、内部識別子は `LLF` です。','')
        if part == 2:
            s = s.replace('LWは滑らかな解で2次精度です。','')
            s = s.replace(r'上のハットと、本文の$\widehat{\boldsymbol F}$は、ともに数値流束を表します。', 'ハットは、物理流束と区別するための記号です。')
        if part == 4:
            s = s.replace('その後にMUSCLでも調べます。','続く「MUSCLでも時間収束を測る」でMUSCLを調べます。')
            s = s.replace('E_t(', r'E_{\mathrm{time}}(')
            s = re.sub(r'(?<!\\bar)\\rho_{\\Delta t,i}', lambda m: r'\bar\rho_{\Delta t,i}', s)
            s = re.sub(r'(?<!\\bar)\\rho_{{\\rm ref},i}', lambda m: r'\bar\rho_{{\rm ref},i}', s)
        s = s.replace('### 基礎説明の参考資料','**基礎説明の参考資料**')
        c.source = s
    key = f'rev-symbols-{part}'
    if not any(c.id==key for c in nb.cells):
        text = r'''**記号と表の対応**

$\bar q_i$・$\bar{\boldsymbol U}_i$はセル平均、$f$・$\boldsymbol F$は物理流束、$\widehat F$・$\widehat{\boldsymbol F}$は数値流束です。半離散式の数値流束は各時刻の値、1ステップの更新式では各段階の重みを含む時間平均の近似を指します。

$C_{\mathrm{adv}}=a\Delta t/\Delta x$は符号付きの移流Courant数、$C_{\mathrm{CFL}}=\Delta t\max(|u|+c)/\Delta x$はEuler系の最大信号速度によるCFL数です。コードの`cfl`は後者の上限設定（スカラーでは$|C_{\mathrm{adv}}|$の上限）に対応します。

$E_{\mathrm{total}}$は連続方程式の厳密解に対する平均絶対誤差、$E_{\mathrm{time}}$は同じ半離散系の時間参照解に対する誤差です。表の`L1`・`L1_rho`は、それぞれスカラー量・密度の$E_{\mathrm{total}}$に対応します。密度以外の誤差列は同じ定義を該当する基本変数へ適用します。'''
        new=nbformat.v4.new_markdown_cell(text);new.id=key
        config_end = next(i for i,c in enumerate(nb.cells) if c.cell_type=='code' and 'CFG = {' in c.source)
        nb.cells.insert(config_end+1,new)


def normalize_headings(nb):
    major, minor, appendix = -1, 0, False
    for c in nb.cells:
        if c.cell_type != 'markdown': continue
        lines=[]; fence=False
        for line in c.source.splitlines():
            if line.startswith('```'): fence=not fence
            if not fence and line.startswith('## '):
                title=re.sub(r'^## (?:\d+\.\s*)?', '', line)
                appendix=any(k in title for k in ('動画','任意','結果の保存','発展実験','まとめ','次の学習','参考文献','確認問題','補足資料'))
                if appendix: line='## '+title
                else:
                    major+=1;minor=0;line=f'## {major}. {title}'
            elif not fence and line.startswith('### '):
                title=re.sub(r'^### (?:\d+(?:\.\d+)*[a-z]?\.?\s+)?', '', line)
                if appendix: line='### '+title
                else:
                    minor+=1;line=f'### {major}.{minor} {title}'
            lines.append(line)
        c.source='\n'.join(lines)


def patch_core():
    path=ROOT/'src/euler1d_core.py';s=path.read_text()
    s=s.replace('# Standalone module extracted from notebooks/euler1d_stage1_light.ipynb\n# Numerical algorithm is unchanged from the notebook.','"""Finite-volume Euler fluxes, states, exact references and conservative updates."""')
    s=s.replace('def __init__(self, stage, U=None, gamma=1.4, detail=""):', 'def __init__(self, stage, U=None, gamma=1.4, detail="", primitive=None):')
    marker='        message = f"{stage}: {detail} min(rho)={self.min_rho:.5e}, min(p)={self.min_p:.5e}"'
    add='''        if primitive is not None:
            W = np.asarray(primitive)
            self.min_rho = float(np.min(W[..., 0]))
            self.min_p = float(np.min(W[..., 2]))
            bad = (~np.all(np.isfinite(W), axis=-1)) | (W[..., 0] <= 0) | (W[..., 2] <= 0)
            indices = np.flatnonzero(bad)
            self.bad_index = int(indices[0]) if len(indices) else None
'''
    if 'if primitive is not None:' not in s:
        assert marker in s;s=s.replace(marker,add+marker)
    s=s.replace('raise NumericalFailure("MUSCL reconstruction", detail="A reconstructed rho or p is not positive.")', 'raise NumericalFailure("MUSCL reconstruction", detail="A reconstructed face rho or p is not positive.", primitive=state)')
    path.write_text(s)
    helper = ROOT / 'src/review_experiments.py'
    source = helper.read_text()
    old = '                ax.plot(r.x, r.U[:, 0], color=COLORS[j], ls=STYLES[j], marker=("o", "s", "^", "D")[j], ms=3, mfc="none", label=r.label)'
    new = '                mask = (r.x >= bounds[0]) & (r.x <= bounds[1])\n                ax.plot(r.x[mask], r.U[mask, 0], color=COLORS[j], ls=STYLES[j], marker=("o", "s", "^", "D")[j], ms=3, mfc="none", label=r.label)'
    if old in source:
        helper.write_text(source.replace(old, new))


def main():
    summary=[]
    for part,name in enumerate(NAMES):
        p=ROOT/'notebooks'/(name+'.ipynb');nb=nbformat.read(p,as_version=4)
        insert_blocks(nb,ROOT/'docs/review_revision'/f'{part+1:02d}.md')
        moves={0:[('basics-i-cells','scalar-003'),('basics-i-nonlinearity','scalar-014')],3:[('basics-iv-purpose','spatial-006'),('basics-iv-reconstruction','spatial-007')],4:[('basics-v-mol','temporal-006'),('basics-v-stages','temporal-007'),('basics-v-errors','temporal-008'),('basics-v-stability','temporal-011')]}
        for a,b in moves.get(part,[]):move_after(nb,a,b)
        cleanup(nb,part)
        if part==0:
            c=cell(nb,'scalar-029');c.source='## 任意：CFL数の比較\n\n`run_extras=True`でCFL数を変えた線形移流を比較します。非保存形の反例は本文で標準実行します。'
            c=cell(nb,'scalar-030')
            pos=c.source.find('    problem = sc.CASES["burgers_shock"]')
            if pos>=0:c.source=c.source[:pos].rstrip()
            cell(nb,'scalar-001').source = cell(nb,'scalar-001').source.split('\n\nスカラー版は終了時刻')[0] + '\n\nスカラー版は終了時刻に合わせて最後の刻みを短くします。Euler版の最後の2ステップを均等にする規則とは異なります。LFは刻みに依存する散逸を持つので、終了時刻付近の刻み方も比較条件に含めます。'
        if part==4:
            c=cell(nb,'temporal-021'); movie_index=nb.cells.index(c)
            movie_cells=nb.cells[movie_index:movie_index+2];del nb.cells[movie_index:movie_index+2]
            target=next(i for i,c in enumerate(nb.cells) if c.id=='temporal-025');nb.cells[target:target]=movie_cells
        refs=nb.cells[-1]
        if '[P1] Godunov' not in refs.source:
            refs.source+='\n\n'+(ROOT/'docs/review_revision/references.md').read_text()
        if '[R7]' in refs.source and '[TVD1]' in refs.source:
            refs.source=re.sub(r'\*\*\[TVD1\]\*\*.*?(?=\n\n)', '**[TVD1]** は上の[R7]（Harten 1983）と同じ文献です。', refs.source, flags=re.S)
        normalize_headings(nb)
        nbformat.validate(nb)
        assert sum(c.source.startswith('# ') for c in nb.cells if c.cell_type=='markdown')==1
        for c in nb.cells:
            if c.cell_type=='code':compile(c.source,p.name,'exec')
        nbformat.write(nb,p)
        summary.append(dict(notebook=p.name,cells=len(nb.cells),code_cells=sum(c.cell_type=='code' for c in nb.cells)))
    patch_core()
    rd=ROOT/'README.md';text=rd.read_text().replace('four_part_outputs','five_part_outputs')
    if '## 学習のための検証演習' not in text:
        text+='\n\n## 学習のための検証演習\n\n非保存形Burgersの波面速度、Roeのエントロピー補正、MUSCLの時間収束、全変動・値の範囲・波面幅を標準実行します。第V部のMUSCL時間参照は5,120・10,240ステップです。詳細な導出は折りたたみ補足に置き、各冊の本文は独立して読める構成です。\n\n追加の比較関数は `src/review_experiments.py` にあります。すべての出力は `five_part_outputs/` の各部のフォルダーへ保存します。\n'
    rd.write_text(text)
    (ROOT/'validation').mkdir(exist_ok=True)
    (ROOT/'validation/review_revision_structure.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
