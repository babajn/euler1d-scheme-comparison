"""Insert the TVD teaching sections without altering existing numerical kernels.

Idempotent: existing TVD sections are retained, including their saved outputs.
Use from the repository root before executing the five main notebooks.
"""
from pathlib import Path
import json

I_DEFINITION = r'''### 2.2a 全変動とTVD条件

ここでは**周期境界の1次元スカラー量のセル平均**を考えます。全変動（total variation）は

$$TV(\bar q^n)=\sum_{i=0}^{N-1}|\bar q_i^n-\bar q_{i-1}^n|,\qquad \bar q_{-1}^n=\bar q_{N-1}^n$$

です。絶対値を取るため、増加と減少は打ち消し合いません。最後と最初のセルの差も含めます。$\Delta x$は掛けません。これは滑らかな関数の$\int\left|\frac{\partial q}{\partial x}\right|\,dx$に対応します。
高さ1の孤立した矩形波の全変動は、上り1と下り1を合わせて2です。

$$\boxed{TV(\bar q^{n+1})\le TV(\bar q^n)}$$

が各ステップで成り立つ性質を**TVD（Total Variation Diminishing：全変動非増加）**と呼びます。減少せず等しい場合も含みます。TVDは特定のアルゴリズム名ではなく、対象方程式・境界条件・時間刻みの条件の下で更新全体が満たす性質です。その性質を満たすスキームをTVD法と呼びます。[TVD1]

TVD性、保存性、精度、エントロピー条件、正値性は別の性質です。強く拡散する解もTVDになり得ます。全体のTVが減ることだけから、局所の振動が一切ないことや、Euler方程式の全変数が正しいことまでは結論できません。流入境界から新しい変動が入る問題や生成項を持つ問題には、この周期境界の不等式をそのまま使いません。

**一次風上差分での確認。** $a>0$、$C=a\Delta t/\Delta x$、$D_i^n=\bar q_i^n-\bar q_{i-1}^n$とおくと、

$$D_i^{n+1}=(1-C)D_i^n+CD_{i-1}^n.$$

$0\le C\le1$なら三角不等式と周期性から

$$\sum_i|D_i^{n+1}|\le(1-C)\sum_i|D_i^n|+C\sum_i|D_{i-1}^n|=TV(\bar q^n).$$

したがって一次風上差分＋前進Eulerはこの条件でTVDです。$a<0$も添字の向きを反転すれば同様です。線形移流のLFも$|C|\le1$で非負の重み付き平均となりTVDです。LWは一般にはTVDでなく、不連続近傍で全変動を増やすことがあります。$|C|=1$の厳密なセル移動は例外的な場合です。
'''
I_EXPERIMENT = r'''### 2.4a 演習：全変動の時間変化を測る

周期境界の矩形波をUpwind・LF・LWで移流します。既存のスカラーソルバーが受理した**すべての時間ステップ**を記録し、$TV(\bar q^n)$を測ります。動画用の補間状態は使いません。

表の`max_step_increase`は$\max_n\max[0,TV(\bar q^{n+1})-TV(\bar q^n)]$、`steps_with_increase`は丸め誤差の許容値を超えた増加の回数です。`TV_check_tol`に許容値を表示します。
**最終値が初期値以下かどうかだけでは、各ステップのTVD条件を確認したことにはなりません。**

以下の`TVD_CFG`はこの演習だけの設定です。速度、格子数、終了時刻は3方式で共通です。有限個の初期条件による数値確認は、任意の初期条件に対する証明の代わりではありません。
'''
I_CODE = '''import tvd_study as tvd

TVD_CFG = {
    "n": 100,
    "cfl": 0.4,
    "t_end": 0.5,
}
if CFG["run_tvd"]:
    print("TVD checks:", len(tvd.run_tests()))
    tvd_runs = [sc.simulate(sc.CASES["advection_box"], method, n=TVD_CFG["n"], cfl=TVD_CFG["cfl"], t_end=TVD_CFG["t_end"], record=True) for method in ("Upwind", "LF", "LW")]
    assert all(run["status"] == "ok" for run in tvd_runs)
    tvd_histories = [(sc.LABELS[run["scheme"]], run["times"], run["states"]) for run in tvd_runs]
    tvd.plot_histories(tvd_histories, "Periodic square pulse: upwind, LF and LW")
    tvd_rows = [tvd.diagnostics(times, states, label) for label, times, states in tvd_histories]
    show_table(tvd_rows)
    if CFG["export_results"]:
        tvd.export_diagnostics(tvd_rows, OUTPUT_DIR / "tvd_classical.csv")
'''
I_ENTROPY = r'''### TVD条件とエントロピー条件は別である

上の$(-1,1)$のBurgers問題では、補正なしRoe流束が残す静止した膨張衝撃波も、全変動は初期値と同じ2です。正しい希薄波の全変動も2です。この問題は流出境界なので、ここでいうTVは端から端までの隣接差の和です。

したがって、**この計算でTVが増えないという事実だけでは、正しいエントロピー解を選べたことを意味しません。** 特定の計算が不等式を満たすことと、スキームが任意の許容初期値に対してTVDであることも区別します。
'''
IV_THEORY = r'''### 2.1 TVDリミッターとは何か

**MUSCLは再構築の方法、minmod・MCは勾配制限の方法、TVDは時間更新全体の性質**です。スカラー保存則について、適切な数値流束・時間積分・CFL条件と組み合わせてTVDスキームを構成するため、minmodやMCはTVDリミッターと呼ばれます。リミッター単独が時間発展をするわけではありません。[TVD1, TVD3]

周期境界では

$$TV(\bar q)=\sum_i|\bar q_i-\bar q_{i-1}|,\qquad TV(\bar q^{n+1})\le TV(\bar q^n)$$

を用います。空間精度を上げるための直線の傾きを制限すると、数値振動を抑える一方で拡散や局所的な精度低下が起こります。特にminmod・MCは両側の増分が異符号の極値で傾きを0にします。MCはminmodより大きい傾きを許す領域がありますが、常に小さい誤差を与えるという意味ではありません。

**ここからのTVD演習はEuler系ではなく、一定速度のスカラー線形移流です。** 系の波の結合と境界からの流入を除き、再構築の効果を分けて観察します。Euler系で$TV(\rho)$だけを測っても、系全体のTVD性や圧力の正値性の証明にはなりません。
'''
IV_BOUND = r'''### 2.2 このスカラー演習に対するTVDの十分条件

以下は、この演習の更新式に対する導出です。$a>0$の線形移流に、再構築した左状態を使う風上流束を適用します。$s_i$はセル幅を掛けた傾きの増分です。

$$F_{i+1/2}=a\left(\bar q_i+\frac{s_i}{2}\right),\qquad
\bar q_i^{n+1}=\bar q_i^n-C\left[D_i+\frac{s_i-s_{i-1}}2\right],\qquad D_i=\bar q_i-\bar q_{i-1}.$$

minmod・MCは、$D_i\ne0$のとき

$$0\le\frac{s_i}{D_i}\le2,\qquad 0\le\frac{s_{i-1}}{D_i}\le2$$

を満たします。$D_i=0$なら両方の傾きは0です。したがって

$$\beta_i=C\left(1+\frac{s_i-s_{i-1}}{2D_i}\right),\qquad 0\le\beta_i\le2C$$

と置き、$D_i=0$では$\beta_i=C$と定めれば、更新は$\bar q_i^{n+1}=\bar q_i^n-\beta_iD_i^n$です。隣接差は

$$D_i^{n+1}=(1-\beta_i)D_i^n+\beta_{i-1}D_{i-1}^n$$

なので、$0\le C\le1/2$なら係数が非負になり、総和を取って$TV(\bar q^{n+1})\le TV(\bar q^n)$が得られます。負の速度では添字を反転します。

**この実装では、minmod・MC＋風上流束＋前進Eulerに$|a|\Delta t/\Delta x\le1/2$が共通の十分条件です。** 最も緩い限界を求めたわけではなく、他のMUSCL実装やEuler系へそのまま移せる値でもありません。区分一定＋前進Eulerの十分条件は$|a|\Delta t/\Delta x\le1$です。

次の演習はSSP-RK2に固定します。SSP-RK2は、各段階の前進Eulerが上の条件でTVDなら、その性質を継承します。証明は[第V部](05_time_integration.ipynb#tvd-ssp)で扱います。
'''
IV_EXPERIMENT = r'''### 2.3 演習：再構築を変えて全変動を比較する

周期境界の同じ矩形波に対して、**風上流束とSSP-RK2を固定**し、区分一定・minmod・MC・制限なし線形再構築を比べます。`tvd_study.py`はこのスカラー演習のためのコードで、Eulerの計算本体は変更しません。

標準CFL数0.4は上記の共通十分条件の範囲内です。表には全時間ステップのTV増加を記録します。**同じTVでも波形の幅や誤差は異なり得る**ため、TVだけを精度の指標と解釈しないでください。
'''
IV_CODE = '''import tvd_study as tvd
import scalar1d as sc

TVD_CFG = {
    "n": 100,
    "cfl": 0.4,
    "t_end": 0.5,
}
if CFG["run_tvd"]:
    edges = np.linspace(0.0, 1.0, TVD_CFG["n"] + 1)
    q0 = sc.exact_cell_average(sc.CASES["advection_box"], edges, 0.0)
    tvd_histories, tvd_rows = [], []
    for rec in ("constant", "minmod", "mc", "unlimited"):
        run = tvd.simulate(q0, reconstruction=rec, integrator="ssprk2", cfl=TVD_CFG["cfl"], t_end=TVD_CFG["t_end"])
        label = "Piecewise constant" if rec == "constant" else "Linear: " + rec
        tvd_histories.append((label, run["times"], run["states"]))
        tvd_rows.append(dict(tvd.diagnostics(run["times"], run["states"], label), CFL=run["CFL"]))
    tvd.plot_histories(tvd_histories, "Scalar advection: reconstruction at fixed SSP-RK2")
    show_table(tvd_rows)
    if CFG["export_results"]:
        tvd.export_diagnostics(tvd_rows, OUTPUT_DIR / "tvd_reconstruction.csv")
'''
V_PROOF = r'''<a id="tvd-ssp"></a>
### 2.1 SSP法がTVD性を引き継ぐ理由

ここでは周期境界のスカラー半離散系を考えます。前進Eulerの更新を

$$\mathcal E_{\Delta t}(q)=q+\Delta t\,\mathcal L(q)$$

とし、ある$\Delta t_{\rm FE}$の下で、すべての対象状態について

$$TV(\mathcal E_{\Delta t}(q))\le TV(q),\qquad 0\le\Delta t\le\Delta t_{\rm FE}$$

が成立すると仮定します。これがSSP法の保証を使うための前提です。[TVD2, T2]

全変動は凸関数なので、$0\le\theta\le1$に対して

$$TV(\theta v+(1-\theta)w)\le\theta TV(v)+(1-\theta)TV(w)$$

が成り立ちます。これは各隣接差に三角不等式を適用した結果です。

**SSP-RK2：** $q^{(1)}=\mathcal E_{\Delta t}(q^n)$、$q^{n+1}=\tfrac12q^n+\tfrac12\mathcal E_{\Delta t}(q^{(1)})$より、

$$TV(q^{n+1})\le\tfrac12TV(q^n)+\tfrac12TV(\mathcal E_{\Delta t}(q^{(1)}))\le\tfrac12TV(q^n)+\tfrac12TV(q^{(1)})\le TV(q^n).$$

**3段階SSP-RK3：** $q^{(1)}=\mathcal E_{\Delta t}(q^n)$、$q^{(2)}=\tfrac34q^n+\tfrac14\mathcal E_{\Delta t}(q^{(1)})$、$q^{n+1}=\tfrac13q^n+\tfrac23\mathcal E_{\Delta t}(q^{(2)})$にも同じ議論を適用すると、$TV(q^{(2)})\le TV(q^n)$、$TV(q^{n+1})\le TV(q^n)$です。

ここで使う2段階SSP-RK2と3段階SSP-RK3のSSP係数はともに1です。したがって、**前進EulerのTVD条件$\Delta t\le\Delta t_{\rm FE}$を同じまま引き継ぎます。** 再構築・境界条件・流束は各段階で計算し直します。

SSPは「前進Eulerが持つ強安定性を保つ」という意味で、時間積分だけで非TVDな空間離散化をTVDにする保証ではありません。前進Eulerの条件が段階ごとに状態依存する場合は、中間状態でも条件を満たす必要があります。

この説明は、連続方程式に対する誤差の次数や線形絶対安定性とは別です。また、Euler系での成分別リミッターの使用だけでは、前提のTVD性も密度・圧力の正値性も自動的には得られません。
'''
V_EXPERIMENT = r'''### 2.2 演習：成立条件と反例を分けて確かめる

以下は**周期境界のスカラー線形移流**であり、Euler系の計算ではありません。

まずminmod再構築と風上流束を固定し、CFL数0.4で前進Euler・SSP-RK2・SSP-RK3の全変動を比較します。第IV部で導いた十分条件$|a|\Delta t/\Delta x\le1/2$を満たします。

次に、同じ空間離散化のまま時間刻みだけを変える反例として、区分一定＋風上流束についてCFL数0.4と1.2を比較します。後者は前進Eulerの十分条件$|a|\Delta t/\Delta x\le1$を外し、初期の6ステップだけを表示します。**SSP法でも、前提条件を外したこの例ではTVが増えます。** 条件外で常に増えるという一般的主張ではありません。

さらに制限なし線形再構築＋SSP-RK3をCFL数0.4で調べます。「SSP」という名前だけでは空間離散化の振動を防げないことを確認してください。

小さい刻みでは精度も変わるため、ここでは誤差の大小ではなく、各ステップのTV不等式の成否を見ます。数値実験は上の証明を補足するもので、証明そのものではありません。
'''
V_CODE = '''import tvd_study as tvd
import scalar1d as sc

TVD_CFG = {
    "n": 100,
    "cfl": 0.4,
    "t_end": 0.5,
    "counterexample_cfl": 1.2,
    "counterexample_steps": 6,
}
if CFG["run_tvd"]:
    edges = np.linspace(0.0, 1.0, TVD_CFG["n"] + 1)
    q0 = sc.exact_cell_average(sc.CASES["advection_box"], edges, 0.0)
    tvd_rows, tvd_histories = [], []
    for ti in tvd.INTEGRATORS:
        run = tvd.simulate(q0, reconstruction="minmod", integrator=ti, cfl=TVD_CFG["cfl"], t_end=TVD_CFG["t_end"])
        label = mol.TIME_LABELS[ti]
        tvd_histories.append((label, run["times"], run["states"]))
        tvd_rows.append(dict(tvd.diagnostics(run["times"], run["states"], label), reconstruction="minmod", CFL=run["CFL"]))
    tvd.plot_histories(tvd_histories, "TVD inheritance: minmod, CFL = 0.4")
    # All time methods have the same dt within each comparison.
    short_end = TVD_CFG["counterexample_steps"] * TVD_CFG["counterexample_cfl"] / TVD_CFG["n"]
    for cfl in (0.4, TVD_CFG["counterexample_cfl"]):
        histories = []
        for ti in tvd.INTEGRATORS:
            run = tvd.simulate(q0, reconstruction="constant", integrator=ti, cfl=cfl, t_end=short_end)
            label = mol.TIME_LABELS[ti]
            histories.append((label, run["times"], run["states"]))
            tvd_rows.append(dict(tvd.diagnostics(run["times"], run["states"], label), reconstruction="constant", CFL=run["CFL"]))
        tvd.plot_histories(histories, f"Piecewise constant: requested CFL = {cfl:g}")
    unlimited = tvd.simulate(q0, reconstruction="unlimited", integrator="ssprk3", cfl=0.4, t_end=TVD_CFG["t_end"])
    tvd_rows.append(dict(tvd.diagnostics(unlimited["times"], unlimited["states"], "SSP-RK3, unlimited"), reconstruction="unlimited", CFL=unlimited["CFL"]))
    show_table(tvd_rows)
    if CFG["export_results"]:
        tvd.export_diagnostics(tvd_rows, OUTPUT_DIR / "tvd_time_integrators.csv")
'''
REFERENCES = '''

**[TVD1]** Harten, A. (1983), *High resolution schemes for hyperbolic conservation laws*, Journal of Computational Physics, 49, 357–393. TVD条件と高解像度スキーム。
https://doi.org/10.1016/0021-9991(83)90136-5

**[TVD2]** Ketcheson, Higgs, Gottlieb, *Strong Stability Preserving Methods*. SSP法と前進Eulerの凸安定性を継承する条件。
https://www.sspsite.org/

**[TVD3]** Clawpack developers, *Pyclaw Limiters*. minmod・MCなどのリミッターの公式資料。ここではClawpackを呼び出さず、本文に記載したスカラー更新式を使います。
https://www.clawpack.org/pyclaw/evolve/limiters.html
'''


def cell(kind, cid, source):
    result = {"cell_type": kind, "id": cid, "metadata": {}, "source": source}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def insert_after(nb, anchor, cells):
    ids = {c["id"] for c in nb["cells"]}
    if all(c["id"] in ids for c in cells):
        return
    if any(c["id"] in ids for c in cells):
        raise RuntimeError("Partially inserted TVD section; inspect before retrying.")
    indices = [i for i, c in enumerate(nb["cells"]) if c["id"] == anchor]
    if len(indices) != 1:
        raise RuntimeError(f"Missing or duplicated anchor {anchor}.")
    i = indices[0] + 1
    nb["cells"][i:i] = cells


def amend(path, additions, reference_id):
    nb = json.loads(path.read_text(encoding="utf-8"))
    before_code = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    for anchor, cells in additions:
        insert_after(nb, anchor, cells)
    for c in nb["cells"]:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and "CFG = {" in s and '"run_tvd"' not in s and not c["id"].startswith("tvd-"):
            s = s.replace("CFG = {\n", 'CFG = {\n    "run_tvd": True,  # スカラー方程式によるTVDの確認演習\n', 1)
            c["source"] = s
        if c["id"] == reference_id and "**[TVD1]**" not in s:
            c["source"] = s + REFERENCES
    # Existing code is unchanged apart from the additional experiment switch.
    switch = '    "run_tvd": True,  # スカラー方程式によるTVDの確認演習\n'
    for c in nb["cells"]:
        if c["id"] in before_code:
            assert "".join(c["source"]).replace(switch, "") == before_code[c["id"]].replace(switch, "")
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.name, "TVD cells:", sum(c["id"].startswith("tvd-") for c in nb["cells"]))


def main():
    root = Path(__file__).resolve().parents[1]
    amend(root / "notebooks/01_advection_burgers.ipynb", [
        ("scalar-007", [cell("markdown", "tvd-i-definition", I_DEFINITION)]),
        ("scalar-011", [cell("markdown", "tvd-i-experiment", I_EXPERIMENT), cell("code", "tvd-i-code", I_CODE)]),
        ("scalar-025", [cell("markdown", "tvd-i-entropy", I_ENTROPY)]),
    ], "scalar-033")
    amend(root / "notebooks/04_spatial_reconstruction.ipynb", [
        ("spatial-008", [cell("markdown", "tvd-iv-theory", IV_THEORY), cell("markdown", "tvd-iv-bound", IV_BOUND), cell("markdown", "tvd-iv-experiment", IV_EXPERIMENT), cell("code", "tvd-iv-code", IV_CODE)]),
    ], "spatial-027")
    amend(root / "notebooks/05_time_integration.ipynb", [
        ("temporal-007", [cell("markdown", "tvd-v-proof", V_PROOF), cell("markdown", "tvd-v-experiment", V_EXPERIMENT), cell("code", "tvd-v-code", V_CODE)]),
    ], "temporal-028")


if __name__ == "__main__":
    main()
