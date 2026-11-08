"""Build the standalone A/B/C comparison; retain the original numerical core."""
from pathlib import Path
import hashlib
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAME = "euler1d_stage1_light.ipynb"
CORE_IDS = ("f208f769", "4fcea455", "4ad541c0", "e5516255", "df3fc6fa", "91d6c64e", "6297cce0", "42a4838f", "597be8ec")


def main():
    path = ROOT / "notebooks" / NAME
    nb = nbformat.read(path, as_version=4)
    cells = [c for c in nb.cells if not c.id.startswith("order-") and c.id != "intro62a8e093"]
    by_id = {c.id: c for c in cells}
    before = [by_id[k].source for k in CORE_IDS]
    core_id = hashlib.sha256("\n\n".join(before).encode()).hexdigest()
    by_id["5806c906"].source = r'''# 1次元Euler方程式：6方式とMUSCLの動画比較

数値流束、空間再構築、時間積分の役割を分けて考えながら、同じ問題の時間変化を比較します。計算に必要な関数はこのNotebookに内蔵しています。

| 比較 | 対象 | 固定するもの・変更するもの |
|:--|:--|:--|
| **A：6方式の基本比較** | LF、LW、Roe、LLF (Rusanov)、HLL、HLLC | 同じ問題・格子・CFL規則で、各方式の基本構成を比較 |
| **B：MUSCLの4流束比較** | Roe、LLF (Rusanov)、HLL、HLLC | MUSCL・同じリミッター・SSP-RK2を固定し、数値流束だけを変更 |
| **C：同じ流束の1次・2次精度比較** | 4流束をそれぞれ別の動画にする | 区分一定＋前進Eulerと、MUSCL＋SSP-RK2を比較 |

**AのLWはRichtmyerの2段階・2次精度法、他の5方式は1次精度の構成です。** B・Cの2次精度は滑らかな解での形式的な精度です。リミッターが働く極値や不連続付近では、その次数がそのまま成立するわけではありません。

Bは数値流束の違いを調べる比較です。Cでは空間再構築と時間積分を同時に変えるため、その差をMUSCLだけの効果、または時間積分だけの効果とは断定しません。各要素を独立に調べる方法は、5部構成教材の第IV部（空間再構築）・第V部（時間積分）で扱います。

最初に設定セルを確認し、上から順に実行してください。Aは7問題・14本、B・Cは標準ではエントロピー波・移動接触面・Sodの密度を使う計15本です。B・Cの問題・物理量・流束は`ORDER_CFG`で変更できます。'''
    by_id["ba899f54"].source = "## 数値計算コア\n\n保存変数、境界条件、数値流束、時間発展、厳密解、診断の関数を定義します。表示用の記録処理は、時間積分に変更を加えません。"
    by_id["bb25b6a6"].source = r'''## A：6方式の基本比較

LF・LW・Roe・LLF (Rusanov)・HLL (HLLE)・HLLCを同じ物理時刻で比較します。

| 方式 | 空間・時間の構成 | 滑らかな解での形式精度 |
|:--|:--|:--|
| LF | 古典的Lax–Friedrichs | 1次 |
| LW | Richtmyerの半ステップ予測を含む更新 | 2次 |
| Roe・LLF (Rusanov)・HLL・HLLC | 区分一定＋前進Euler | 1次 |

コードの`order=1`は「MUSCL再構築・SSP-RK2を追加しない」分岐を選ぶ指定で、LWを1次精度にする指定ではありません。`order=2`は4種類の近似リーマンソルバーに対してMUSCL＋SSP-RK2を選びます。LF・LWに`order=2`は指定しません。

MP4保存にはffmpegを使います。ffmpegが利用できないときはNotebook内のJavaScriptアニメーションに切り替わります。保存済みの出力を開く場合、数値計算の再実行は不要です。'''
    by_id["e208b72f"].source = by_id["e208b72f"].source.replace("## 表示を変更する例", "### Aの表示を変更する例").replace("### 実装の確認先", "### 実装の確認先")
    # Group A's data, numerical updates, labels, and settings remain unchanged.
    config_md = r'''### B・Cの設定と選択肢

`ORDER_CFG`はB・Cだけの設定です。格子数、CFL数、断熱指数、波速評価、表示時刻、フレーム数、MP4保存設定はAの`ANIM_CFG`を共通に使います。

| `problems`に指定する文字列 | 問題 |
|:--|:--|
| `entropy` | 滑らかなエントロピー波 |
| `contact_static` | 静止接触面 |
| `contact` | 移動接触面 |
| `sod` | Sod衝撃波管 |
| `double_rarefaction` | 二重希薄波：負圧による停止も比較 |
| `shu_osher` | Shu–Osher：高解像度の数値参照解を使用 |
| `lax` | Lax衝撃波管 |
| `transonic` | 音速をまたぐ希薄波 |

`fluxes`と`pair_solvers`には`Roe`, `LLF`, `HLL`, `HLLC`を指定します（表示名はLLF (Rusanov)）。`limiter`は`minmod`, `mc`, `unlimited`です。`unlimited`は制限なしの比較例で、振動や非物理的な状態による停止があり得ます。

表示量は`fields_by_problem`で問題別に指定します。`rho`は密度、`u`は速度、`p`は圧力、`e`は比内部エネルギーです。例えばSodの密度・速度・圧力を表示するには`"sod": ("rho", "u", "p")`とします。静止接触面では、密度を選ぶと違いを読み取りやすくなります。

`enabled=False`ならB・C全体を省略します。`run_four_flux_comparison`と`run_order_pairs`でB・Cを個別に省略できます。Shu–Osherを選ぶ場合は`n_shu_osher`と`n_reference`の整数格子比も確認してください。

**1次・2次の記録は別に保存し、MUSCLのリミッター・設定に合うものだけを再利用します。** 計算コードそのものを編集した場合は、`ANIM_CFG["reuse_data"]=False`、`ANIM_CFG["reuse_movies"]=False`として再計算してください。'''
    config_code = '''ORDER_CFG = {
    "enabled": True,
    "problems": ("entropy", "contact", "sod"),
    "fluxes": ("Roe", "LLF", "HLL", "HLLC"),
    "limiter": "minmod",
    "pair_solvers": ("Roe", "LLF", "HLL", "HLLC"),
    "run_four_flux_comparison": True,
    "run_order_pairs": True,
    "show_final_snapshot": True,
    "fields_by_problem": {
        "entropy": ("rho",),
        "contact_static": ("rho",),
        "contact": ("rho",),
        "sod": ("rho",),
        "double_rarefaction": ("rho", "p"),
        "shu_osher": ("rho",),
        "lax": ("rho",),
        "transonic": ("rho",),
    },
    "cache_revision": "muscl-order-comparison-v1",
}
'''
    def md(cid, source):
        return nbformat.v4.new_markdown_cell(source, id=cid)
    def code(cid, source):
        return nbformat.v4.new_code_cell(source, id=cid)
    at = next(i for i,c in enumerate(cells) if c.id == "e74b1cc1") + 1
    cells[at:at] = [md("order-settings-note", config_md), code("order-settings", config_code)]
    extension = (ROOT / "src/euler1d_stage1_order_comparison.py").read_text()
    cells.extend([
        md("order-methods-note", r'''## B・C：比較方法と計算記録

**Bでは同じMUSCL再構築・リミッター・SSP-RK2を使い、4種類の数値流束を比べます。** Cでは流束を固定し、1次精度構成と2次精度構成を比べます。LF・LWを一律に高次化した「6方式のMUSCL比較」ではありません。

セル平均からMUSCLで境界状態を求め、各数値流束を計算して、SSP-RK2で時間発展させます。B・CはAと同じ`simulate()`を使用します。

動画の表示時刻は共通ですが、各計算のCFL時間刻みはそれぞれの状態から決まります。受理された二つの保存状態を表示用に補間するだけで、計算へ戻しません。停止後の曲線は表示しません。Shu–Osher以外の黒線は保存変数をセル平均した厳密参照値、Shu–Osherは高解像度の数値参照解です。

B・Cは同じ問題・物理量について共通の固定軸範囲を使います。AはLF・LWも含むため、Aの軸範囲とは異なる場合があります。'''),
        code("order-functions", f'STAGE1_CORE_ID = "{core_id}"\n\n' + extension),
        code("order-compute", '''ORDER_BASE_DATA = {}
MUSCL_DATA = {}
if ORDER_CFG["enabled"]:
    validate_order_config(ANIM_CFG, ORDER_CFG)
    for key in ORDER_CFG["problems"]:
        ORDER_BASE_DATA[key] = ANIMATION_DATA[key] if key in ANIMATION_DATA else get_animation_data(key, ANIM_CFG)
        MUSCL_DATA[key] = get_muscl_animation_data(key, ANIM_CFG, ORDER_CFG, ORDER_BASE_DATA[key])
    ORDER_STATUS = order_status_table(ORDER_BASE_DATA, MUSCL_DATA)
else:
    print("B・Cは省略しています。ORDER_CFG['enabled']=Trueで有効にできます。")'''),
        md("order-b-title", '''## B：MUSCL＋SSP-RK2の4流束比較

滑らかなエントロピー波では振幅と位相、接触面では広がり、Sodでは衝撃波・接触面・希薄波を観察します。同じ再構築でも流束ごとの散逸や頑健性は異なります。動画の下には、静的なプレビューでも確認できるよう最終時刻の図を表示します。'''),
        code("order-b-show", '''if ORDER_CFG["enabled"] and ORDER_CFG["run_four_flux_comparison"]:
    for key, high in MUSCL_DATA.items():
        for field in ORDER_CFG["fields_by_problem"][key]:
            display(HTML("<h4>B: " + CASE_JA.get(key, PROBLEMS[key].title) + " / " + FIELD_JA[field] + "</h4>"))
            show_order_animation(ORDER_BASE_DATA[key], high, field, ANIM_CFG, ORDER_CFG, mode="muscl")'''),
        md("order-c-title", '''## C：同じ流束の1次精度と2次精度を比較

各流束に別々の動画を作ります。青い破線は区分一定＋前進Euler、赤橙の実線はMUSCL＋SSP-RK2、黒線は参照解です。同じ色の意味を全動画で維持します。

まず同じ流束の2本を比べ、波面の幅、振幅、振動、停止の有無を確認してください。その後にBへ戻り、高次精度化した4流束の違いを調べます。**Cは空間再構築と時間積分を同時に変更する比較です。** 空間精度だけの影響を分離する実験ではありません。'''),
        code("order-c-show", '''if ORDER_CFG["enabled"] and ORDER_CFG["run_order_pairs"]:
    for key, high in MUSCL_DATA.items():
        for solver in ORDER_CFG["pair_solvers"]:
            for field in ORDER_CFG["fields_by_problem"][key]:
                display(HTML("<h4>C: " + solver_display_name(solver, ANIM_CFG["wave_speeds"]) + " / " + CASE_JA.get(key, PROBLEMS[key].title) + " / " + FIELD_JA[field] + "</h4>"))
                show_order_animation(ORDER_BASE_DATA[key], high, field, ANIM_CFG, ORDER_CFG, mode="pair", solver=solver)'''),
        md("order-gallery-title", '''## A・B・Cの動画一覧

MP4を保存した場合、`stage1_orders_gallery.html`にA・B・Cの動画をまとめます。このHTMLは動画を内蔵しているため、Pythonを起動せずに再生できます。ffmpegを使わない場合は、各NotebookセルのJavaScript動画を使用してください。'''),
        code("order-gallery", '''if ORDER_CFG["enabled"]:
    all_base = dict(ANIMATION_DATA, **ORDER_BASE_DATA)
    ORDER_GALLERY = export_combined_order_gallery(all_base, MUSCL_DATA, ANIM_CFG, ORDER_CFG)
    if ORDER_GALLERY is not None:
        print("Combined gallery:", ORDER_GALLERY)'''),
        md("order-reading", '''## 比較のまとめ

Aは、古典的LF・LWと4流束の基本構成の比較です。LWだけが2次精度なので、すべてを同じ次数に揃えた実験ではありません。BはMUSCLと時間積分を揃えた流束比較、Cは同じ流束での高次精度化の総合比較です。

波形が鋭いこと、全変動が増えないこと、密度・圧力が正であること、厳密解に近いことは、それぞれ別に確認します。図だけで高次精度を証明せず、滑らかな問題の格子収束も併せて調べます。''')
    ])
    assert before == [by_id[k].source for k in CORE_IDS], "Numerical core changed"
    nb.cells = cells
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
            c.metadata.pop("execution", None)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(NAME, len(nb.cells), "cells; numerical core preserved; core SHA256", core_id)


if __name__ == "__main__":
    main()
