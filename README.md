# 流体スキームを学ぶ：移流・Burgers・Euler

1次元の保存則を、スカラー方程式から圧縮性Euler方程式まで段階的に学ぶ日本語Notebook教材です。

## 4つのNotebook

| 部 | Notebook | 内容 |
|:--|:--|:--|
| I | [移流方程式とBurgers方程式](notebooks/01_advection_burgers.ipynb) | 一次風上差分・LF・LW・Godunov、CFL、衝撃波・希薄波、保存則・エントロピー条件 |
| II | [Euler方程式の古典的数値解法](notebooks/02_euler_classical.ipynb) | 移流・圧縮・圧力勾配、3つの特性速度、一様背景まわりの線形化、LF・LW |
| III | [Euler方程式の近似リーマンソルバー](notebooks/03_euler_riemann_solvers.ipynb) | Roe、LLF (Rusanov)、HLL (HLLE)、HLLCを区分一定＋前進Eulerで比較 |
| IV | [Euler方程式の高次精度数値解法](notebooks/04_euler_high_accuracy.ipynb) | 同じ4流束＋MUSCL＋SSP-RK2、1次精度との比較 |

数値流束、空間再構築、時間積分は分けて考えます。第III部の1次精度構成は流束を比較する基準であり、研究用の最終設定を意味しません。LWは滑らかな解に対して2次精度です。

## 取得・実行

**リポジトリ一式**を取得し、`notebooks/` と `src/` の配置を保ってください。各Notebookの設定セルを確認し、上から実行します。

```sh
git clone https://github.com/babajn/euler1d-scheme-comparison.git
cd euler1d-scheme-comparison
python -m pip install -r requirements.txt
python -m notebook notebooks/01_advection_burgers.ipynb
```

GitHubの「Code → Download ZIP」でも取得できます。既にclone済みの場合は、リポジトリ内でローカルの変更を保存してから `git pull --ff-only` で更新します。

## 設定・動画・保存先

- `run_convergence`：滑らかな波の格子収束試験。
- `run_movies`：JavaScriptによる時間発展の比較。ffmpegは不要です。第I部は `movie_case`、第II～IV部は `movie_problem` で問題を選びます。
- `run_extras`：CFL依存性などの追加実験。第II～IV部の `run_shu_osher` は高解像度参照解を使うShu–Osher実験です。

動画・追加実験・Shu–Osherは標準では無効です。配布Notebookの出力は空で、実行後に図・表・動画が表示されます。結果は次の場所に保存します。

```text
four_part_outputs/
  scalar/
  classical/
  modern/
  high_accuracy/
```

CSVに誤差・保存収支、NPZに数値解、JSONに設定を保存します。第II～IV部の動画ヘルパーは厳密参照解のある問題に限り、Shu–Osher動画には対応しません。

## 数値的な注意点

線形移流・Burgersの未知量はセル平均です。Burgersの正弦波で特性が交差した後は、逆特性写像を厳密解として使わず、数値解のみを表示します。Roeの補正なしの流束は、保存則だけでは正しいエントロピー解を選べないことを示す反例です。

Eulerの低密度問題で負の密度・圧力を検出した場合は停止情報を記録し、floor・自動再試行・別ソルバーへの切替で隠しません。途中で停止した解を最終時刻の比較図には混ぜません。動画の状態補間は表示専用で、積分の時間刻みは変えません。

`LLF (Rusanov)` は表示名で、内部識別子は `LLF` です。HLLは標準でEinfeldt波速を使います。Shu–Osherの参照線は高解像度数値解であり、厳密解ではありません。

## コードと検証

`src/scalar1d.py` がスカラー方程式、`src/euler1d_core.py` がEulerの計算、`src/euler1d_study.py` がEulerの比較・描画を担当します。

```sh
python scripts/check_numerics.py
```

スカラー11グループ・Euler9グループの基本検証を実行します。4冊の標準設定と追加実験の検証範囲は [validation/README.md](validation/README.md) に記載しています。これは学習用の実装であり、任意の問題に対する正値性や研究用コードとしての全面的な検証を意味しません。

PPM・WENO・SSP-RK3やPyClawによる比較は対象外です。参考文献と導出は各Notebookにあります。
