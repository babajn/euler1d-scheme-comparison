# 1次元保存則と流体スキーム：5部構成の数値計算演習

線形移流・Burgers方程式から圧縮性Euler方程式へ進み、数値流束・空間再構築・時間積分を分けて学ぶ日本語教材です。

## Notebook

| 部 | Notebook | 比較の中心 |
|:--|:--|:--|
| I | [移流方程式とBurgers方程式](notebooks/01_advection_burgers.ipynb) | 一次風上差分、LF、LW、Godunov、保存性・エントロピー条件 |
| II | [Euler方程式の古典的数値解法](notebooks/02_euler_classical.ipynb) | 3つの特性速度、LF・LW |
| III | [Euler方程式の近似リーマンソルバー](notebooks/03_euler_riemann_solvers.ipynb) | Roe・LLF (Rusanov)・HLL・HLLCを区分一定＋前進Eulerで比較 |
| IV | [空間再構築](notebooks/04_spatial_reconstruction.ipynb) | 各流束で区分一定・MUSCL-minmod・MUSCL-MC・制限なし線形再構築を比較。SSP-RK2に固定 |
| V | [時間積分](notebooks/05_time_integration.ipynb) | 各流束で前進Euler・SSP-RK2・SSP-RK3を比較。空間離散化を固定 |

第IV部では流束ごとに図を分け、時間積分を固定します。第V部では時間収束試験と衝撃波を含む実用問題を分け、最後に流束・空間・時間の総合比較を行います。

## 取得・実行

**Notebookだけでなく、`src/`を含むリポジトリ一式を取得してください。** 各Notebookは設定セルから順に実行します。

```sh
git clone https://github.com/babajn/euler1d-scheme-comparison.git
cd euler1d-scheme-comparison
python -m pip install -r requirements.txt
python -m notebook notebooks/04_spatial_reconstruction.ipynb
```

第V部を開く場合：

```sh
python -m notebook notebooks/05_time_integration.ipynb
```

clone済みの作業ディレクトリでは、ローカルの変更を保存したうえで`git pull --ff-only`を使って更新します。ZIPを展開したディレクトリはGit作業ツリーではないため、その中では`git pull`を使いません。

## 独立した設定

第IV・V部の計算は`src/euler1d_mol.py`で行います。

| 項目 | 値 |
|:--|:--|
| 数値流束 `solver` | `Roe`, `LLF`, `HLL`, `HLLC` |
| 空間再構築 `reconstruction` | `constant`, `muscl` |
| リミッター `limiter` | `minmod`, `mc`, `unlimited`（constantでは不使用） |
| 時間積分 `time_integrator` | `forward_euler`, `ssprk2`, `ssprk3` |

`LF`とRichtmyerの`LW`は時間刻みを含む古典的更新として第II部で扱い、独立な半離散流束の選択肢には含めません。
表示ラベルは**LLF (Rusanov)**、内部識別子は`LLF`です。HLLは標準でEinfeldt波速を使用し、**HLL (HLLE)**と表示します。

第IV部では`spatial_methods`、第V部では`time_integrators`が比較する候補です。設定辞書は1項目1行で記載しています。

## 時間精度の測定

第V部の時間収束試験は、同じ格子・数値流束・空間再構築で細かい時間刻みを使った数値参照解と比較します。空間格子は細かくしません。

標準設定は48セル、終了時刻0.2、40・80・160・320ステップ、SSP-RK3参照計算は1280・2560ステップです。各方式で同じ固定時間刻みを用い、CFL上限違反があれば刻みを勝手に変更せず停止します。

`E_time`（同じ半離散系の参照解との差）と`E_total`（連続方程式の厳密解との差）を別々に記録します。2つの時間参照解の差も確認します。実際の流体問題ではMUSCL-minmodとCFL規則を固定し、時間積分を比較します。

## 動画・追加実験・出力

全5冊で`run_movies=True`が標準です。**標準設定で実行した図・表・動画の出力を各Notebookに保存しています。** 静止図・表はGitHub上でも確認できます。JavaScriptアニメーションはGitHubのプレビューでは再生されない場合があるため、JupyterでNotebookを開いてください。動画表示にffmpegは不要です。Notebookを再実行すると表示結果が更新されます。

第I部は`movie_case`、第II〜V部は`movie_problem`で対象を選びます。設定セルの直前に全選択肢を記載しています。第IV・V部の`movie_fluxes`は動画を作る流束、`movie_field`は`rho`・`u`・`p`・`e`です。標準はHLLC・Sod・密度です。各流束に別の動画を作ります。

Shu–Osherの簡易動画には対応していません。`run_shu_osher=True`で数値参照解を伴う静止図比較を実行します。`run_extras=True`は追加実験を有効にします。

CSV・NPZ・JSONの保存先：

```text
four_part_outputs/scalar/       # 第I部
four_part_outputs/classical/    # 第II部
four_part_outputs/modern/       # 第III部
five_part_outputs/spatial/      # 第IV部
five_part_outputs/temporal/     # 第V部
```

同じ冊を再実行すると、その保存先の結果が更新されます。時間参照解と通常の計算結果は設定・指標とともに保存します。

## 出力付きNotebookの更新

GitHubの **Actions → Render and save notebook outputs → Run workflow** から、5冊をそれぞれ独立したカーネルで標準設定により実行し、図・表・アニメーションの出力を保存できます。全5冊のセル実行と画像生成が成功した場合にのみ、`main`のNotebookをまとめて更新します。再実行にはGitHub Actionsの書き込み権限が必要です。

## 数値的な注意

- 比較は同じ問題・時刻・格子・熱力学設定で行います。滑らかな波の収束次数を不連続点の局所次数と混同しません。
- MUSCL＋SSP-RK3を使っても、全体が空間・時間ともに3次精度になるわけではありません。
- 制限なし再構築はリミッターの役割を学ぶ反例です。負の再構築状態が生じた場合は停止として記録します。
- 密度・圧力の下限補正、試行ステップの自動再試行、流束の自動切替はありません。失敗した途中状態を完走した最終時刻の図に混ぜません。
- 動画では受理済み状態を表示時刻へ補間します。時間積分のステップは変更せず、停止後の曲線は表示しません。
- 1次元・一様格子・単一の理想気体の学習用実装です。PPM・WENO、真空、重力、冷却、MHD、多次元、包括的な正値性保証は対象外です。

## 検証

```sh
python scripts/check_numerics.py
```

スカラー11・Euler9・半離散法8グループの検証を行います。Notebookの実行条件と検証範囲は[validation/README.md](validation/README.md)に記載しています。

著作権者による明示的なライセンス指定はまだありません。
