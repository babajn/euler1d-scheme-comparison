# 1D Euler equation: comparison of numerical schemes

学習用の1次元圧縮性Euler方程式ソルバーと、日本語の比較演習です。

## 推奨：3冊に分けて学ぶ

| Notebook | 比較対象 | この教材での構成 | 主な学習目的 |
|:--|:--|:--|:--|
| [01 古典的手法](notebooks/01_classical_methods.ipynb) | LF、LW | 古典的LF、Richtmyerの2段階LW | 数値拡散と数値分散・振動の違い |
| [02 現代の実用的手法](notebooks/02_practical_riemann_solvers.ipynb) | Roe、LLF (Rusanov)、HLL、HLLC | 区分一定再構成＋前進Euler | 1次精度に統一して数値流束の違いを調べる |
| [03 実用的な高精度手法](notebooks/03_high_accuracy_methods.ipynb) | 第II部の4流束 | 共通のMUSCL (minmod)＋SSP-RK2 | 同じ流束を高次精度化した効果を調べる |

各Notebookは冒頭の設定セルに従って計算を実行します。第III部のHLLCの1次・2次精度比較も、その冊の中で両方を計算します。

計算コードは複製せず、`src/euler1d_core.py` と `src/euler1d_study.py` を共用します。初期条件、厳密解、境界条件、数値流束、時間発展、失敗判定は共通コードで定義しています。

これは**精度の低い順の分類ではありません**。古典的LWは滑らかな解で2次精度です。第II部の「実用的」は数値流束の位置づけを指し、学習用の1次精度構成全体を最終的な研究用設定として推奨する意味ではありません。第III部は第II部と排他的な分類ではなく、同じ流束に高次精度化を組み合わせます。

各冊で、滑らかなエントロピー波、静止・移動する接触不連続面、Sod衝撃波管、二重希薄波と格子収束試験を扱います。図には原則としてその冊の手法だけを表示します。数式・初期条件・誤差基準と参考文献も各冊に収録しています。

空間再構築と時間積分法を独立に変える実験は**次の学習計画**です。PPM、WENO、SSP-RK3、PyClawはこの教材の対象外です。

## 取得・実行

各Notebookは **リポジトリ一式**を取得して使います。Notebookだけを移動せず、`notebooks/` と `src/` の配置を保ってください。

```sh
git clone https://github.com/babajn/euler1d-scheme-comparison.git
cd euler1d-scheme-comparison
python -m pip install -r requirements.txt
python -m notebook notebooks/01_classical_methods.ipynb
```

`git clone`で取得済みの場合は、そのリポジトリのディレクトリ内で、ローカルの変更を保存したうえで更新してください。ZIP展開したディレクトリでは`git pull`は使いません。

```sh
git pull --ff-only
python -m notebook notebooks/01_classical_methods.ipynb
```

第II部・第III部を開くときはファイル名を変更してください。Gitを使わない場合はGitHubの「Code → Download ZIP」で取得できます。

各冊でカーネルを再起動し、上から実行してください。設定変更後も再起動して全セルを実行すると、図・表の条件を揃えられます。

### 設定・動画・追加実験

最初の `CFG` セルで設定します。基本問題と格子収束試験は標準で実行し、以下は初期状態では無効です。

- `run_movies=True`：`movie_problem` の1問題を、その冊の手法だけで動画表示します。初期値はSodです。JavaScript表示のためffmpegは不要です。この動画機能はShu–Osher問題には対応しません。
- `run_shu_osher=True`：その冊の手法をShu–Osher問題で比較します。2解像度の数値参照解もその冊の中で計算し、参照解同士の差を表示します。
- `run_extras=True`：各冊のLax衝撃波管に加え、第I部ではLFのCFL数依存性、第II部ではLFとLLFのCFL数依存性およびRoeのエントロピー補正を調べます。第II部で補助比較に必要なLFも、その冊の中で計算します。

`export_results=True` のとき、結果の保存先は次のように分かれます。

```text
three_part_outputs/
├── classical/
├── modern/
└── high_accuracy/
```

各フォルダーに `metrics.csv`、`states.npz`、`settings.json` を保存します。他の冊の結果は上書きしません。同じ冊を再実行すると、その冊の出力は更新します。CSVには内部識別子 `solver`、表示名 `method`、学習分類 `part` を別々に記録します。参照解・部間比較も有効にした場合は、その計算結果を含みます。

## 関連する演習・動画生成ツール

- [総合演習＋動画生成](notebooks/euler1d_complete_light.ipynb)：基本6方式の総合比較、MUSCL、誤差解析、動画生成。
- [Stage 1動画専用](notebooks/euler1d_stage1_light.ipynb)：6方式を重ねる時間発展比較。

これらのNotebookは計算コアを内蔵しています。14本のMP4動画は `scripts/run_stage1.py` からも生成できます。

```sh
python scripts/run_stage1.py
```

MP4の書き出しにはffmpegが必要です。動画コードはシステムのffmpegを優先し、利用できない場合には `imageio-ffmpeg` の実行ファイルを探します。動画の保存先は `stage1_animation_outputs/animations/` です。作成済みMP4自体はこのリポジトリに含めていません。

## 数値的な注意点

- LFは古典的Lax–Friedrichs、LLFはLocal Lax–Friedrichs/Rusanovです。表示名は **LLF (Rusanov)**、内部識別子は `LLF` です。高精度版の凡例は **LLF (Rusanov) + MUSCL** です。
- LWはRichtmyerの2段階法です。HLLは標準でEinfeldt波速を用い、図では **HLL (HLLE)** と表示します。
- 負の密度・圧力などを検出した計算は、下限値で補正せず最後に受理した状態を保持して停止します。自動再試行や別ソルバーへの切替はありません。
- 失敗した途中時刻の解を、完走した方式の最終時刻の比較図には混ぜません。動画では停止後の曲線を消します。
- 動画は共通の表示時刻に揃えますが、数値積分の時間刻みをフレーム出力のために変更しません。保存状態の補間は描画専用です。
- Shu–Osherの黒線は高解像度の**数値参照解**であり、厳密解ではありません。
- 数値流束、空間再構築、時間積分、波速推定、リミッター、CFL数は区別して報告してください。本教材全体を研究用に検証済みとみなすものではありません。

## 動作確認

Python 3.13.5の検証環境で、各冊を別々の新しいカーネルから実行しました。3冊の合計40コードセル（13＋13＋14）について、以下の両設定でセル実行エラーは0でした。各冊の9グループの基本検証も通過しています。

**標準設定：** 基本問題 `n_basic=200`、格子収束試験 `n_convergence=(100, 200, 400, 800)`。追加実験・動画・Shu–Osherは無効です。第III部では1次・2次精度HLLCの部間比較も実行しました。

**追加機能の小規模確認：** `n_basic=40`、`n_convergence=(20, 40, 80)`、`n_shu_osher=40`、`n_reference=(80, 160)`、`run_movies=run_extras=run_shu_osher=True`、`movie_frames=5`。各冊のSodアニメーション、追加実験、Shu–Osher、結果保存を確認しました。動画の記録処理の有無による最終状態・ステップ数・境界流束の一致も検査します。

両設定で、二重希薄波のLW、1次Roe、MUSCL-Roeは数値的な破綻として記録されています。これはセル実行エラーとは区別しています。Shu–Osherの標準高解像度参照計算（1600・3200セル）は、この検証の対象外です。


配布Notebookには出力画像・動画を埋め込んでいません。実行すると生成されます。参考文献は各Notebook末尾にまとめています。

著作権者による明示的なライセンス指定はまだありません。
