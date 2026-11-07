# 1D Euler equation: comparison of numerical schemes

学習用の1次元圧縮性Euler方程式ソルバーと、日本語の比較演習です。

## 推奨：3部構成で学ぶ

**[3部構成Notebook — euler1d_three_parts.ipynb](notebooks/euler1d_three_parts.ipynb)**

従来の「基本6方式＋MUSCL」の教材を、学習目的に合わせて整理しました。数値計算本体 `src/euler1d_core.py` は変更していません。

| 部 | 比較対象 | この教材での構成 | 主な学習目的 |
|:--|:--|:--|:--|
| 第I部：古典的手法 | LF、LW | 古典的LF、Richtmyerの2段階LW | 数値拡散と数値分散・振動の違い |
| 第II部：現代の実用的手法 | Roe、LLF (Rusanov)、HLL、HLLC | 区分一定再構成＋前進Euler | 1次精度に統一して数値流束の違いを調べる |
| 第III部：実用的な高精度手法 | 第II部の4流束 | 共通のMUSCL (minmod)＋SSP-RK2 | 同じ流束を高次精度化した効果を調べる |

これは**精度の低い順の分類ではありません**。古典的LWは滑らかな解で2次精度です。第II部の「実用的」は数値流束の位置づけを指し、学習用の1次精度構成全体を最終的な研究用設定として推奨する意味ではありません。第III部は第II部と排他的な分類ではなく、同じ流束に高次精度化を組み合わせます。

各部で、滑らかなエントロピー波、静止・移動する接触不連続面、Sod衝撃波管、二重希薄波と格子収束試験を扱います。図には原則としてその部の手法だけを表示します。同じHLLCの1次精度とMUSCL＋SSP-RK2を比べる部間比較もあります。

空間再構築と時間積分法を独立に変える実験は**次の学習計画**として末尾に分けました。この版でPPM、WENO、SSP-RK3、PyClawを追加したわけではありません。

## 取得・実行

この3部構成版は、長い関数定義を `src/` に分けています。**Notebookだけでなく、リポジトリ一式を取得し、`notebooks/` と `src/` の配置を保ってください。**

```sh
git clone https://github.com/babajn/euler1d-scheme-comparison.git
cd euler1d-scheme-comparison
python -m pip install -r requirements.txt
jupyter notebook notebooks/euler1d_three_parts.ipynb
```

取得済みのリポジトリでは、ローカルの変更を保存したうえで更新してください。

```sh
git pull --ff-only
jupyter notebook notebooks/euler1d_three_parts.ipynb
```

`src/euler1d_study.py` が分類別の実験・描画・結果保存を担当し、既存の `src/euler1d_core.py` を呼び出します。設定変更後はカーネルを再起動して上から実行すると、図・表の条件を揃えられます。

### 設定と追加実験

最初の `CFG` セルで設定します。基本問題と格子収束試験は標準で実行し、以下は初期状態では無効です。

- `run_movies=True`：選択した1問題を3部に分けて動画表示します。初期値はSodです。JavaScript表示のため、この機能にffmpegは不要です。
- `run_shu_osher=True`：Shu–Osher問題を3部で比較し、2解像度の数値参照解の差も確認します。
- `run_extras=True`：Lax衝撃波管、LFとLLFのCFL数依存性、Roeのエントロピー補正の有無を調べます。

`export_results=True` のとき、結果は `three_part_outputs/` に保存します。CSVには内部識別子 `solver`、表示名 `method`、学習分類 `part` を別々に記録します。

## 従来版も保存しています

- [従来の全演習＋Stage 1動画](notebooks/euler1d_complete_light.ipynb)：基本6方式の総合比較、MUSCL、誤差解析、動画生成。
- [従来のStage 1動画専用](notebooks/euler1d_stage1_light.ipynb)：6方式を重ねる時間発展比較。

この2冊は計算コアを内蔵した単独実行版です。従来の14本のMP4動画生成は、これらと `scripts/run_stage1.py` に残しています。

```sh
python scripts/run_stage1.py
```

MP4の書き出しにはffmpegが必要です。動画コードはシステムのffmpegを優先し、利用できない場合には `imageio-ffmpeg` の実行ファイルを探します。既存の動画は `stage1_animation_outputs/animations/` に保存されます。作成済みMP4自体はこのリポジトリに含めていません。

## 数値的な注意点

- LFは古典的Lax–Friedrichs、LLFはLocal Lax–Friedrichs/Rusanovです。表示名は **LLF (Rusanov)**、内部の識別子は `LLF` です。
- LWはRichtmyerの2段階法です。HLLは標準でEinfeldt波速を用い、図では **HLL (HLLE)** と表示します。
- 負の密度・圧力などを検出した計算は、下限値で補正せず最後に受理した状態を保持して停止します。自動再試行や別ソルバーへの切替はありません。
- 失敗した途中時刻の解を、完走した方式の最終時刻の比較図には混ぜません。動画では停止後の曲線を消します。
- 動画は共通の表示時刻に揃えますが、数値積分の時間刻みをフレーム出力のために変更しません。保存状態の補間は描画専用です。
- Shu–Osherの黒線は高解像度の**数値参照解**であり、厳密解ではありません。
- 数値流束、空間再構築、時間積分、波速推定、リミッター、CFL数は区別して報告してください。本教材全体を研究用に検証済みとみなすものではありません。

## 3部構成版の動作確認

既存の9グループの基本検証に加え、全27コードセルを次の**小規模設定**で実行しました。

```python
n_basic = 40
n_convergence = (20, 40, 80)
n_shu_osher = 40
n_reference = (80, 160)
run_shu_osher = True
run_extras = True
run_movies = True
movie_frames = 5
```

セルの実行エラーは0でした。98条件のうち、二重希薄波のLW、1次Roe、MUSCL-Roeの3条件は数値的な破綻として記録されました。動画では記録処理の有無による最終状態・ステップ数・境界流束の一致も検査します。これは小規模設定での動作確認であり、標準の高解像度設定一式を再実行した結果ではありません。

配布Notebookには出力画像・動画を埋め込んでいません。実行すると生成されます。参考文献はNotebook末尾にまとめています。

著作権者による明示的なライセンス指定はまだありません。
