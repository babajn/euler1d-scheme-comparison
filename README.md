# 1D Euler equation: comparison of numerical schemes

学習用の1次元圧縮性Euler方程式ソルバーと、6種類の流体スキームの比較演習です。

## Notebook

- [全演習＋Stage 1 動画](notebooks/euler1d_complete_light.ipynb): LF、LW、Roe、LLF (Rusanov)、HLL/HLLE、HLLC、MUSCL＋SSP-RK2、厳密解・誤差・収束試験・動画。
- [Stage 1 動画専用](notebooks/euler1d_stage1_light.ipynb): 6方式の時間発展比較に特化した独立実行版。

どちらも計算コアを内蔵しています。GitHubではファイルを開き「Download raw file」で `.ipynb` として保存するか、`git clone` してください。

## 比較するテスト問題

1. 滑らかなエントロピー波（移流・位相誤差・収束次数）
2. 静止する接触不連続面
3. 移動する接触不連続面
4. Sod衝撃波管
5. 二重希薄波（正値性・破綻）
6. Shu–Osher問題（衝撃波と微細な密度構造）
7. Lax衝撃波管（追加）

## 実行方法

```sh
git clone https://github.com/babajn/euler1d-scheme-comparison.git
cd euler1d-scheme-comparison
python -m pip install -r requirements.txt
jupyter notebook notebooks/euler1d_stage1_light.ipynb
```

全演習の場合は `notebooks/euler1d_complete_light.ipynb` を開いてください。

**動画書き出しには、Pythonパッケージとは別に `ffmpeg` が必要です。**
Ubuntuでは `sudo apt install ffmpeg`、macOS（Homebrew）では `brew install ffmpeg`。
Notebook実行時に生成される `stage1_animation_outputs/animations/*.mp4` が比較動画（14本）です。動画の物理時刻は全方式で揃えていますが、時間積分自体の `dt` は各方式固有のCFL条件から決まります。

## 数値的な注意点

- LFは古典的Lax–Friedrichs、LLFはLocal Lax–Friedrichs/Rusanovであり、異なる方法として実装しています。
- LWはRichtmyerの2段階Lax–Wendroff法です。
- Stage 1のRoe・LLF・HLL・HLLCは区分一定再構成（1次精度）、LWは2次精度です。Stage 2ではMUSCL＋SSP-RK2を共通化します。
- HLLでは標準設定でEinfeldt波速を使用したHLLE型の流束を使います。
- 二重希薄波で負の密度・圧力が発生した場合は、下限で補正せず、その時刻と理由を記録して停止します。動画では停止後の曲線を消します。
- Shu–Osherの黒線は高解像度の**数値参照解**であり、厳密解ではありません。

## ファイルについて

このリポジトリのNotebookは、以前ChatGPT内で作成した計算教材を、配布・取得しやすいよう**画像・動画の埋め込み結果だけ除いた**軽量版です。計算コード・日本語解説・動画生成セルを収録しています。

著作権者による明示的なライセンス指定はまだありません。
