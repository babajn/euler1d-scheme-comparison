# Standalone module extracted from notebooks/euler1d_stage1_light.ipynb
# Numerical algorithm is unchanged from the notebook.
from euler1d_core import ALL_SOLVERS, cons_to_prim
from euler1d_animation_data import ANIM_CFG

"""Single-panel comparison animations, MP4 export, and a self-contained HTML gallery."""
from pathlib import Path
import base64
import hashlib
import html
import json
import shutil
import subprocess
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, Video, display

# 岡部・伊藤パレット。元のNotebookと方式ごとの対応を変えない。
ANIM_COLORS = {"LF": "#0072B2", "LW": "#D55E00", "Roe": "#009E73", "LLF": "#CC79A7", "HLL": "#E69F00", "HLLC": "#56B4E9"}
ANIM_LINES = {"LF": "-", "LW": "-", "Roe": "--", "LLF": "-.", "HLL": ":", "HLLC": "--"}
ANIM_MARKERS = {"LF": "s", "LW": "^", "Roe": "o", "LLF": "D", "HLL": "v", "HLLC": "x"}
FIELD_LABELS = {"rho": r"Density $\rho$", "u": r"Velocity $u$", "p": r"Pressure $p$", "e": r"Specific internal energy $e$"}
FIELD_JA = {"rho": "密度", "u": "速度", "p": "圧力", "e": "比内部エネルギー"}
CASE_JA = {"entropy": "滑らかなエントロピー波", "contact_static": "静止する接触不連続面", "contact": "移動する接触不連続面", "sod": "Sod衝撃波管", "double_rarefaction": "二重希薄波", "shu_osher": "Shu–Osher問題", "lax": "Lax衝撃波管（追加）"}
CASE_NOTES = {
    "entropy": "密度の正弦波が1周します。振幅の低下と位相のずれを比較してください。",
    "contact_static": "密度だけが不連続な静止状態です。接触面が数値拡散で広がるかを確認します。",
    "contact": "密度の矩形波が右へ移動します。境界の広がりとLWの数値振動を比較します。",
    "sod": "左向きの希薄波・接触不連続面・右向きの衝撃波の発生と伝播を比べます。",
    "double_rarefaction": "中央で密度・圧力が低下します。負圧などで停止した方式は、最後の受理時刻を超えて描画しません。密度・圧力は対数軸です。",
    "shu_osher": "衝撃波後方の短波長の密度変動を比較します。黒線は高解像度の数値参照解であり、厳密解ではありません。",
    "lax": "Sodとは異なる速度・圧力差を持つ追加の衝撃波管問題です。",
}


def animation_field(U, field, gamma):
    W = cons_to_prim(U, gamma, check=False)
    if field in ("rho", "u", "p"):
        return W[..., ("rho", "u", "p").index(field)]
    if field == "e":
        return W[..., 2] / ((gamma - 1.0) * W[..., 0])
    raise ValueError("field must be rho, u, p, or e.")


def make_comparison_animation(data, field="rho", cfg=ANIM_CFG, solvers=None, xlim=None, ylim=None, yscale=None):
    """各動画は一つの座標軸。全方式は同じ物理時刻・同じ固定軸範囲で表示する。"""
    solvers = ALL_SOLVERS if solvers is None else tuple(solvers)
    if not solvers or any(s not in ALL_SOLVERS for s in solvers):
        raise ValueError("Select at least one known solver.")
    if field not in FIELD_LABELS or cfg["fps"] <= 0:
        raise ValueError("Invalid field or frame rate.")
    for name in ("hold_start", "hold_end"):
        if cfg[name] < 0 or int(cfg[name]) != cfg[name]:
            raise ValueError("Hold counts must be nonnegative integers.")
    times, x = data["times"], data["x"]
    values = {s: animation_field(data["states"][s], field, data["gamma"]) for s in solvers}
    reference = animation_field(data["reference"], field, data["gamma"])
    bounds = (float(data["edges"][0]), float(data["edges"][-1])) if xlim is None else tuple(xlim)
    visible_x = (x >= bounds[0]) & (x <= bounds[1])
    if not visible_x.any():
        raise ValueError("xlim does not contain any cells.")
    scale = ("log" if data["key"] == "double_rarefaction" and field in ("rho", "p") else "linear") if yscale is None else yscale
    extrema = [reference[:, visible_x].ravel()] + [values[s][data["valid"][s]][:, visible_x].ravel() for s in solvers]
    low = min(float(np.nanmin(a)) for a in extrema)
    high = max(float(np.nanmax(a)) for a in extrema)
    if ylim is None:
        if scale == "log":
            if low <= 0.0:
                raise ValueError("Logarithmic movies require positive accepted values.")
            yrange = (low / 1.45, high * 1.45)
        else:
            span = high - low
            margin = 0.09 * span if span > 1e-10 * max(1.0, abs(low), abs(high)) else 0.05 * max(1.0, abs(low), abs(high))
            yrange = (low - margin, high + margin)
    else:
        yrange = tuple(ylim)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "figure.dpi": cfg["dpi"]}):
        fig = plt.figure(figsize=cfg["figsize"], dpi=cfg["dpi"])
        ax = fig.add_axes((0.085, 0.22, 0.66, 0.60))
        fig.text(0.085, 0.945, data["title"], fontsize=17, weight="bold")
        clock = fig.text(0.085, 0.895, "", fontsize=12)
        ref_line, = ax.plot(x, reference[0], color="#000000", lw=1.65, label=data["reference_label"], zorder=10)
        lines = {}
        for i, s in enumerate(solvers):
            stride = max(1, data["n"] // 19)
            lines[s], = ax.plot(x, values[s][0], color=ANIM_COLORS[s], ls=ANIM_LINES[s], lw=1.65, marker=ANIM_MARKERS[s], ms=4.0, markevery=(i % stride, stride), mfc="none", label=("LLF (Rusanov)" if s == "LLF" else data["runs"][s]["label"]))
        ax.set(xlabel="$x$", ylabel=FIELD_LABELS[field], xlim=bounds, ylim=yrange, yscale=scale)
        ax.grid(alpha=0.22, which="both")
        ax.tick_params(direction="out")
        ax.legend(loc="upper left", bbox_to_anchor=(1.025, 1.025), borderaxespad=0.0, frameon=False, fontsize=10, handlelength=2.5, labelspacing=0.65)
        fig.text(0.775, 0.45, "Solver status", fontsize=11, weight="bold")
        status_lines = {s: fig.text(0.775, 0.413 - 0.034 * i, "", fontsize=9.5, color=ANIM_COLORS[s]) for i, s in enumerate(solvers)}
        failure_note = fig.text(0.085, 0.115 if data["key"] == "double_rarefaction" else 0.135, "", fontsize=10)
        fig.text(0.085, 0.083, "LF: classical; Roe/LLF/HLL/HLLC: 1st order; LW: Richtmyer two-stage, 2nd order.", fontsize=9.5)
        fig.text(0.085, 0.050, "Common display times: conserved states are interpolated only for plotting; solver time steps are unchanged.", fontsize=9)
        fig.text(0.085, 0.020, "Fixed axes throughout. Markers denote cells; styles are fixed across problems. Failed runs are never extrapolated.", fontsize=9)

        def update(j):
            t = float(times[j])
            clock.set_text(f"Stage 1  |  N = {data['n']}  |  CFL = {data['cfl']:g}  |  t = {t:.6f} / {times[-1]:g}  |  frame {j + 1}/{len(times)}")
            ref_line.set_ydata(reference[j])
            notes = []
            for s in solvers:
                info = data["runs"][s]
                display_name = "LLF (Rusanov)" if s == "LLF" else s
                if data["valid"][s][j]:
                    lines[s].set_ydata(values[s][j])
                    status_lines[s].set_text(f"{display_name}: {'complete' if j == len(times) - 1 else 'active'}")
                else:
                    lines[s].set_ydata(np.full_like(x, np.nan))
                    status_lines[s].set_text(f"{display_name}: STOPPED at {info['t_reached']:.5g}")
                    trial = info.get("fail_time", float("nan"))
                    pmin = info.get("fail_min_p", float("nan"))
                    reason = f"{display_name}: last accepted t={info['t_reached']:.5g}; rejected trial t={trial:.5g}"
                    if np.isfinite(pmin):
                        reason += f", min(p)={pmin:.3g}"
                    notes.append(reason)
            failure_note.set_text("\n".join(notes) if notes else "All displayed states are admissible. No density/pressure floor or automatic solver fallback.")
            return [ref_line, clock, failure_note] + list(lines.values()) + list(status_lines.values())

        frames = [0] * int(cfg["hold_start"]) + list(range(len(times))) + [len(times) - 1] * int(cfg["hold_end"])
        movie = animation.FuncAnimation(fig, update, frames=frames, interval=1000.0 / cfg["fps"], repeat=True, blit=False)
        update(0)
    return fig, movie, update


def configure_ffmpeg():
    """Use system ffmpeg, or imageio-ffmpeg when installed. No shell-specific commands."""
    if animation.writers.is_available("ffmpeg"):
        return True
    path = shutil.which("ffmpeg")
    if path:
        matplotlib.rcParams["animation.ffmpeg_path"] = path
    else:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return False
    return animation.writers.is_available("ffmpeg")


def export_animation(data, field="rho", cfg=ANIM_CFG, solvers=None, xlim=None, ylim=None, yscale=None, name_suffix=""):
    """Create/reuse an MP4. Return None when MP4 export is disabled or ffmpeg is absent."""
    if not cfg["save_mp4"] or not configure_ffmpeg():
        return None
    folder = Path(cfg["output_dir"]) / "animations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"stage1_{data['key']}_{field}{name_suffix}.mp4"
    settings = {k: cfg[k] for k in ("fps", "dpi", "figsize", "hold_start", "hold_end")}
    settings.update({"data_id": data["data_id"], "field": field, "solvers": solvers, "xlim": xlim, "ylim": ylim, "yscale": yscale, "render_version": 4})
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    stamp = path.with_suffix(".json")
    if cfg["reuse_movies"] and path.exists() and stamp.exists():
        if json.loads(stamp.read_text())["render_id"] == digest:
            return path
    fig, movie, update = make_comparison_animation(data, field, cfg, solvers=solvers, xlim=xlim, ylim=ylim, yscale=yscale)
    # Draw ticks, grid, labels and legend once; redraw only the curves and changing text.
    # Streaming raw RGB(A) frames directly avoids a complete savefig() for each movie frame.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    moving = update(0)
    for artist in moving:
        artist.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    width, height = fig.canvas.get_width_height()
    temp_path = path.with_name(path.stem + ".tmp.mp4")
    command = [str(matplotlib.rcParams["animation.ffmpeg_path"]), "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-pix_fmt", "rgba", "-framerate", str(cfg["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-metadata", "title=" + data["title"], str(temp_path)]
    sequence = [0] * int(cfg["hold_start"]) + list(range(len(data["times"]))) + [len(data["times"]) - 1] * int(cfg["hold_end"])
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
        try:
            for frame in sequence:
                fig.canvas.restore_region(background)
                for artist in sorted(update(frame), key=lambda a: a.get_zorder()):
                    fig.draw_artist(artist)
                process.stdin.write(fig.canvas.buffer_rgba())
            process.stdin.close()
            code = process.wait()
            if code:
                error_log.seek(0)
                raise RuntimeError("ffmpeg failed: " + error_log.read().decode(errors="replace"))
            temp_path.replace(path)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if temp_path.exists():
                temp_path.unlink()
            plt.close(fig)
    stamp.write_text(json.dumps({"render_id": digest, "settings": settings, "frames": len(data["times"]) + cfg["hold_start"] + cfg["hold_end"], "seconds": (len(data["times"]) + cfg["hold_start"] + cfg["hold_end"]) / cfg["fps"]}, indent=2))
    print(f"Saved: {path.name}")
    return path


def show_animation(data, field="rho", cfg=ANIM_CFG, solvers=None, xlim=None, ylim=None, yscale=None, name_suffix=""):
    """MP4 playback by default; JavaScript controls are available without ffmpeg."""
    path = export_animation(data, field, cfg, solvers=solvers, xlim=xlim, ylim=ylim, yscale=yscale, name_suffix=name_suffix)
    if path is not None:
        display(Video(filename=str(path), embed=True, width=1050, html_attributes="controls loop playsinline preload='metadata'"))
        return path
    fig, movie, _ = make_comparison_animation(data, field, cfg, solvers=solvers, xlim=xlim, ylim=ylim, yscale=yscale)
    # A JSHTML frame is larger than a compressed video; limit the resolution only in this fallback.
    fig.set_dpi(min(80, cfg["dpi"]))
    try:
        with matplotlib.rc_context({"animation.embed_limit": 160}):
            markup = movie.to_jshtml(fps=cfg["fps"], default_mode="loop")
        display(HTML(markup))
    finally:
        plt.close(fig)
    return None


def export_gallery(datasets, cfg=ANIM_CFG, embed=True):
    """One video at a time; switch problem/quantity without Python or an Internet connection."""
    folder = Path(cfg["output_dir"])
    items = []
    for key, data in datasets.items():
        fields = []
        for field in cfg["fields_by_problem"][key]:
            path = folder / "animations" / f"stage1_{key}_{field}.mp4"
            if not path.exists():
                continue
            source = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode() if embed else "animations/" + path.name
            fields.append({"key": field, "name": FIELD_JA[field], "src": source, "file": path.name})
        if fields:
            stopped = []
            for s in ALL_SOLVERS:
                r = data["runs"][s]
                if r["status"] != "ok":
                    stopped.append(f"{s}：最後の受理時刻 t={r['t_reached']:.8g}、棄却した試行の終端 t={r['fail_time']:.8g}、min(p)={r['fail_min_p']:.6g}。")
            items.append({"key": key, "title": CASE_JA[key], "note": CASE_NOTES[key], "n": data["n"], "end": float(data["times"][-1]), "fields": fields, "stopped": stopped})
    if not items:
        print("No MP4 files exist yet; the gallery was not written.")
        return None
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Stage 1 | 1D Euler scheme animations</title>
<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:1180px;margin:28px auto;padding:0 20px;color:#182432;background:#f5f7fa}h1{font-size:27px;margin-bottom:8px}p{line-height:1.7}.box{background:#fff;border:1px solid #d9e0e8;border-radius:12px;padding:20px;margin-top:18px}label{display:inline-flex;gap:10px;align-items:center;margin:5px 18px 5px 0}select,button{padding:8px;font-size:15px;border:1px solid #acb9c7;border-radius:5px;background:white}video{width:100%;height:auto;background:white;margin-top:10px}small{line-height:1.7;color:#465666}#failure{background:#fff4ec;padding:12px;border-left:4px solid #b75b19;white-space:pre-line}a{color:#006ba2}.badge{font-size:13px;letter-spacing:1px;color:#38637f}.key{font-family:monospace}</style>
<div class="badge">COMPUTATIONAL FLUID DYNAMICS · STAGE 1</div><h1>1次元Euler方程式：スキームの時間発展比較</h1>
<p>LF・LW・Roe・LLF (Rusanov)・HLL（HLLE）・HLLCを、同じ物理時刻と同じ軸範囲で比較します。保存済み動画なので再生にPythonやインターネット接続は不要です。</p>
<div class="box"><label>問題<select id="case"></select></label><label>表示量<select id="field"></select></label><label>再生速度<select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label><input id="loop" type="checkbox" checked>繰り返す</label>
<h2 id="title"></h2><p id="note"></p><video id="video" controls loop playsinline preload="metadata"></video><p><span id="meta"></span>　<a id="download" download>このMP4を保存</a></p><p id="failure" hidden></p></div>
<div class="box"><h2>比較時の注意</h2><p>黒線は厳密リーマン解などの保存変数を同じ格子でセル平均した参照値です。Shu–OsherのみHLLC＋MUSCL/SSP-RK2の高解像度数値参照解です。LF・Roe・LLF・HLL・HLLCは1次精度の構成、LWはRichtmyerの2段階・2次精度法です。</p><p>動画の出力時刻は共通ですが、各方式のCFL時間刻みは変更していません。フレーム時刻を挟む二つの受理済み保存状態を表示用に線形補間しています。補間状態で計算を再開することはありません。補間は衝撃波面をわずかに平滑化し得るため、定量評価には元の数値解を用いてください。停止後の状態を外挿・再利用して現在時刻の曲線として描くことはありません。</p><small>動画の最初と最後には静止時間があります。物理時刻は図の上部に表示されます。各問題の物理的な終了時刻は異なります。動画の長さを揃えているので、異なる問題間の再生速度は同一の物理時間尺度ではありません。</small></div>
<script>const items=__DATA__;const c=document.getElementById('case'),f=document.getElementById('field'),v=document.getElementById('video'),s=document.getElementById('speed');for(const a of items)c.add(new Option(a.title,a.key));
function selectField(){const a=items.find(x=>x.key===c.value),b=a.fields.find(x=>x.key===f.value);v.pause();v.src=b.src;v.load();v.playbackRate=Number(s.value);document.getElementById('meta').textContent='N = '+a.n+' | t_end = '+a.end+' | '+b.name;const d=document.getElementById('download');d.href=b.src;d.download=b.file;}
function selectCase(){const a=items.find(x=>x.key===c.value);f.replaceChildren();for(const b of a.fields)f.add(new Option(b.name,b.key));document.getElementById('title').textContent=a.title;document.getElementById('note').textContent=a.note;const z=document.getElementById('failure');z.hidden=!a.stopped.length;z.textContent=a.stopped.join(String.fromCharCode(10));selectField();}
c.onchange=selectCase;f.onchange=selectField;s.onchange=()=>{v.playbackRate=Number(s.value)};document.getElementById('loop').onchange=e=>{v.loop=e.target.checked};selectCase();</script></html>'''
    page = page.replace("__DATA__", payload)
    path = folder / ("stage1_gallery.html" if embed else "index.html")
    path.write_text(page, encoding="utf-8")
    print(f"Saved gallery: {path.name}")
    return path