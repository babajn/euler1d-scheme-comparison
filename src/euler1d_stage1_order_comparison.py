"""Animation helpers embedded in the standalone Euler comparison notebook.

Execute after the notebook's numerical core and Stage 1 animation definitions.
These helpers call the existing simulate() function without altering its updates.
"""
ORDER_EXTENSION_VERSION = "1.0"


def solver_display_name(solver, wave_speeds="einfeldt"):
    if solver == "LLF":
        return "LLF (Rusanov)"
    return "HLL (HLLE)" if solver == "HLL" and wave_speeds == "einfeldt" else solver


def validate_order_config(cfg, order_cfg):
    keys = tuple(order_cfg["problems"])
    fluxes = tuple(order_cfg["fluxes"])
    pairs = tuple(order_cfg["pair_solvers"])
    if not keys or len(set(keys)) != len(keys) or any(k not in PROBLEMS for k in keys):
        raise ValueError("problems must contain distinct keys from PROBLEMS.")
    if not fluxes or len(set(fluxes)) != len(fluxes) or any(s not in RIEMANN_SOLVERS for s in fluxes):
        raise ValueError("MUSCL comparisons support only Roe, LLF, HLL and HLLC; LF/LW remain in group A.")
    if len(set(pairs)) != len(pairs) or any(s not in fluxes for s in pairs):
        raise ValueError("pair_solvers must be distinct members of fluxes.")
    if order_cfg["limiter"] not in ("minmod", "mc", "unlimited"):
        raise ValueError("limiter must be minmod, mc or unlimited.")
    for key in keys:
        fields = order_cfg["fields_by_problem"].get(key, ())
        if not fields or len(set(fields)) != len(fields) or any(f not in FIELD_LABELS for f in fields):
            raise ValueError(f"Set distinct rho/u/p/e fields for {key}.")
    if int(cfg["n_frames"]) != cfg["n_frames"] or cfg["n_frames"] < 2:
        raise ValueError("n_frames must be an integer >= 2.")


def order_cache_parameters(key, cfg, order_cfg):
    return {"format_version": 1, "base": frame_cache_parameters(key, cfg), "fluxes": list(order_cfg["fluxes"]), "reconstruction": "MUSCL (primitive variables)", "limiter": order_cfg["limiter"], "time_integrator": "SSP-RK2", "core_id": STAGE1_CORE_ID, "extension": ORDER_EXTENSION_VERSION, "cache_revision": order_cfg["cache_revision"]}


def get_muscl_animation_data(key, cfg, order_cfg, base=None):
    """Record order=2 runs; reuse the same grid, times and reference as group A."""
    validate_order_config(cfg, order_cfg)
    if base is None:
        base = get_animation_data(key, cfg)
    if base["parameters"] != frame_cache_parameters(key, cfg):
        raise ValueError("Group A data are stale. Rerun the data-generation cell after changing settings.")
    params = order_cache_parameters(key, cfg, order_cfg)
    cache = Path(cfg["output_dir"]) / "data" / f"muscl_{key}_{order_cfg['limiter']}.npz"
    common = {k: base[k] for k in ("key", "title", "gamma", "cfl", "n", "times", "x", "edges", "reference", "reference_label", "reference_check")}
    if cfg["reuse_data"] and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            if meta["parameters"] == params:
                data = dict(common, **meta)
                data["states"] = {s: saved["U_" + s].copy() for s in order_cfg["fluxes"]}
                data["valid"] = {s: saved["valid_" + s].copy() for s in order_cfg["fluxes"]}
                data["brackets"] = {s: tuple(saved[p + s].copy() for p in ("left_", "right_", "weight_")) for s in order_cfg["fluxes"]}
                print(f"{key}: reused MUSCL/{order_cfg['limiter']} frames")
                return data
    data = dict(common, parameters=params, data_id=_signature(params), runs={}, states={}, valid={}, brackets={})
    for solver in order_cfg["fluxes"]:
        run, recorder, summary = _run_and_record(PROBLEMS[key], solver, base["n"], base["times"], cfg, order=2, limiter=order_cfg["limiter"])
        summary.update(reconstruction="MUSCL", limiter=order_cfg["limiter"], time_integrator="SSP-RK2", formal_space_order=2, formal_time_order=2)
        if cfg["verify_observer"]:
            plain = simulate(PROBLEMS[key], solver, n=base["n"], gamma=cfg["gamma"], cfl=cfg["cfl"], order=2, limiter=order_cfg["limiter"], wave_speeds=cfg["wave_speeds"], roe_entropy=cfg["roe_entropy"])
            same = np.array_equal(run.U, plain.U) and np.array_equal(run.boundary_exchange, plain.boundary_exchange) and (run.t, run.steps, run.status) == (plain.t, plain.steps, plain.status)
            if not same:
                raise AssertionError("The MUSCL observer changed numerical evolution.")
            summary["observer_identical"] = True
        data["runs"][solver], data["states"][solver], data["valid"][solver] = summary, recorder.states, recorder.valid
        data["brackets"][solver] = (recorder.left_time, recorder.right_time, recorder.weight)
    arrays = {}
    for s in order_cfg["fluxes"]:
        arrays["U_" + s], arrays["valid_" + s] = data["states"][s], data["valid"][s]
        arrays["left_" + s], arrays["right_" + s], arrays["weight_" + s] = data["brackets"][s]
    meta = {k: data[k] for k in ("parameters", "data_id", "runs")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, metadata=np.array(json.dumps(meta)), **arrays)
    print(f"{key}: MUSCL/{order_cfg['limiter']} " + ", ".join(s + "=" + data["runs"][s]["status"] for s in order_cfg["fluxes"]))
    return data


def order_comparison_curves(base, high, mode, solver=None):
    """Return curve specifications, not new numerical solutions."""
    if mode == "muscl":
        return [(s, high, s, solver_display_name(s, high["parameters"]["base"]["wave_speeds"]), ANIM_COLORS[s], ANIM_LINES[s], ANIM_MARKERS[s]) for s in high["runs"]]
    if mode == "pair" and solver in high["runs"]:
        return [("first", base, solver, "1st order\nConstant + Forward Euler", "#0072B2", "--", "o"), ("second", high, solver, "2nd order\nMUSCL + SSP-RK2", "#D55E00", "-", "s")]
    raise ValueError("Use mode='muscl', or mode='pair' with a computed Riemann solver.")


def comparison_axis_limits(base, high, field):
    """All B/C plots for this problem and field use the same fixed y range."""
    values = [animation_field(base["reference"], field, base["gamma"]).ravel()]
    for d in (base, high):
        for s in high["runs"]:
            good = d["valid"][s]
            values.append(animation_field(d["states"][s][good], field, d["gamma"]).ravel())
    lo, hi = min(float(np.nanmin(v)) for v in values), max(float(np.nanmax(v)) for v in values)
    log = base["key"] == "double_rarefaction" and field in ("rho", "p")
    if log:
        if lo <= 0:
            raise ValueError("Accepted states must be positive for logarithmic axes.")
        return (lo / 1.45, hi * 1.45), "log"
    span = hi - lo
    margin = 0.09 * span if span > 1e-10 * max(1.0, abs(lo), abs(hi)) else 0.05 * max(1.0, abs(lo), abs(hi))
    return (lo - margin, hi + margin), "linear"


def make_order_comparison_animation(base, high, field, cfg, mode="muscl", solver=None):
    if field not in FIELD_LABELS or not np.array_equal(base["times"], high["times"]) or not np.array_equal(base["edges"], high["edges"]):
        raise ValueError("The field must be supported and both configurations must share times and grid.")
    curves = order_comparison_curves(base, high, mode, solver)
    times, x, limiter = base["times"], base["x"], high["parameters"]["limiter"]
    values = {cid: animation_field(d["states"][s], field, d["gamma"]) for cid, d, s, *_ in curves}
    reference = animation_field(base["reference"], field, base["gamma"])
    ylim, scale = comparison_axis_limits(base, high, field)
    name = "Four Riemann solvers" if mode == "muscl" else solver_display_name(solver, cfg["wave_speeds"])
    subtitle = f"B: {name} | MUSCL ({limiter}) + SSP-RK2" if mode == "muscl" else f"C: {name} | 1st vs 2nd order | limiter: {limiter}"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": cfg["dpi"]}):
        fig = plt.figure(figsize=cfg["figsize"], dpi=cfg["dpi"])
        ax = fig.add_axes((0.085, 0.24, 0.64, 0.55))
        fig.text(0.085, 0.95, base["title"], fontsize=16, weight="bold")
        fig.text(0.085, 0.902, subtitle, fontsize=12)
        clock = fig.text(0.085, 0.853, "", fontsize=11)
        ref_line, = ax.plot(x, reference[0], color="#000000", lw=1.6, label=base["reference_label"], zorder=10)
        lines = {}
        for j, (cid, d, s, label, color, ls, marker) in enumerate(curves):
            stride = max(1, len(x) // 19)
            lines[cid], = ax.plot(x, values[cid][0], color=color, ls=ls, marker=marker, markevery=(j % stride, stride), mfc="none", ms=4, lw=1.6, label=label)
        ax.set(xlim=(base["edges"][0], base["edges"][-1]), ylim=ylim, yscale=scale, xlabel="$x$", ylabel=FIELD_LABELS[field])
        ax.grid(alpha=0.22, which="both")
        ax.legend(loc="upper left", bbox_to_anchor=(1.025, 1.025), borderaxespad=0, frameon=False, fontsize=9.5, handlelength=2.5)
        fig.text(0.765, 0.44, "Solver status", fontsize=11, weight="bold")
        statuses = {cid: fig.text(0.765, 0.40 - 0.036 * j, "", fontsize=9.5, color=color) for j, (cid, d, s, label, color, ls, marker) in enumerate(curves)}
        failures = fig.text(0.085, 0.15, "", fontsize=9)
        concept = "Same reconstruction and time method: only the Riemann solver changes." if mode == "muscl" else "Space AND time change together: this is a full-configuration comparison."
        fig.text(0.085, 0.087, concept, fontsize=9.4)
        fig.text(0.085, 0.053, "Nominal orders apply to smooth solutions; limiters may reduce local accuracy.", fontsize=9)
        fig.text(0.085, 0.021, "Display-only interpolation of accepted states. Fixed axes. Failed runs are not extrapolated.", fontsize=9)

        def update(j):
            clock.set_text(f"N = {base['n']} | CFL = {base['cfl']:g} | t = {times[j]:.6f} / {times[-1]:g} | frame {j + 1}/{len(times)}")
            ref_line.set_ydata(reference[j])
            stopped = []
            for cid, d, s, label, color, ls, marker in curves:
                info = d["runs"][s]
                brief = ("1st" if cid == "first" else "2nd") if mode == "pair" else solver_display_name(s, cfg["wave_speeds"])
                if d["valid"][s][j]:
                    lines[cid].set_ydata(values[cid][j])
                    statuses[cid].set_text(brief + (": complete" if j == len(times) - 1 else ": active"))
                else:
                    lines[cid].set_ydata(np.full_like(x, np.nan))
                    statuses[cid].set_text(brief + ": STOPPED")
                    stopped.append(f"{brief}: last accepted t={info['t_reached']:.5g}; rejected trial t={info['fail_time']:.5g}")
            failures.set_text("\n".join(stopped) if stopped else "No density/pressure floor, step retry or automatic solver fallback.")
            return [ref_line, clock, failures] + list(lines.values()) + list(statuses.values())

        frames = [0] * int(cfg["hold_start"]) + list(range(len(times))) + [len(times) - 1] * int(cfg["hold_end"])
        movie = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / cfg["fps"], repeat=True, blit=False)
        update(0)
    return fig, movie, update


def order_movie_path(base, high, field, cfg, mode, solver=None):
    tag = "four_fluxes" if mode == "muscl" else "pair_" + str(solver)
    return Path(cfg["output_dir"]) / "animations" / f"orders_{base['key']}_{field}_{high['parameters']['limiter']}_{tag}.mp4"


def export_order_animation(base, high, field, cfg, mode="muscl", solver=None):
    """Save/reuse MP4 without changing numerical integration or frame sampling."""
    if not cfg["save_mp4"] or not configure_ffmpeg():
        return None
    path = order_movie_path(base, high, field, cfg, mode, solver)
    settings = {k: cfg[k] for k in ("fps", "dpi", "figsize", "hold_start", "hold_end")}
    settings.update(base_id=base["data_id"], high_id=high["data_id"], field=field, mode=mode, solver=solver, render_version=1)
    digest, stamp = _signature(settings), path.with_suffix(".json")
    if cfg["reuse_movies"] and path.exists() and stamp.exists() and json.loads(stamp.read_text())["render_id"] == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, movie, update = make_order_comparison_animation(base, high, field, cfg, mode, solver)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    for artist in update(0):
        artist.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    width, height = fig.canvas.get_width_height()
    temp = path.with_name(path.stem + ".tmp.mp4")
    command = [str(matplotlib.rcParams["animation.ffmpeg_path"]), "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-pix_fmt", "rgba", "-framerate", str(cfg["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(temp)]
    sequence = [0] * int(cfg["hold_start"]) + list(range(len(base["times"]))) + [len(base["times"]) - 1] * int(cfg["hold_end"])
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
        try:
            for frame in sequence:
                fig.canvas.restore_region(background)
                for artist in sorted(update(frame), key=lambda a: a.get_zorder()):
                    fig.draw_artist(artist)
                process.stdin.write(fig.canvas.buffer_rgba())
            process.stdin.close()
            if process.wait():
                error_log.seek(0)
                raise RuntimeError("ffmpeg failed: " + error_log.read().decode(errors="replace"))
            temp.replace(path)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if temp.exists():
                temp.unlink()
            plt.close(fig)
    stamp.write_text(json.dumps({"render_id": digest, "settings": settings, "frames": len(sequence)}, indent=2))
    print("Saved:", path.name)
    return path


def show_order_animation(base, high, field, cfg, order_cfg, mode="muscl", solver=None):
    path = export_order_animation(base, high, field, cfg, mode, solver)
    if path is not None:
        display(Video(filename=str(path), embed=True, width=1050, html_attributes="controls loop playsinline preload='metadata'"))
    else:
        fig, movie, update = make_order_comparison_animation(base, high, field, cfg, mode, solver)
        fig.set_dpi(min(80, cfg["dpi"]))
        try:
            with matplotlib.rc_context({"animation.embed_limit": 160}):
                display(HTML(movie.to_jshtml(fps=cfg["fps"], default_mode="loop")))
        finally:
            plt.close(fig)
    if order_cfg["show_final_snapshot"]:
        fig, movie, update = make_order_comparison_animation(base, high, field, cfg, mode, solver)
        fig.canvas.draw()
        movie.event_source.stop()
        update(len(base["times"]) - 1)
        # Display explicitly so the end state is also visible in static notebook previews.
        display(fig)
        movie._draw_was_started = True
        plt.close(fig)
    return path


def order_status_table(base_data, high_data):
    """Actual endpoint diagnostics, not errors measured from interpolated frames."""
    rows = []
    for key, high in high_data.items():
        for s in high["runs"]:
            for order, d in ((1, base_data[key]), (2, high)):
                r = d["runs"][s]
                rows.append([key, solver_display_name(s, high["parameters"]["base"]["wave_speeds"]), str(order), "constant + FE" if order == 1 else "MUSCL/" + high["parameters"]["limiter"] + " + SSP-RK2", r["status"], f"{r['t_reached']:.6g}", str(r["steps"]), f"{r['balance_max']:.3e}", str(r.get("observer_identical", False))])
    headers = ("Problem", "Flux", "Nominal order", "Configuration", "Status", "Time reached", "Steps", "Conservation residual", "Observer identical")
    head = "<tr>" + "".join("<th>" + h + "</th>" for h in headers) + "</tr>"
    body = "".join("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in row) + "</tr>" for row in rows)
    display(HTML("<table>" + head + body + "</table>"))
    return rows


def export_combined_order_gallery(base_data, high_data, cfg, order_cfg):
    """Self-contained MP4 gallery; stale videos from another setting are excluded."""
    items = []
    def add(path, title, note):
        if path is not None and path.exists():
            items.append({"title": title, "note": note, "src": "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode(), "file": path.name})
    if cfg["save_mp4"] and configure_ffmpeg():
        for key in cfg["problems"]:
            if key not in base_data:
                continue
            for field in cfg["fields_by_problem"][key]:
                path = export_animation(base_data[key], field, cfg)
                add(path, "A | " + CASE_JA.get(key, PROBLEMS[key].title) + " | " + FIELD_JA[field], "LFと4種類のリーマンソルバーは1次精度、LWはRichtmyerの2次精度法です。")
        for key, high in high_data.items():
            for field in order_cfg["fields_by_problem"][key]:
                if order_cfg["run_four_flux_comparison"]:
                    add(export_order_animation(base_data[key], high, field, cfg, "muscl"), "B | " + CASE_JA.get(key, PROBLEMS[key].title) + " | " + FIELD_JA[field], "MUSCL、リミッター、SSP-RK2を共通にして、数値流束だけを変えます。")
                if order_cfg["run_order_pairs"]:
                    for solver in order_cfg["pair_solvers"]:
                        add(export_order_animation(base_data[key], high, field, cfg, "pair", solver), "C | " + solver_display_name(solver, cfg["wave_speeds"]) + " | " + CASE_JA.get(key, PROBLEMS[key].title) + " | " + FIELD_JA[field], "区分一定＋前進Eulerと、MUSCL＋SSP-RK2を比較します。空間と時間を同時に変更した総合比較です。")
    if not items:
        print("No MP4 gallery: Notebook animations are available without ffmpeg.")
        return None
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Euler scheme and order comparison</title><style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:30px auto;padding:0 18px;line-height:1.65}video{width:100%}select{max-width:100%;padding:8px;font-size:15px}p{max-width:1000px}</style><h1>Euler方程式：数値流束と精度の動画比較</h1><p>A：6方式の基本比較　B：MUSCLの流束比較　C：同じ流束の1次・2次精度比較</p><select id="select"></select><p id="note"></p><video id="video" controls loop playsinline preload="metadata"></video><p>同じ物理時刻で比較しています。表示用の補間は計算の時間刻みを変更しません。停止した計算は、その後の時刻に外挿して表示しません。Shu–Osherの黒線は数値参照解であり、厳密解ではありません。</p><script>const data=__DATA__;const s=document.getElementById('select'),v=document.getElementById('video');data.forEach((x,i)=>s.add(new Option(x.title,i)));function change(){const x=data[Number(s.value)];v.pause();v.src=x.src;v.load();document.getElementById('note').textContent=x.note;}s.onchange=change;change();</script></html>'''
    path = Path(cfg["output_dir"]) / "stage1_orders_gallery.html"
    path.write_text(page.replace("__DATA__", payload), encoding="utf-8")
    print("Saved combined gallery:", path.name, "|", len(items), "videos")
    return path
