"""Generate the Stage 1 numerical-scheme comparison movies.

Usage:
    python scripts/run_stage1.py
    python scripts/run_stage1.py --quick
    python scripts/run_stage1.py --problem sod
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euler1d_animation_data import ANIM_CFG, get_animation_data
from euler1d_animation_plot import export_animation, export_gallery


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 1 Euler-scheme animations")
    parser.add_argument("--problem", choices=ANIM_CFG["problems"], help="Only simulate one problem")
    parser.add_argument("--quick", action="store_true", help="Use smaller grids and fewer frames")
    parser.add_argument("--output", type=Path, default=ROOT / "stage1_animation_outputs")
    args = parser.parse_args()
    cfg = dict(ANIM_CFG, output_dir=str(args.output.resolve()))
    if args.quick:
        cfg.update(n_basic=100, n_shu_osher=200, n_reference=(400, 800), n_frames=41, hold_start=3, hold_end=5, verify_observer=False)
    problems = (args.problem,) if args.problem else cfg["problems"]
    datasets = {}
    for key in problems:
        data = get_animation_data(key, cfg)
        datasets[key] = data
        for field in cfg["fields_by_problem"][key]:
            path = export_animation(data, field, cfg)
            print(f"{key} {field}: {path if path is not None else 'MP4 disabled; ffmpeg unavailable'}")
    if not args.problem:
        export_gallery(datasets, cfg, embed=False)
        export_gallery(datasets, cfg, embed=True)
    print("Results:", cfg["output_dir"])


if __name__ == "__main__":
    main()
