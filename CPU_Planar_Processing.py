"""
Batch planar CPU-PIV pipeline -- im7 input via lvpyio, processed with
plain openpiv-python (https://github.com/OpenPIV/openpiv-python)
================================================================================
No-GPU counterpart to Planar.py: reads LaVision .im7 images directly and
runs them through openpiv-python's basic single-pass
`pyprocess.extended_search_area_piv` instead of piv_gpu. Same
config-file-driven format and CLI as the GPU pipelines -- see CONFIG FILE
below. Use this when CUDA/cupy/openpiv_gpu aren't available, or as a CPU
cross-check against the GPU result.

Requires: pip install lvpyio openpiv

CONFIG FILE
-----------
Every setting below lives in a JSON file -- CONFIG_PATH, default
"planar_cpu_piv_config.json" next to this script (or pass a different path
as argv[1]: `python CPU_Planar_Processing.py my_config.json`). If that
file doesn't exist yet, load_controls() writes one out populated with
DEFAULT_CONFIG's values and proceeds using them. You only need to include
the keys you're actually changing -- anything missing from the file falls
back to DEFAULT_CONFIG.

INPUT_MODE options (set in the config file):
  "set"   -- point input_path at either:
               - a single DaVis image set (a .set path, or a plain folder
                 lvpyio can read directly) -- processes just that one set.
               - a folder that itself contains multiple *.set entries --
                 every set inside is batch-processed in turn, each into
                 its own subfolder of output_dir.
             See piv_common.resolve_set_paths() for the exact detection
             rule.
  "loose" -- a plain folder of standalone .im7 files. Auto-detects whether
             each file already contains both exposures (double-frame, one
             file per pair) or frame A/B are separate files matched by
             suffix. Always treated as a single run (no folder-of-sets
             batching).

SINGLE-SET PREVIEW
-------------------
When input_mode="set" and input_path points at exactly one set (not a
folder of several), the FIRST pair's velocity field is computed, plotted,
and opened for review before the rest of that set is processed -- see
piv_common.preview_first_snapshot(). Declining at the prompt aborts the
run. This step is skipped in folder-of-sets batch mode and in "loose"
mode.
"""

import os
import sys
import csv
import numpy as np

import piv_common as pc


# ======================================================================
# Config -- all pipeline settings, defaulted here and overridable via a
# JSON file (see load_controls() and the CONFIG FILE note above)
# ======================================================================
CONFIG_PATH = "planar_cpu_piv_config.json"

DEFAULT_CONFIG = {
    # ---------------- Input source ----------------
    "input_mode": "set",                     # "set" or "loose"
    "input_path": "D:\\messy_data\\Planar\\6-12_5.set",  # .set file / set folder / plain folder / folder-of-sets

    # Only used for input_mode == "set", if a given set turns out to be a
    # DaVis multi-set (e.g. one sub-set per stereo camera). Which sub-set
    # to process; 0 is usually camera 1.
    "multiset_index": 0,

    # Only used for input_mode == "loose" when frame A/B turn out to be
    # SEPARATE files rather than both exposures in one .im7 (auto-detected
    # from the first file; these suffixes only matter in the separate-file
    # case). Adjust to match your actual naming convention.
    "suffix_a": "_a.im7",
    "suffix_b": "_b.im7",
    "loose_glob": "*.im7",                   # glob used to find files in "loose" mode

    # ---------------- Output ----------------
    "output_dir": "piv_output_cpu",

    # ---------------- PIV window size / core settings ----------
    # Forwarded to piv_common.CPUPIVProcess(frame_shape, **cpu_settings),
    # which wraps openpiv-python's basic
    # pyprocess.extended_search_area_piv -- a SINGLE interrogation pass, no
    # window deformation/multi-pass refinement (unlike piv_gpu). Unknown
    # keys are warned about, not silently dropped.
    "cpu_settings": {
        "window_size": 32,
        "search_area_size": 64,
        "overlap_ratio": 0.5,
        "dt": 1.0,
        "sig2noise_method": "peak2mean",
        "sig2noise_threshold": 1.05,
        "subpixel_method": "gaussian",
    },

    # ---------------- Post-processing ----------------
    "global_outlier_std": None,     # e.g. 4.0 rejects |val - mean| > 4*std
    "replace_invalid": False,       # interpolate over invalid/NaN vectors
    "smooth_field": False,
    "smooth_sigma": 1.0,

    # ---------------- Calibration ----------------
    "pixel_pitch_mm": None,         # mm/pixel; None keeps units px/frame
    "frame_dt_s": None,             # s between frames; None keeps units px/frame

    # ---------------- Sign / axis convention ----------------
    "apply_v_sign_flip": False,

    # ---------------- Output artifacts ----------------
    "save_npz": True,
    "save_plot": False,
    "save_summary_csv": False,
    "plot_dpi": 150,
    "quiver_scale": 1000,
    "show_plots": False,

    "verbose": True,
}


class CONTROLS:
    """Populated at runtime by load_controls() -- see DEFAULT_CONFIG and
    the CONFIG FILE note in the module docstring above."""
    pass


def load_controls(config_path):
    return pc.load_controls(config_path, DEFAULT_CONFIG, CONTROLS)


# ======================================================================
# Per-pair processing
# ======================================================================
def handle_pair(process, pair_id, frame_a, frame_b, x, y, ctrl, output_dir):
    if ctrl.verbose:
        print(f"Processing {pair_id} ...", end=" ", flush=True)

    u, v, valid, elapsed = pc.process_frames(process, frame_a, frame_b, ctrl, report_gpu_mem=False)
    u, v = pc.apply_calibration(u, v, ctrl)
    n_valid, n_total = int(valid.sum()), int(valid.size)
    if ctrl.verbose:
        print(f"{elapsed:.3f} s, {n_valid}/{n_total} valid vectors")

    if ctrl.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_velocity.npz"),
                 x=x, y=y, u=u, v=v, valid=valid)

    if ctrl.save_plot:
        pc.plot_and_save_planar(x, y, u, v, valid,
                                 os.path.join(output_dir, f"{pair_id}_quiver.png"),
                                 ctrl, title=f"CPU PIV velocity field -- {pair_id}")

    row = (pair_id, elapsed, n_valid, n_total)
    return row, u, v, valid


def process_pairs(pair_source, ctrl, output_dir, interactive_preview):
    """Run every (pair_id, frame_a, frame_b) from pair_source through the
    same CPU engine (built once, from the first pair's frame shape). If
    interactive_preview, the first pair's result is plotted and the user
    is asked to confirm before the rest are processed."""
    process = None
    x = y = None
    summary_rows = []

    for idx, (pair_id, frame_a, frame_b) in enumerate(pair_source):
        if process is None:
            process, x, y = pc.init_cpu_processor(frame_a.shape, ctrl.cpu_settings)

        row, u, v, valid = handle_pair(process, pair_id, frame_a, frame_b, x, y, ctrl, output_dir)
        summary_rows.append(row)

        if idx == 0 and interactive_preview:
            preview_path = os.path.join(output_dir, f"{pair_id}_first_snapshot_preview.png")
            pc.plot_and_save_planar(x, y, u, v, valid, preview_path, ctrl,
                                     title=f"First snapshot preview (CPU) -- {pair_id}")
            pc.preview_first_snapshot(preview_path)

    return summary_rows


def write_summary(summary_rows, output_dir, ctrl):
    if not summary_rows:
        return
    if ctrl.save_summary_csv:
        csv_path = os.path.join(output_dir, "processing_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "process_time_s", "n_valid", "n_total"])
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    total_time = sum(row[1] for row in summary_rows)
    print(f"Done: {len(summary_rows)} pair(s) in {total_time:.3f} s "
          f"({total_time / len(summary_rows):.3f} s/pair average)")


# ======================================================================
# Main
# ======================================================================
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    ctrl = load_controls(config_path)
    os.makedirs(ctrl.output_dir, exist_ok=True)

    if ctrl.input_mode == "set":
        set_paths, is_batch = pc.resolve_set_paths(ctrl.input_path)
    elif ctrl.input_mode == "loose":
        set_paths, is_batch = [ctrl.input_path], False
    else:
        sys.exit(f"Unknown input_mode: {ctrl.input_mode!r} (use 'set' or 'loose')")

    if is_batch:
        print(f"[info] '{ctrl.input_path}' contains {len(set_paths)} set(s) -- "
              "batch-processing each (no first-snapshot preview in this mode)")

    grand_summary = []
    for set_path in set_paths:
        output_dir = (os.path.join(ctrl.output_dir, pc.set_label(set_path))
                       if is_batch else ctrl.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if ctrl.input_mode == "set":
            print(f"[info] processing set '{set_path}'")
            pair_source = pc.iter_pairs_from_set(ctrl, set_path)
        else:
            pair_source = pc.iter_pairs_from_loose_files(ctrl)

        summary_rows = process_pairs(pair_source, ctrl, output_dir, interactive_preview=not is_batch)
        if not summary_rows:
            print(f"[warn] no image pairs were processed for '{set_path}'")
            continue

        write_summary(summary_rows, output_dir, ctrl)
        grand_summary.extend(summary_rows)

    if not grand_summary:
        sys.exit("No image pairs were processed -- check input_mode/input_path")


if __name__ == "__main__":
    main()
