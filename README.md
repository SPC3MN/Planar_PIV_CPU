# Planar CPU-PIV Processing (raw im7 input)

No-GPU counterpart to
[`Planar_PIV_GPU`](https://github.com/SPC3MN/Planar_PIV_GPU): reads raw
LaVision/DaVis `.im7` image pairs directly and runs them through plain
[`openpiv-python`](https://github.com/OpenPIV/openpiv-python)'s basic
single-pass `pyprocess.extended_search_area_piv` (instead of `piv_gpu`) to
produce a 2-component (u, v) planar velocity field per pair. No CUDA, no
GPU, nothing beyond a normal Python + pip install. Same config-file-driven
format as the rest of this pipeline family.

This is deliberately the **basic** openpiv-python API -- a single
interrogation pass, no window deformation / multi-pass refinement -- so
treat it as a no-GPU fallback or a CPU cross-check, not a
feature-for-feature replacement for the GPU pipeline's accuracy.

## What it does

- Reads `.im7` images directly via `lvpyio` in one of two `input_mode`s:
  - `"set"` -- point `input_path` at either a **single** DaVis image set,
    or a **folder containing several `*.set` entries**, in which case
    every set inside is batch-processed in turn into its own subfolder of
    `output_dir` (see `piv_common.resolve_set_paths()`)
  - `"loose"` -- a plain folder of standalone `.im7` files, auto-detecting
    whether each file already contains both exposures (double-frame, one
    file per pair) or frame A/B are separate files matched by
    `suffix_a`/`suffix_b`
- **Single-set preview:** when `input_path` resolves to exactly one set
  (not a folder of several), the first pair's velocity field is computed,
  plotted, and opened for review -- the run pauses on a terminal `y/N`
  prompt before processing the rest of that set. Skipped entirely in
  folder-of-sets batch mode and in `"loose"` mode.
- Runs `openpiv-python`'s `extended_search_area_piv` once per run (the
  same engine instance is reused across every pair in a set), with
  outlier rejection, invalid-vector interpolation, and smoothing as
  optional post-processing steps, using `sig2noise_val` for vector
  validation
- Saves results per pair as `.npz` (and optionally a quiver plot), plus an
  optional CSV summary across the batch

## Files

| File | Purpose |
|---|---|
| `CPU_Planar_Processing.py` | Entry point -- run this |
| `piv_common.py` | Shared config loading, post-processing, GPU/CPU PIV engine adapters, plain im7 frame iteration, set-folder resolution, preview/confirm prompt |

## Requirements

- Python 3.9+
- No GPU or CUDA Toolkit needed
- [`openpiv-python`](https://github.com/OpenPIV/openpiv-python) (`pip install openpiv`)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration file

All pipeline settings live in a JSON file -- `planar_cpu_piv_config.json`
next to `CPU_Planar_Processing.py` by default, or pass a different path as
the first argument: `python CPU_Planar_Processing.py my_config.json`. On
first run, if that file doesn't exist, the script writes one out populated
with its built-in defaults and proceeds using them. You only need to
include the keys you're actually changing in the file.

## Usage

1. Run `python CPU_Planar_Processing.py` once to generate
   `planar_cpu_piv_config.json` with default values.
2. Set `input_mode`/`input_path` to point at your image set, a folder of
   several image sets, or a loose folder.
3. Edit the rest of the config file (interrogation window settings, output
   options), then run:

   ```bash
   python CPU_Planar_Processing.py
   ```

### Key settings (`planar_cpu_piv_config.json`)

| Setting | Description |
|---|---|
| `input_mode` | `"set"` (DaVis image set(s)) or `"loose"` (plain folder of `.im7` files) |
| `input_path` | `.im7` source -- a single `.set` file/set folder, a folder containing multiple `*.set` entries, or a plain folder (`"loose"` mode) |
| `suffix_a` / `suffix_b` | (`"loose"` mode only) filename suffixes used to pair frame A/B when they aren't combined into one file |
| `loose_glob` | (`"loose"` mode only) glob pattern used to find files in `input_path` |
| `cpu_settings` | Forwarded to `openpiv-python`'s `extended_search_area_piv` -- `window_size`, `search_area_size`, `overlap_ratio`, `dt`, `sig2noise_method`, `sig2noise_threshold`, `subpixel_method`. Unrecognized keys are warned about, not silently dropped. |
| `global_outlier_std` | Reject vectors more than N standard deviations from the mean (`None` disables) |
| `replace_invalid` | Interpolate over invalid/NaN vectors |
| `smooth_field` / `smooth_sigma` | Gaussian-smooth the field |
| `pixel_pitch_mm` / `frame_dt_s` | If both are set, converts `u`/`v` from px/frame to physical velocity; otherwise stays px/frame |
| `apply_v_sign_flip` | Flip the sign of `v` |
| `save_npz` / `save_plot` / `save_summary_csv` | Which output artifacts to write |

## Output

Same layout as `Planar_PIV_GPU`: per-pair `<pair_id>_velocity.npz` and
(optionally) `<pair_id>_quiver.png` in `output_dir` (or
`output_dir/<set_name>` in batch mode), plus an optional
`processing_summary.csv` for the whole batch.
