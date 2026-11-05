import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import dan_scale
import osu_file_parser as osu_parser

BASE_DIR = os.path.dirname(__file__)

# --- Constants ---

BREAK_ZERO_THRESHOLD_MS = 400
GRAPH_RESAMPLE_INTERVAL_MS = 100
SMOOTH_SIGMA_MS = 800
LN_TYPE_FLAG = 128


# --- Helper Functions ---

def gaussian_filter1d(data, sigma, mode="constant", cval=0.0):
    kernel_radius = int(4 * sigma + 0.5)
    x = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    if mode == "constant":
        padded = np.pad(data, kernel_radius, mode="constant", constant_values=cval)
    else:
        padded = np.pad(data, kernel_radius, mode=mode)
    return np.convolve(padded, kernel, mode="valid")


def _resolve_xdc1_command():
    env_path = os.environ.get("XDC1_BIN_PATH")
    if env_path:
        if os.name != "nt" and env_path.lower().endswith(".exe"):
            wine = shutil.which("wine64") or shutil.which("wine")
            if wine:
                return [wine, env_path], env_path
        return [env_path], env_path

    windows_bin = os.path.join(BASE_DIR, "xdc1.exe")
    native_bin = os.path.join(BASE_DIR, "xdc1")

    if os.name == "nt":
        return [windows_bin], windows_bin

    if os.path.exists(native_bin):
        return [native_bin], native_bin

    if os.path.exists(windows_bin):
        wine = shutil.which("wine64") or shutil.which("wine")
        if wine:
            return [wine, windows_bin], windows_bin

    return [native_bin], native_bin


# --- Preprocessing ---

def _apply_rate(t, rate):
    """Map time at a clock rate (DT = 1.5, HT = 0.75, or any lazer rate)."""
    return t if rate == 1.0 else int(math.floor(t / rate))


def preprocess_file(file_path, rate):
    """Read notes as (column, head, tail) with the rate applied; tail is -1 for a tap."""
    p_obj = osu_parser.parser(file_path)
    p_obj.process()
    p = p_obj.get_parsed_data()

    note_seq = []
    for k, h, e, typ in zip(p[1], p[2], p[3], p[4]):
        t = _apply_rate(e, rate) if typ & LN_TYPE_FLAG else -1
        note_seq.append((k, _apply_rate(h, rate), t))
    note_seq.sort(key=lambda n: (n[1], n[0]))
    return p[0], note_seq


# --- XDC1 ---

def run_xdc1(note_seq):
    cmd, bin_path = _resolve_xdc1_command()

    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"XDC1 binary not found at '{bin_path}'. Set XDC1_BIN_PATH or add a compatible executable to src/."
        )

    if os.name != "nt" and bin_path.lower().endswith(".exe") and len(cmd) == 1:
        raise RuntimeError(
            "Found xdc1.exe on Linux/macOS, but Wine is not installed. Install Wine or provide a native xdc1 via XDC1_BIN_PATH."
        )

    # the grid starts at 0, so negative times are shifted forward and back again after.
    offset = max(0, -min(n[1] for n in note_seq))
    lines = "".join(
        f"{k} {h + offset} {t + offset if t >= 0 else -1}\n" for k, h, t in note_seq
    )

    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    p = subprocess.Popen(cmd + ["--graph"], **popen_kwargs)
    output, err = p.communicate(lines)

    if err:
        print("XDC1 ERROR:", err, file=sys.stderr)

    # XDC1 prints points; Daniel works on the raw scale and shows dans.
    out = json.loads(output)
    raw = dan_scale.points_to_raw(out["rating"])
    raw_rice = dan_scale.points_to_raw(out["rating_rice"])
    skillsets_raw = {k: dan_scale.points_to_raw(v) for k, v in out["skillsets"].items()}
    graph = np.asarray(out["graph"], dtype=float)
    strain_rested = dan_scale.POINTS_C * np.expm1(graph / dan_scale.POINTS_A) / dan_scale.STRAIN_TO_RAW
    return {
        "rating": dan_scale.to_dan(raw),
        "rating_rice": dan_scale.to_dan(raw_rice),
        "rating_raw": raw,
        "skillsets": {k: dan_scale.to_dan(v) for k, v in skillsets_raw.items()},
        "skillsets_raw": skillsets_raw,
        "grid": np.asarray(out["grid"], dtype=float) - offset,
        "strain_rested": strain_rested,
    }


# --- Graph Post-Processing ---

def _apply_proximity_envelope(all_corners, D_all, note_seq):
    if not note_seq:
        return D_all.copy()

    note_times = np.sort(np.array([float(n[1]) for n in note_seq]))
    PROXIMITY_FADE_MS = 500.0

    idx = np.searchsorted(note_times, all_corners)
    d_after = np.abs(note_times[np.clip(idx, 0, len(note_times) - 1)] - all_corners)
    d_before = np.abs(note_times[np.clip(idx - 1, 0, len(note_times) - 1)] - all_corners)
    d = np.minimum(d_after, d_before)

    envelope = 0.5 * (1.0 + np.cos(np.pi * np.clip(d / PROXIMITY_FADE_MS, 0.0, 1.0)))
    return D_all * envelope


def smooth_D_for_graph(all_corners, D_all, note_seq):
    note_times = np.array(sorted(float(n[1]) for n in note_seq), dtype=float)

    t_start = float(all_corners[0])
    t_end = float(all_corners[-1])
    uniform_t = np.arange(t_start, t_end + GRAPH_RESAMPLE_INTERVAL_MS, GRAPH_RESAMPLE_INTERVAL_MS, dtype=float)

    if len(note_times) > 0:
        idx = np.searchsorted(note_times, uniform_t)
        idx_after = np.clip(idx, 0, len(note_times) - 1)
        idx_before = np.clip(idx - 1, 0, len(note_times) - 1)
        dist = np.minimum(np.abs(uniform_t - note_times[idx_before]), np.abs(uniform_t - note_times[idx_after]))
        break_mask = dist > BREAK_ZERO_THRESHOLD_MS
    else:
        break_mask = np.zeros(len(uniform_t), dtype=bool)

    uniform_D = np.interp(uniform_t, all_corners, D_all)
    uniform_D[break_mask] = 0.0

    sigma_samples = SMOOTH_SIGMA_MS / GRAPH_RESAMPLE_INTERVAL_MS
    uniform_result = gaussian_filter1d(uniform_D, sigma=sigma_samples, mode="constant", cval=0.0)
    uniform_result[break_mask] = 0.0

    return np.interp(all_corners, uniform_t, uniform_result)


# --- Main Entry Point ---

def calculate(file_path, rate=1.0, ln=True):
    """Return (dan, grid times, graph strain, details). ln=False reads long notes as taps."""
    K, note_seq = preprocess_file(file_path, rate)
    if not ln:
        note_seq = [(k, h, -1) for k, h, _ in note_seq]
    if K != 4:
        raise ValueError(f"XDC1 is 4k only (keycount={K})")
    if len(note_seq) < 2:
        raise ValueError("Not enough notes")

    result = run_xdc1(note_seq)
    grid = result["grid"]
    if len(grid) < 2:
        raise ValueError("Not enough notes")

    D_pre = _apply_proximity_envelope(grid, result["strain_rested"], note_seq)
    D_graph = smooth_D_for_graph(grid, D_pre, note_seq)

    return (
        result["rating"],
        grid,
        D_graph,
        {
            "rating_raw": result["rating_raw"],
            "rice": result["rating_rice"],
            "skillsets": result["skillsets"],
            "skillsets_raw": result["skillsets_raw"],
        },
    )
