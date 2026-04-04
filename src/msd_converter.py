import json
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(__file__)


def _resolve_msd_command():
    env_path = os.environ.get("MSD_BIN_PATH")
    if env_path:
        if os.name != "nt" and env_path.lower().endswith(".exe"):
            wine = shutil.which("wine64") or shutil.which("wine")
            if wine:
                return [wine, env_path], env_path
        return [env_path], env_path

    windows_msd = os.path.join(BASE_DIR, "msd.exe")
    native_msd = os.path.join(BASE_DIR, "msd")

    if os.name == "nt":
        return [windows_msd], windows_msd

    if os.path.exists(native_msd):
        return [native_msd], native_msd

    if os.path.exists(windows_msd):
        wine = shutil.which("wine64") or shutil.which("wine")
        if wine:
            return [wine, windows_msd], windows_msd

    return [native_msd], native_msd


def parse_hitobjects(osu_file, mod="NM"):
    hitobjects = []
    in_section = False

    with open(osu_file, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()

            if line == "[HitObjects]":
                in_section = True
                continue

            if not in_section or not line:
                continue

            parts = line.split(",")
            x = int(parts[0])
            time = int(parts[2])
            obj_type = int(parts[3])

            if mod == "DT":
                time = int(time * 2 / 3)
            elif mod == "HT":
                time = int(time * 4 / 3)

            hitobjects.append({"x": x, "time": time, "type": obj_type})

    return hitobjects


def osu_to_etterna_rows(hitobjects, keycount=4):
    rows = {}
    column_width = 512 / keycount

    for obj in hitobjects:
        time = round(obj["time"] / 1000.0, 4)
        column = int(obj["x"] // column_width)
        rows[time] = rows.get(time, 0) | (1 << column)
        # LN releases are intentionally ignored (obj_type & 128)

    return [{"notes": rows[t], "time": t} for t in sorted(rows)]


def calculate_msd(notes):
    cmd, msd_path = _resolve_msd_command()

    if not os.path.exists(msd_path):
        raise FileNotFoundError(
            f"MSD binary not found at '{msd_path}'. Set MSD_BIN_PATH or add a compatible executable to src/."
        )

    if os.name != "nt" and msd_path.lower().endswith(".exe") and len(cmd) == 1:
        raise RuntimeError(
            "Found msd.exe on Linux/macOS, but Wine is not installed. Install Wine or provide a native msd via MSD_BIN_PATH."
        )

    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    p = subprocess.Popen(cmd, **popen_kwargs)
    output, err = p.communicate(json.dumps(notes))

    if err:
        print("MSD ERROR:", err, file=sys.stderr)

    return json.loads(output)
