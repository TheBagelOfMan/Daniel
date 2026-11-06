import os
import sys
import json
import socket
import time
import threading
import ctypes
import tkinter as tk

import numpy as np
import websocket

import algorithm
import osu_memory
from graph_fast import FastGraph


def resource_path(relative_path):
    """Get absolute path to resource — works for dev and when compiled with PyInstaller."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# --- Constants ---

TOSU_HOST = "localhost"
TOSU_PORT = 24050
TOSU_WS = f"ws://{TOSU_HOST}:{TOSU_PORT}/websocket/v2"
# how often to look for tosu and for osu!, and how often to read osu! directly.
TOSU_RECHECK_S = 1.0
OSU_RECHECK_S = 0.5
NATIVE_POLL_S = 0.03

BREAK_ZERO_THRESHOLD_MS = 400
TIME_JUMP_THRESHOLD_MS = 2000
PAUSE_DETECT_S = 0.25
_OSU_TIMEOUT = 1.0

MODE_COMPACT = 0
MODE_STATISTICS = 1
MODE_FULL = 2
MODE_NAMES = ["compact", "statistics", "full"]

GRAPH_HEIGHT = 250
BAR_HEIGHT = 120
WINDOW_WIDTH = 650

COMPACT_HEIGHT = 65
STATISTICS_HEIGHT = 120
FULL_HEIGHT = GRAPH_HEIGHT + BAR_HEIGHT

COMPACT_WIDTH = 550
STATISTICS_WIDTH = 650
FULL_WIDTH = 650

MODE_HEIGHTS = {
    MODE_COMPACT: COMPACT_HEIGHT,
    MODE_STATISTICS: STATISTICS_HEIGHT,
    MODE_FULL: FULL_HEIGHT,
}
MODE_WIDTHS = {
    MODE_COMPACT: COMPACT_WIDTH,
    MODE_STATISTICS: STATISTICS_WIDTH,
    MODE_FULL: FULL_WIDTH,
}

BG_COLOR = "#000000"
PREFIX_FILL = "#FFFFFF"

FONT_SCALE = float(os.environ.get("DANIEL_FONT_SCALE", "1.0" if os.name == "nt" else "0.67"))


def _font_size(size):
    # negative is pixels, which tk scaling never touches (Windows resets it on a display change).
    return -max(8, int(round(size * FONT_SCALE)))


FONT_PREFIX = ("Segoe UI Semibold", _font_size(30))
FONT_DAN = ("Segoe UI Bold", _font_size(45))
FONT_SKILLSET = ("Segoe UI Semibold", _font_size(29))
FONT_CONNECTION = ("Segoe UI Semibold", _font_size(18))

PREFIX_Y_OFFSET = 4.1
SKILLSET_RELEVANCE_FRACTION = 0.20

DAN_COLORS = {
    "4th":     "#FFFFFF",
    "5th":     "#FFFFFF",
    "6th":     "#FFFFFF",
    "7th":     "#FFFFFF",
    "8th":     "#FFFFFF",
    "9th":     "#FFFFFF",
    "10th":    "#FFFFFF",
    "Alpha":   "#ff5a5a",
    "Beta":    "#ffd84d",
    "Gamma":   "#00ffd5",
    "Delta":   "#ff7b00",
    "Epsilon": "#ff7a9e",
    "Zeta":    "#D7F7FF",
    "Eta":     "#ff2b2b",
    "Theta":   "#CC00FF",
    "Iota":    "#FFFFFF",
    "Kappa":   "#FFFFFF",
}

# mean XDC1 rating of each dan's Dan ~ REFORM ~ songs (skillset map packs, Finalmaps, FINAL Ext).
DAN_MEANS = {
    "4th":     3.74,
    "5th":     5.62,
    "6th":     6.54,
    "7th":     7.44,
    "8th":     9.00,
    "9th":     9.45,
    "10th":    10.30,
    "Alpha":   11.28,
    "Beta":    12.04,
    "Gamma":   13.11,
    "Delta":   13.97,
    "Epsilon": 15.12,
    "Zeta":    15.71,
    "Eta":     16.98,
    "Theta":   18.21,
    "Iota":    19.00,
    "Kappa":   19.60,
}
ORDER = list(DAN_MEANS.keys())
DAN_ORDER_START = 4


# --- State ---

lock = threading.Lock()

current_map = None
current_rate = 1.0  # clock rate: 1.5 for DT, 0.75 for HT, anything on lazer
last_state = None
current_song_time_ms = 0

_ws_receive_time = 0.0
_ws_song_time_ms = 0
_prev_song_time_ms = 0
_prev_receive_time = 0.0
_last_message_time = 0.0

_paused = False
_pause_time_ms = 0
_frozen_interp_ms = 0.0
_time_changed_at = 0.0

loading = False
loading_step = 0
_last_loading_dot = 0.0

current_strain_data = None
current_skillset_data = None
connection_phase = "connecting"
source_name = "tosu"  # who is feeding map data: "tosu" or "osu!"

_last_dan_label = "."
_last_dan_numeric = ""
current_mode = MODE_FULL
ln_enabled = True


# --- Window setup ---

if os.name == "nt" and hasattr(ctypes, "windll"):
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

root = tk.Tk()
root.tk.call("tk", "scaling", 1.0)
root.title("Daniel by TheBagelOfMan")
root.geometry(f"{WINDOW_WIDTH}x{FULL_HEIGHT}")
root.resizable(False, False)
root.configure(bg=BG_COLOR)
root.attributes("-topmost", True)


def _set_dark_title_bar(window):
    try:
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass
    try:
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass


root.update_idletasks()
_set_dark_title_bar(root)

_icon_path = resource_path("icon.ico")
if os.path.exists(_icon_path):
    try:
        root.iconbitmap(_icon_path)
    except Exception:
        pass

_icon_png_path = resource_path("icon.png")
if os.path.exists(_icon_png_path):
    try:
        _icon_img = tk.PhotoImage(file=_icon_png_path)
        root.iconphoto(True, _icon_img)
    except Exception:
        pass

canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=FULL_HEIGHT, bg=BG_COLOR, highlightthickness=0)
canvas.pack(expand=True, fill="both")

graph = FastGraph(canvas, GRAPH_HEIGHT, WINDOW_WIDTH)

text_items = []
skillset_items = []
accent_bar = None
current_bar_color = "#333333"
_connection_items = []


# --- Drawing helpers ---

def rgb(hex_color):
    r, g, b = root.winfo_rgb(hex_color)
    return r // 256, g // 256, b // 256


def lerp_color(c1, c2, t):
    r1, g1, b1 = rgb(c1)
    r2, g2, b2 = rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_text(x, y, text, fill, font, anchor="w"):
    return [canvas.create_text(x, y, text=text, fill=fill, font=font, anchor=anchor)]


def draw_outline_text(x, y, text, fill, outline, font):
    items = []
    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        items.append(canvas.create_text(x + ox, y + oy, text=text, fill=outline, font=font, anchor="w"))
    items.append(canvas.create_text(x, y, text=text, fill=fill, font=font, anchor="w"))
    return items


# drawn like "? ? ? ? ?": black text outlined in white, with a white bar.
OUTLINED_DANS = ("Iota", "Kappa")


def is_outlined_text(text):
    return text == "? ? ? ? ?" or text.split(" ")[0] in OUTLINED_DANS


def is_loading_text(text):
    return text in (".", "..", "...")


def _get_text_y_offset():
    return GRAPH_HEIGHT if current_mode == MODE_FULL else 0


# --- Connection screen ---

def _draw_connection_screen():
    global _connection_items

    for item in _connection_items:
        canvas.delete(item)
    _connection_items.clear()

    if connection_phase == "ready":
        return

    if connection_phase == "connecting":
        label = "Waiting for osu!..."
    else:
        label = f"{source_name} connected - waiting for map data"

    cx, cy = MODE_WIDTHS[current_mode] // 2, MODE_HEIGHTS[current_mode] // 2
    note = "osu!lazer needs tosu running" if connection_phase == "connecting" else None
    title = canvas.create_text(
        cx, cy - (16 if note else 0),
        text=label, fill="#AAAAAA",
        font=FONT_CONNECTION, anchor="center",
    )
    _connection_items.append(title)
    if note:
        _connection_items.append(canvas.create_text(
            cx, cy + 16, text=note, fill="#AAAAAA", font=FONT_CONNECTION, anchor="center",
        ))


def _clear_connection_screen():
    global _connection_items

    for item in _connection_items:
        canvas.delete(item)
    _connection_items.clear()

    if current_mode == MODE_FULL and _last_dan_label not in ("Invalid Beatmap", ".", "..", "..."):
        graph.show()


def _clear_normal_ui():
    global text_items, skillset_items, accent_bar

    for item in text_items:
        canvas.delete(item)
    text_items.clear()

    for item in skillset_items:
        canvas.delete(item)
    skillset_items.clear()

    if accent_bar:
        canvas.delete(accent_bar)
        accent_bar = None

    graph.hide()


def _clear_invalid_ui():
    global skillset_items
    for item in skillset_items:
        canvas.delete(item)
    skillset_items.clear()
    graph.hide()


# --- UI components ---

def get_relevant_skillsets(skillset_data):
    overall, skillsets = skillset_data
    threshold = overall * SKILLSET_RELEVANCE_FRACTION
    ranked = sorted(((k, v) for k, v in skillsets.items() if v > 0), key=lambda x: x[1], reverse=True)
    # skillsets within the threshold of overall, and always at least the top one.
    return [kv for i, kv in enumerate(ranked) if i == 0 or overall - kv[1] <= threshold][:3]


def draw_skillsets(skillset_data, color):
    global skillset_items
    for item in skillset_items:
        canvas.delete(item)
    skillset_items.clear()

    if current_mode == MODE_COMPACT or skillset_data is None:
        return

    top3 = get_relevant_skillsets(skillset_data)
    if not top3:
        return

    skillset_str = ", ".join(key.capitalize() for key, _ in top3)
    y = _get_text_y_offset() + 80
    skillset_items += draw_text(14, y, skillset_str, "#FFFFFF", FONT_SKILLSET)


def draw_accent_bar():
    global accent_bar
    if accent_bar:
        canvas.delete(accent_bar)
    h = MODE_HEIGHTS[current_mode]
    accent_bar = canvas.create_rectangle(0, 0, 6, h, fill=current_bar_color, outline="")
    return accent_bar


def fade_items(text_item, bar_item, start_color, end_color, steps=14):
    global current_bar_color

    def _step(i):
        global current_bar_color
        if i < steps:
            color = lerp_color(start_color, end_color, i / steps)
            canvas.itemconfig(text_item, fill=color)
            canvas.itemconfig(bar_item, fill=color)
            current_bar_color = color
            root.after(15, lambda: _step(i + 1))
        else:
            canvas.itemconfig(text_item, fill=end_color)
            canvas.itemconfig(bar_item, fill=end_color)
            current_bar_color = end_color
            if current_mode == MODE_FULL:
                graph.set_color(end_color)

    _step(0)


def update_dan_text(dan_label, dan_numeric):
    global text_items, current_bar_color

    if connection_phase != "ready":
        return

    for item in text_items:
        canvas.delete(item)
    text_items.clear()

    if is_loading_text(dan_label):
        fill = "#888888"
        new_bar_color = "#333333"
    else:
        if dan_label.startswith("<"):
            fill = "#7DF0FF"
            new_bar_color = fill
        else:
            base = dan_label.split()[0]
            fill = DAN_COLORS.get(base, "#FFFFFF")
            new_bar_color = fill

    bar = draw_accent_bar()
    y_off = _get_text_y_offset()
    y = y_off + 28
    prefix_y = y + PREFIX_Y_OFFSET

    prefix = draw_text(14, prefix_y, "Est. Dan:", PREFIX_FILL, FONT_PREFIX)
    text_items.extend(prefix)

    bbox = canvas.bbox(prefix[-1])
    pw = bbox[2] - bbox[0] if bbox else 0
    xpos = 14 + pw + 8

    if is_outlined_text(dan_label):
        dan_items = draw_outline_text(xpos, y, dan_label, fill="#000000", outline="#FFFFFF", font=FONT_DAN)
        new_bar_color = "#FFFFFF"
    else:
        dan_items = draw_text(xpos, y, dan_label, current_bar_color, FONT_DAN)

    text_items.extend(dan_items)

    if not is_loading_text(dan_label) and dan_numeric:
        dan_bbox = canvas.bbox(dan_items[-1])
        numeric_x = (dan_bbox[2] if dan_bbox else xpos) + 10
        text_items.extend(draw_text(numeric_x, prefix_y, f"({dan_numeric})", "#FFFFFF", FONT_PREFIX))

    text_items.extend(draw_ln_label())

    if current_mode != MODE_COMPACT:
        draw_skillsets(current_skillset_data, new_bar_color if not is_loading_text(dan_label) else "#333333")
    else:
        for item in skillset_items:
            canvas.delete(item)
        skillset_items.clear()

    if current_mode == MODE_FULL:
        graph.set_color(
            new_bar_color if (is_loading_text(dan_label) or is_outlined_text(dan_label)) else current_bar_color
        )

    if not is_loading_text(dan_label) and not is_outlined_text(dan_label):
        fade_items(dan_items[-1], bar, current_bar_color, new_bar_color)
    else:
        canvas.itemconfig(bar, fill=new_bar_color)
        current_bar_color = new_bar_color
        if not is_outlined_text(dan_label):
            canvas.itemconfig(dan_items[-1], fill=fill)


def draw_ln_label():
    """Draw LN on/off on the skillset row, right-aligned; L toggles it."""
    if current_mode == MODE_COMPACT:
        return []
    label = "LN on" if ln_enabled else "LN off"
    fill = "#FFFFFF" if ln_enabled else "#888888"
    y = _get_text_y_offset() + 80
    return draw_text(MODE_WIDTHS[current_mode] - 14, y, label, fill, FONT_SKILLSET, anchor="e")


def toggle_ln(event=None):
    global ln_enabled
    ln_enabled = not ln_enabled
    print(f"[LN] {'On' if ln_enabled else 'Off'}")
    if connection_phase == "ready":
        update_dan_text(_last_dan_label, _last_dan_numeric)


def set_dan_text(label, numeric):
    root.after(0, lambda: update_dan_text(label, numeric))


# --- Mode switching ---

def _apply_mode():
    h = MODE_HEIGHTS[current_mode]
    w = MODE_WIDTHS[current_mode]
    root.geometry(f"{w}x{h}")
    canvas.configure(width=w, height=h)

    if current_mode == MODE_FULL:
        graph.show()
    else:
        graph.hide()

    if connection_phase != "ready":
        _draw_connection_screen()
    else:
        update_dan_text(_last_dan_label, _last_dan_numeric)


def cycle_mode(event=None):
    global current_mode
    current_mode = (current_mode + 1) % 3
    _apply_mode()
    print(f"[Mode] Switched to {MODE_NAMES[current_mode]}")


root.bind("<Tab>", cycle_mode)
root.bind("<l>", toggle_ln)
root.bind("<L>", toggle_ln)


# --- Tick loop ---

def _tick():
    global loading_step, _last_loading_dot

    if connection_phase != "ready":
        root.after(16, _tick)
        return

    now = time.monotonic()

    if loading and now - _last_loading_dot >= 0.4:
        dots = [".", "..", "..."]
        update_dan_text(dots[loading_step % 3], "")
        loading_step += 1
        _last_loading_dot = now

    if current_mode == MODE_FULL:
        with lock:
            ws_time = _ws_song_time_ms
            ws_recv = _ws_receive_time
            prev_time = _prev_song_time_ms
            prev_recv = _prev_receive_time
            md = current_rate
            paused = _paused
            frozen_ms = _frozen_interp_ms

        if paused:
            graph.update_position(frozen_ms, md)
        else:
            real_dt = ws_recv - prev_recv
            rate = (ws_time - prev_time) / real_dt if real_dt > 0.01 and ws_time > prev_time else 1000.0
            rate = max(0.0, min(rate, 5000.0))
            interpolated_ms = ws_time + rate * (now - ws_recv)
            graph.update_position(interpolated_ms, md)

    root.after(16, _tick)


# --- Source callbacks ---

def on_open(ws_app, name="tosu"):
    global connection_phase, last_state, source_name
    print(f"[Source] Connected to {name}.")
    source_name = name
    last_state = None
    connection_phase = "waiting_map"
    root.after(0, _draw_connection_screen)


def on_message(ws_app, msg):
    try:
        handle_state(json.loads(msg))
    except ValueError:
        pass


def handle_state(d):
    """Handle one update from tosu's v2 API, or the memory reader in the same shape."""
    global current_map, current_rate, current_song_time_ms
    global _ws_receive_time, _ws_song_time_ms, _prev_song_time_ms, _prev_receive_time
    global connection_phase, last_state, _last_message_time
    global _paused, _pause_time_ms, _frozen_interp_ms, _time_changed_at

    _last_message_time = time.monotonic()

    try:
        folder = d.get("folders", {}).get("beatmap")
        file = d.get("files", {}).get("beatmap")

        if not folder or not file:
            if connection_phase == "ready":
                print("[WS] osu closed — no map data.")
                last_state = None
                connection_phase = "waiting_map"
                root.after(0, _clear_normal_ui)
                root.after(0, _draw_connection_screen)
            return

        songs = d["folders"]["songs"]
        new_map = os.path.join(songs, folder, file)
        new_rate = read_rate(d)
        new_time = d.get("beatmap", {}).get("time", {}).get("live", 0)
        now = time.monotonic()

        with lock:
            prev_ws_time = _ws_song_time_ms
            current_map = new_map
            current_rate = new_rate
            current_song_time_ms = new_time
            # a repeated time is just a sample between osu! frames (fewer when tabbed out).
            if new_time != _ws_song_time_ms:
                _prev_song_time_ms = _ws_song_time_ms
                _prev_receive_time = _ws_receive_time
                _ws_song_time_ms = new_time
                _ws_receive_time = now
                _time_changed_at = now

        sd = current_strain_data
        t_max_ms = float(sd[0][-1]) if (sd is not None and len(sd[0]) > 0) else None
        at_end = (t_max_ms is not None) and (new_time >= t_max_ms - 500)

        time_delta = new_time - prev_ws_time
        jumped = abs(time_delta) > TIME_JUMP_THRESHOLD_MS and not (0 < time_delta < TIME_JUMP_THRESHOLD_MS)

        if jumped and not at_end:
            if _paused:
                _paused = False
                _pause_time_ms = 0
            print(f"[Jump] Time jumped {time_delta:+.0f} ms — clearing pause markers")
            root.after(0, graph.clear_all_pause_markers)

        elif new_time == prev_ws_time:
            # paused only once the time has held still for PAUSE_DETECT_S.
            if not _paused and not at_end and now - _time_changed_at >= PAUSE_DETECT_S:
                _paused = True
                _pause_time_ms = new_time
                _frozen_interp_ms = float(new_time)
                print(f"[Pause] Detected at {new_time} ms")
                root.after(0, lambda t=new_time, r=new_rate: graph.add_pause_marker(t, r))

        else:
            if _paused:
                _paused = False
                _pause_time_ms = 0
                print(f"[Pause] Resumed at {new_time} ms")

        if connection_phase != "ready":
            connection_phase = "ready"
            print("[WS] Map data received. Entering normal operation.")
            root.after(0, _clear_connection_screen)

    except Exception:
        pass


def on_close(ws_app, close_status_code, close_msg):
    global connection_phase, last_state, _paused
    print(f"[Source] Disconnected from {source_name}.")
    last_state = None
    _paused = False
    connection_phase = "connecting"
    root.after(0, _clear_normal_ui)
    root.after(0, _draw_connection_screen)


def on_error(ws_app, error):
    print(f"[WS] Error: {error}")


def read_rate(d):
    """Get the clock rate of the current mods, including lazer's custom speeds."""
    try:
        rate = float(d.get("play", {}).get("mods", {}).get("rate") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    # lazer stores rates as floats (1.2 reads as 1.2000000477).
    return round(rate, 3) if rate > 0 else 1.0


# --- Calculation loop ---

def calculation_loop():
    global last_state, loading, loading_step
    global current_strain_data, current_skillset_data
    global _last_dan_label, _last_dan_numeric

    while True:
        if connection_phase != "ready":
            time.sleep(0.1)
            continue

        with lock:
            state = (current_map, current_rate, ln_enabled)

        mp, rate, ln = state

        if not mp or not os.path.exists(mp):
            time.sleep(0.1)
            continue

        if state == last_state:
            time.sleep(0.1)
            continue

        try:
            loading = True
            loading_step = 0

            import osu_file_parser as osu_parser
            _p = osu_parser.parser(mp)
            _p.process()
            if _p.get_parsed_data()[0] != 4:
                raise ValueError(f"Not a 4k map (keycount={_p.get_parsed_data()[0]})")

            rating, times, strain, details = algorithm.calculate(mp, rate, ln)

            t_arr = np.asarray(times, dtype=float)
            d_arr = np.asarray(strain, dtype=float)
            current_strain_data = (t_arr, d_arr)

            with lock:
                current_skillset_data = (details["rating_raw"], details["skillsets_raw"])

            dan_label, dan_numeric = get_dan_from_diff(rating)

            _last_dan_label = dan_label
            _last_dan_numeric = dan_numeric

            print(f"\n[XDC1 Skillsets] {os.path.basename(mp)} [{rate:g}x] [LN {'on' if ln else 'off'}]")
            for k, v in details["skillsets"].items():
                print(f"{k:<10}: {v:.2f}")
            print(f"Rating    : {rating:.4f}")
            print(f"Rice      : {details['rice']:.4f}")
            print(f"Dan       : {dan_label} ({dan_numeric})\n")

            loading = False
            last_state = state

            if current_mode == MODE_FULL:
                root.after(0, lambda _t=t_arr, _d=d_arr: graph.set_data(_t, _d))
                root.after(0, lambda: graph.set_color(current_bar_color))
                root.after(0, graph.show)
            set_dan_text(dan_label, dan_numeric)

        except Exception as e:
            loading = False
            last_state = state
            print("Calculation error:", e)
            _last_dan_label = "Invalid Beatmap"
            _last_dan_numeric = ""
            with lock:
                current_skillset_data = None
                current_strain_data = None
            root.after(0, _clear_invalid_ui)
            set_dan_text("Invalid Beatmap", "")

        time.sleep(0.1)


# --- Dan boundary tables ---

def _precompute_dan_boundaries():
    means = [DAN_MEANS[d] for d in ORDER]
    boundaries = []
    for i in range(len(ORDER)):
        mean = means[i]
        lower = (means[i - 1] + mean) / 2 if i > 0 else mean - ((means[1] + mean) / 2 - mean)
        upper = (mean + means[i + 1]) / 2 if i < len(means) - 1 else mean + (mean - means[i - 1]) / 2
        boundaries.append((lower, upper))
    return boundaries


_DAN_BOUNDARIES = _precompute_dan_boundaries()


def get_dan_from_diff(diff):
    if diff < _DAN_BOUNDARIES[0][0]:
        return f"<{ORDER[0]} Low", "N/A"
    if diff >= _DAN_BOUNDARIES[-1][1]:
        return "? ? ? ? ?", "N/A"

    for i, dan in enumerate(ORDER):
        lower, upper = _DAN_BOUNDARIES[i]
        if lower <= diff < upper:
            # half a tier either side of its mean, so the mean is always Mid.
            mean = DAN_MEANS[dan]
            if diff < mean:
                t = 0.5 * (diff - lower) / (mean - lower)
            else:
                t = 0.5 + 0.5 * (diff - mean) / (upper - mean)
            t = max(0.0, min(t, 1.0))
            numeric = round(DAN_ORDER_START + i + t, 2)
            if t < 1 / 3:
                label = f"{dan} Low"
            elif t < 2 / 3:
                label = f"{dan} Mid"
            else:
                label = f"{dan} High"
            return label, numeric

    return "? ? ? ? ?", "N/A"


# --- Data sources ---

def _tosu_running():
    # 127.0.0.1, not localhost: Windows tries ::1 first and each refused try costs time.
    try:
        with socket.create_connection(("127.0.0.1", TOSU_PORT), timeout=0.3):
            return True
    except OSError:
        return False


_tosu_up = threading.Event()


def _tosu_watcher():
    """Check for tosu off the reading thread, as a refused connect blocks on Windows."""
    while True:
        if _tosu_running():
            _tosu_up.set()
        else:
            _tosu_up.clear()
        time.sleep(TOSU_RECHECK_S)


def _run_tosu():
    print("[WS] Connecting to tosu...")
    ws = websocket.WebSocketApp(
        TOSU_WS,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error,
    )
    ws.run_forever()


def _run_native(reader):
    """Read osu! until it closes or tosu starts."""
    on_open(None, "osu!")
    while True:
        d = reader.read()
        if d is None:
            print("[osu!] osu! closed.")
            break
        handle_state(d)
        if _tosu_up.is_set():
            print("[osu!] tosu started, switching to it.")
            reader.detach()
            break
        time.sleep(NATIVE_POLL_S)
    on_close(None, None, None)


def _source_loop():
    """Use tosu when it is running, otherwise read osu! stable's memory directly."""
    global connection_phase
    reader = osu_memory.StableReader() if osu_memory.AVAILABLE else None
    _tosu_up.wait(timeout=1.0)  # let the watcher's first check land
    while True:
        if _tosu_up.is_set():
            _run_tosu()
            _tosu_up.clear()  # the watcher sets it again if tosu is still there
        elif reader is not None and reader.attach():
            _run_native(reader)
            continue
        if connection_phase != "connecting":
            connection_phase = "connecting"
            root.after(0, _clear_normal_ui)
            root.after(0, _draw_connection_screen)
        time.sleep(OSU_RECHECK_S)


# --- Boot ---

root.after(100, _draw_connection_screen)


def _message_timeout_watcher():
    global connection_phase, last_state
    while True:
        time.sleep(1)
        if connection_phase != "ready":
            continue
        elapsed = time.monotonic() - _last_message_time
        if elapsed > _OSU_TIMEOUT:
            print(f"[Watcher] No message for {elapsed:.1f}s — osu likely closed.")
            last_state = None
            connection_phase = "waiting_map"
            root.after(0, _clear_normal_ui)
            root.after(0, _draw_connection_screen)


threading.Thread(target=calculation_loop, daemon=True).start()
threading.Thread(target=_tosu_watcher, daemon=True).start()
threading.Thread(target=_source_loop, daemon=True).start()
threading.Thread(target=_message_timeout_watcher, daemon=True).start()

root.after(16, _tick)
root.mainloop()