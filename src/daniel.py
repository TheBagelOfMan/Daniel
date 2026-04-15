import os
import sys
import json
import time
import threading
import ctypes
import tkinter as tk

import numpy as np
import websocket

import algorithm
import msd_converter
from graph_fast import FastGraph


def resource_path(relative_path):
    """Get absolute path to resource — works for dev and when compiled with PyInstaller."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# --- Constants ---

TOSU_WS = "ws://localhost:24050/ws"

BREAK_ZERO_THRESHOLD_MS = 400
TIME_JUMP_THRESHOLD_MS = 2000
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
DOT_RED = "#FF3B3B"
DOT_GREEN = "#00E676"

FONT_SCALE = float(os.environ.get("DANIEL_FONT_SCALE", "1.0" if os.name == "nt" else "0.67"))


def _font_size(size):
    return max(8, int(round(size * FONT_SCALE)))


FONT_PREFIX = ("Segoe UI Semibold", _font_size(30))
FONT_DAN = ("Segoe UI Bold", _font_size(45))
FONT_MSD_SKILL = ("Segoe UI Semibold", _font_size(29))
FONT_CONNECTION = ("Segoe UI Semibold", _font_size(18))

PREFIX_Y_OFFSET = 4.1
MSD_RELEVANCE_FRACTION = 0.15
VIBRO_JACKSPEED_THRESHOLD = 0.90

DAN_COLORS = {
    "Alpha":   "#ff5a5a",
    "Beta":    "#ffd84d",
    "Gamma":   "#00ffd5",
    "Delta":   "#ff7b00",
    "Epsilon": "#ff7a9e",
    "Zeta":    "#D7F7FF",
    "Eta":     "#ff2b2b",
    "Theta":   "#CC00FF",
}

DAN_MEANS = {
    "Alpha":   6.562,
    "Beta":    6.957,
    "Gamma":   7.459,
    "Delta":   7.939,
    "Epsilon": 9.095,
    "Zeta":    9.473,
    "Eta":     10.162,
    "Theta":   10.782,
}
ORDER = list(DAN_MEANS.keys())
DAN_ORDER_START = 11


# --- State ---

lock = threading.Lock()

current_map = None
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

loading = False
loading_step = 0
_last_loading_dot = 0.0

current_strain_data = None
current_msd_data = None
connection_phase = "connecting"

_last_dan_label = "."
_last_dan_numeric = ""
current_mode = MODE_FULL


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
msd_items = []
accent_bar = None
current_bar_color = "#333333"
_connection_items = []
_pulse_job = None


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


def is_loading_text(text):
    return text in (".", "..", "...")


def _get_text_y_offset():
    return GRAPH_HEIGHT if current_mode == MODE_FULL else 0


# --- Connection screen ---

def _draw_connection_screen():
    global _connection_items, _pulse_job

    if _pulse_job is not None:
        root.after_cancel(_pulse_job)
        _pulse_job = None

    for item in _connection_items:
        canvas.delete(item)
    _connection_items.clear()

    if connection_phase == "ready":
        return

    cy = MODE_HEIGHTS[current_mode] // 2

    if connection_phase == "connecting":
        label = "Waiting for tosu connection..."
        dot_color = DOT_RED
    else:
        label = "tosu connected - waiting for map data"
        dot_color = DOT_GREEN

    dot_r = 6
    dot_cx = 20
    inner = canvas.create_oval(
        dot_cx - dot_r, cy - dot_r,
        dot_cx + dot_r, cy + dot_r,
        fill=dot_color, outline="",
    )
    title = canvas.create_text(
        dot_cx + dot_r + 10, cy,
        text=label, fill="#AAAAAA",
        font=FONT_CONNECTION, anchor="w",
    )
    _connection_items += [inner, title]
    _pulse_connection(inner, dot_color, 0)


def _pulse_connection(inner, dot_color, step):
    global _pulse_job

    if connection_phase == "ready":
        _pulse_job = None
        return

    phase = (step % 40) / 40
    alpha = 0.4 + 0.6 * abs(1 - 2 * phase)
    pulsed = lerp_color("#000000", dot_color, alpha)
    canvas.itemconfig(inner, fill=pulsed)
    _pulse_job = root.after(50, lambda: _pulse_connection(inner, dot_color, step + 1))


def _clear_connection_screen():
    global _connection_items, _pulse_job

    if _pulse_job is not None:
        root.after_cancel(_pulse_job)
        _pulse_job = None

    for item in _connection_items:
        canvas.delete(item)
    _connection_items.clear()

    if current_mode == MODE_FULL and _last_dan_label not in ("Invalid Beatmap", ".", "..", "..."):
        graph.show()


def _clear_normal_ui():
    global text_items, msd_items, accent_bar

    for item in text_items:
        canvas.delete(item)
    text_items.clear()

    for item in msd_items:
        canvas.delete(item)
    msd_items.clear()

    if accent_bar:
        canvas.delete(accent_bar)
        accent_bar = None

    graph.hide()


def _clear_invalid_ui():
    global msd_items
    for item in msd_items:
        canvas.delete(item)
    msd_items.clear()
    graph.hide()


# --- UI components ---

def get_relevant_skillsets(msd_result):
    overall = msd_result.get("overall", 0)
    threshold = overall * MSD_RELEVANCE_FRACTION
    relevant = {k: v for k, v in msd_result.items() if k != "overall" and (overall - v) <= threshold}
    top3 = sorted(relevant.items(), key=lambda x: x[1], reverse=True)[:3]
    jackspeed = msd_result.get("jackspeed", 0)
    is_vibro = (overall > 0) and (jackspeed / overall >= VIBRO_JACKSPEED_THRESHOLD)
    return overall, top3, is_vibro


def draw_msd(msd_result, color):
    global msd_items
    for item in msd_items:
        canvas.delete(item)
    msd_items.clear()

    if current_mode == MODE_COMPACT:
        return

    if msd_result is None:
        y = _get_text_y_offset() + 80
        msd_items += draw_text(14, y, "MSD Error", "#FF4444", FONT_MSD_SKILL)
        return

    overall, top3, _ = get_relevant_skillsets(msd_result)
    if not top3:
        return

    skillset_str = ", ".join(key.capitalize() for key, _ in top3)
    y = _get_text_y_offset() + 80
    msd_items += draw_text(14, y, f"{skillset_str}  {overall:.2f}MSD", "#FFFFFF", FONT_MSD_SKILL)


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

    is_vibro = False
    if (
        not is_loading_text(dan_label)
        and dan_label not in ("Invalid Beatmap", "? ? ? ? ?")
        and current_msd_data is not None
    ):
        _, _, is_vibro = get_relevant_skillsets(current_msd_data)

    if dan_label == "? ? ? ? ?":
        dan_items = draw_outline_text(xpos, y, dan_label, fill="#000000", outline="#FFFFFF", font=FONT_DAN)
        new_bar_color = "#FFFFFF"
    elif is_vibro:
        dan_items = draw_text(xpos, y, "VIBRO", "#FFFFFF", FONT_DAN)
        new_bar_color = "#FFFFFF"
    else:
        dan_items = draw_text(xpos, y, dan_label, current_bar_color, FONT_DAN)

    text_items.extend(dan_items)

    if not is_loading_text(dan_label) and dan_numeric:
        dan_bbox = canvas.bbox(dan_items[-1])
        numeric_x = (dan_bbox[2] if dan_bbox else xpos) + 10
        display_numeric = "N/A" if is_vibro else f"({dan_numeric})"
        text_items.extend(draw_text(numeric_x, prefix_y, display_numeric, "#FFFFFF", FONT_PREFIX))

    if current_mode != MODE_COMPACT:
        draw_msd(current_msd_data, new_bar_color if not is_loading_text(dan_label) else "#333333")
    else:
        for item in msd_items:
            canvas.delete(item)
        msd_items.clear()

    if current_mode == MODE_FULL:
        graph.set_color(
            new_bar_color if (is_loading_text(dan_label) or dan_label == "? ? ? ? ?") else current_bar_color
        )

    if not is_loading_text(dan_label) and dan_label != "? ? ? ? ?":
        fade_items(dan_items[-1], bar, current_bar_color, new_bar_color)
    else:
        canvas.itemconfig(bar, fill=new_bar_color)
        current_bar_color = new_bar_color
        if dan_label != "? ? ? ? ?":
            canvas.itemconfig(dan_items[-1], fill=fill)


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
            mod_rate = current_rate
            paused = _paused
            frozen_ms = _frozen_interp_ms

        if paused:
            graph.update_position(frozen_ms, mod_rate)
        else:
            real_dt = ws_recv - prev_recv
            rate = (ws_time - prev_time) / real_dt if real_dt > 0.01 and ws_time > prev_time else 1000.0
            rate = max(0.0, min(rate, 5000.0))
            interpolated_ms = ws_time + rate * (now - ws_recv)
            graph.update_position(interpolated_ms, mod_rate)

    root.after(16, _tick)


# --- WebSocket callbacks ---

def on_open(ws_app):
    global connection_phase, last_state
    print("[WS] Connected to tosu.")
    last_state = None
    connection_phase = "waiting_map"
    root.after(0, _draw_connection_screen)


def on_message(ws_app, msg):
    global current_map, current_rate, current_song_time_ms
    global _ws_receive_time, _ws_song_time_ms, _prev_song_time_ms, _prev_receive_time
    global connection_phase, last_state, _last_message_time
    global _paused, _pause_time_ms, _frozen_interp_ms

    _last_message_time = time.monotonic()

    try:
        d = json.loads(msg)
        bm = d.get("menu", {}).get("bm")
        if not bm:
            return

        folder = bm["path"]["folder"]
        file = bm["path"]["file"]
        stats = bm["stats"]

        if not folder or not file:
            if connection_phase == "ready":
                print("[WS] osu closed — no map data.")
                last_state = None
                connection_phase = "waiting_map"
                root.after(0, _clear_normal_ui)
                root.after(0, _draw_connection_screen)
            return

        songs = d["settings"]["folders"]["songs"]
        new_map = os.path.join(songs, folder, file)
        new_rate = get_rate(stats)
        new_time = bm.get("time", {}).get("current", 0)
        now = time.monotonic()

        with lock:
            prev_ws_time = _ws_song_time_ms
            current_map = new_map
            current_rate = new_rate
            current_song_time_ms = new_time
            _prev_song_time_ms = _ws_song_time_ms
            _prev_receive_time = _ws_receive_time
            _ws_song_time_ms = new_time
            _ws_receive_time = now

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

        elif new_time == prev_ws_time and not at_end:
            if not _paused:
                _paused = True
                _pause_time_ms = new_time
                _frozen_interp_ms = float(new_time)
                print(f"[Pause] Detected at {new_time} ms")
                root.after(0, lambda t=new_time, m=new_rate: graph.add_pause_marker(t, m))

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
    print(f"[WS] Disconnected from tosu (code={close_status_code}).")
    last_state = None
    _paused = False
    connection_phase = "connecting"
    root.after(0, _clear_normal_ui)
    root.after(0, _draw_connection_screen)


def on_error(ws_app, error):
    print(f"[WS] Error: {error}")


def read_mods(d):
    return (
        d.get("gameplay", {}).get("mods", {}).get("str")
        or d.get("menu", {}).get("mods", {}).get("str")
        or ""
    )


def get_rate(stats):

    ar = stats["AR"]
    memoryAR = stats["memoryAR"]

    # 13 is a zero ms AR
    rate = round((13-memoryAR)/(13-ar), 2)

    return rate


# --- Calculation loop ---

def calculation_loop():
    global last_state, loading, loading_step
    global current_strain_data, current_msd_data
    global _last_dan_label, _last_dan_numeric

    while True:
        if connection_phase != "ready":
            time.sleep(0.1)
            continue

        with lock:
            state = (current_map, current_rate)

        mp, rate = state

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

            SR, times, strain, factors = algorithm.calculate(mp, rate)

            t_arr = np.asarray(times, dtype=float)
            d_arr = np.asarray(strain, dtype=float)
            current_strain_data = (t_arr, d_arr)

            try:
                hitobjects = msd_converter.parse_hitobjects(mp, rate)
                etterna_rows = msd_converter.osu_to_etterna_rows(hitobjects)
                msd_result = msd_converter.calculate_msd(etterna_rows)
                print("\n[MSD Skillsets]")
                for k, v in msd_result.items():
                    print(f"{k:<10}: {v:.2f}")
            except Exception as msd_e:
                print(f"[MSD] Error calculating MSD, skipping: {msd_e}")
                msd_result = None

            with lock:
                current_msd_data = msd_result

            averages = algorithm.factor_averages(times, factors)
            dan_label, dan_numeric = get_dan_from_diff(SR)

            _last_dan_label = dan_label
            _last_dan_numeric = dan_numeric

            print(f"\n[Map Factors] {os.path.basename(mp)} [{rate}]")
            for k, v in averages.items():
                print(f"{k:<6}: {v:.4f}")
            print(f"SR    : {SR:.4f}★")
            print(f"Dan   : {dan_label} ({dan_numeric})\n")

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
                current_msd_data = None
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
            t = max(0.0, min((diff - lower) / (upper - lower), 1.0))
            numeric = round(DAN_ORDER_START + i + t, 2)
            if t < 1 / 3:
                label = f"{dan} Low"
            elif t < 2 / 3:
                label = f"{dan} Mid"
            else:
                label = f"{dan} High"
            return label, numeric

    return "? ? ? ? ?", "N/A"


# --- Boot ---

root.after(100, _draw_connection_screen)


def _ws_loop():
    global connection_phase
    while True:
        print("[WS] Attempting to connect to tosu...")
        ws = websocket.WebSocketApp(
            TOSU_WS,
            on_open=on_open,
            on_message=on_message,
            on_close=on_close,
            on_error=on_error,
        )
        ws.run_forever()
        if connection_phase == "ready":
            connection_phase = "connecting"
            root.after(0, _draw_connection_screen)
        print("[WS] Retrying in 3 seconds...")
        time.sleep(3)


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
threading.Thread(target=_ws_loop, daemon=True).start()
threading.Thread(target=_message_timeout_watcher, daemon=True).start()

root.after(16, _tick)
root.mainloop()
