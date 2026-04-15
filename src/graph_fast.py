import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw

PAD_X = 6
PAD_Y_TOP = 8
PAD_Y_BOT = 6
BG_COLOR_RGB = (0, 0, 0)
UNPLAYED_FILL_RGB = (17, 17, 17)
UNPLAYED_STROKE_RGB = (42, 42, 42)
LINE_WIDTH = 5          # Width at 2x render resolution
LINE_BASELINE_INSET = 4
MAX_GRAPH_POINTS = 300  # Lower cap for smoother curves
SUPERSAMPLE = 2         # Render at 2x then downscale for anti-aliased lines
MIN_BREAK_MS = 2000     # Breaks shorter than this get interpolated through

PAUSE_LINE_COLOR = "#FF3B3B"
PAUSE_LINE_WIDTH = 2


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _lerp_rgb(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


class FastGraph:
    def __init__(self, canvas, graph_height, window_width):
        self.canvas = canvas
        self.graph_height = graph_height
        self.window_width = window_width

        self._w = window_width
        self._h = graph_height
        self._plot_w = self._w - PAD_X
        self._plot_h = self._h - PAD_Y_TOP - PAD_Y_BOT
        self._bottom_y = self._h - PAD_Y_BOT
        self._poly_bottom_y = self._bottom_y + 1

        self._times = None
        self._strain = None
        self._t_min = 0.0
        self._t_max = 1.0
        self._poly_data = None

        self._played_rgb = None
        self._unplayed_rgb = None
        self._composite_rgb = None

        self._color_rgb = (255, 90, 90)
        self._played_fill_rgb = _lerp_rgb(BG_COLOR_RGB, (255, 90, 90), 0.45)
        self._played_stroke_rgb = _lerp_rgb(BG_COLOR_RGB, (255, 90, 90), 0.85)

        self._ppm_header = b"P6\n%d %d\n255\n" % (self._w, self._h)
        self._tk_photo = tk.PhotoImage(width=self._w, height=self._h)
        self._tk_photo.put("#000000", to=(0, 0, self._w, self._h))

        self._canvas_item = self.canvas.create_image(0, 0, image=self._tk_photo, anchor="nw")
        self.canvas.tag_lower(self._canvas_item)

        self._last_split_px = -1
        self._visible = True
        self._pause_line_items = []

    # --- Public API ---

    def set_data(self, times, strain):
        self._times = np.asarray(times, dtype=float)
        self._strain = np.asarray(strain, dtype=float)

        nonzero = np.where(self._strain > 0)[0]
        if len(nonzero) == 0:
            nonzero = np.arange(len(self._times))
        crop_start = max(nonzero[0] - 1, 0)
        crop_end = min(nonzero[-1] + 2, len(self._times))
        self._times = self._times[crop_start:crop_end]
        self._strain = self._strain[crop_start:crop_end]

        if len(self._times) < 2:
            self._poly_data = None
            return

        self._t_min = float(self._times[0])
        self._t_max = float(self._times[-1])
        self._poly_data = self._build_polygon()
        self._rebuild_images()
        self._last_split_px = -1
        self.clear_all_pause_markers()

    def set_color(self, hex_color):
        self._color_rgb = _hex_to_rgb(hex_color)
        self._played_fill_rgb = _lerp_rgb(BG_COLOR_RGB, self._color_rgb, 0.45)
        self._played_stroke_rgb = _lerp_rgb(BG_COLOR_RGB, self._color_rgb, 0.85)

        if self._poly_data is not None:
            self._rebuild_images()
            self._last_split_px = -1

    def hide(self):
        if self._visible and self._canvas_item is not None:
            self.canvas.itemconfigure(self._canvas_item, state="hidden")
        self._visible = False
        self.clear_all_pause_markers()

    def show(self):
        if not self._visible and self._canvas_item is not None:
            self.canvas.itemconfigure(self._canvas_item, state="normal")
        self._visible = True
        self._last_split_px = -1

    def update_position(self, song_time_ms, rate):
        if not self._visible or self._played_rgb is None or self._unplayed_rgb is None:
            return

        scale = 2 / (rate * 2)
        adj_time = song_time_ms * scale
        duration = self._t_max - self._t_min
        frac = max(0.0, min((adj_time - self._t_min) / duration, 1.0)) if duration > 0 else 0.0
        split_px = max(0, min(round(PAD_X + frac * self._plot_w), self._w))

        if split_px == self._last_split_px:
            return
        self._last_split_px = split_px

        buf = self._composite_rgb
        if split_px > 0:
            buf[:, :split_px, :] = self._played_rgb[:, :split_px, :]
        if split_px < self._w:
            buf[:, split_px:, :] = self._unplayed_rgb[:, split_px:, :]

        self._tk_photo.configure(data=self._ppm_header + buf.tobytes())

    def add_pause_marker(self, song_time_ms, rate):
        """Add a red vertical line at the given song time. Call from main thread only."""
        if not self._visible or self._t_max <= self._t_min:
            return

        scale = 2 / (rate * 2)
        adj_time = song_time_ms * scale
        duration = self._t_max - self._t_min
        frac = max(0.0, min((adj_time - self._t_min) / duration, 1.0))
        x = max(PAD_X, min(round(PAD_X + frac * self._plot_w), self._w - 1))

        if x <= PAD_X:
            return

        hw = max(1, PAUSE_LINE_WIDTH // 2)
        item = self.canvas.create_rectangle(
            x - hw, PAD_Y_TOP, x + hw, self._bottom_y,
            fill=PAUSE_LINE_COLOR, outline="", tags="pause_marker",
        )
        self.canvas.tag_raise(item, self._canvas_item)
        self._pause_line_items.append(item)

    def clear_all_pause_markers(self):
        """Remove every pause marker line. Call from main thread only."""
        for item in self._pause_line_items:
            self.canvas.delete(item)
        self._pause_line_items.clear()

    def destroy(self):
        self.clear_all_pause_markers()
        if self._canvas_item is not None:
            self.canvas.delete(self._canvas_item)
            self._canvas_item = None
        self._tk_photo = None
        self._played_rgb = None
        self._unplayed_rgb = None
        self._composite_rgb = None
        self._poly_data = None
        self._last_split_px = -1

    # --- Internal ---

    def _build_polygon(self):
        t = self._times.copy()
        d = self._strain.copy()

        is_zero = (d == 0).astype(np.int8)
        transitions = np.diff(is_zero, prepend=0, append=0)
        gap_starts = np.where(transitions == 1)[0]
        gap_ends = np.where(transitions == -1)[0]

        for gs, ge in zip(gap_starts, gap_ends):
            gap_duration = t[min(ge, len(t) - 1)] - t[max(gs - 1, 0)]
            if gap_duration < MIN_BREAK_MS and gs > 0 and ge < len(d):
                val_before = d[gs - 1]
                val_after = d[ge] if ge < len(d) else 0
                n_gap = ge - gs
                if n_gap > 0:
                    d[gs:ge] = np.linspace(val_before, val_after, n_gap + 2)[1:-1]

        d_max = max(d.max(), 1.0)
        px_x = PAD_X + (t - self._t_min) / (self._t_max - self._t_min) * self._plot_w
        px_y = self._h - PAD_Y_BOT - d / d_max * self._plot_h

        n = len(d)
        if n > MAX_GRAPH_POINTS:
            is_zero = (d == 0).astype(np.int8)
            transitions = np.abs(np.diff(is_zero))
            critical = set(np.where(transitions == 1)[0].tolist())
            critical |= set((np.where(transitions == 1)[0] + 1).tolist())
            critical = {i for i in critical if 0 <= i < n}
            base = set(np.round(np.linspace(0, n - 1, MAX_GRAPH_POINTS)).astype(int).tolist())
            keep = sorted(base | critical)
            idx = np.array(keep)
            px_x = px_x[idx]
            px_y = px_y[idx]

        x0 = float(px_x[0])
        x1 = float(px_x[-1])
        poly_bottom = float(self._poly_bottom_y)
        line_bottom = float(self._bottom_y - LINE_BASELINE_INSET)

        pts = [(float(x), float(y)) for x, y in zip(px_x, px_y)]
        poly = [(x0, poly_bottom)] + pts + [(x1, poly_bottom)]
        line = [(x0, line_bottom)] + pts + [(x1, line_bottom)]

        return [poly], [line]

    def _rebuild_images(self):
        if self._poly_data is None:
            return
        self._unplayed_rgb = self._render_to_numpy(UNPLAYED_FILL_RGB, UNPLAYED_STROKE_RGB)
        self._played_rgb = self._render_to_numpy(self._played_fill_rgb, self._played_stroke_rgb)
        self._composite_rgb = np.empty_like(self._unplayed_rgb)

    def _render_to_numpy(self, fill_rgb, stroke_rgb):
        ss = SUPERSAMPLE
        sw, sh = self._w * ss, self._h * ss

        img = Image.new("RGB", (sw, sh), BG_COLOR_RGB)
        draw = ImageDraw.Draw(img)

        polys, lines = self._poly_data

        for seg_poly in polys:
            if len(seg_poly) >= 3:
                draw.polygon([(x * ss, y * ss) for x, y in seg_poly], fill=fill_rgb)

        for seg_line in lines:
            if len(seg_line) >= 2:
                draw.line([(x * ss, y * ss) for x, y in seg_line], fill=stroke_rgb, width=LINE_WIDTH)

        img = img.resize((self._w, self._h), Image.LANCZOS)
        return np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((self._h, self._w, 3)).copy()
