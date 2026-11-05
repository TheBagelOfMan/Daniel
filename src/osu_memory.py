"""Read the selected map, mods and song time from osu! stable's memory (Windows only)."""

# anchors and pointer chains from LeoBlackMT's osumania_map_analyser (MIT), verified against tosu.
# each anchor is a byte pattern in osu!'s JIT code whose address moves every launch.

import ctypes
import os
import re
import sys
import time
from ctypes import wintypes

AVAILABLE = sys.platform == "win32"

# (pattern, offset added to the match), "??" is any byte.
ANCHORS = {
    # beatmap object = [[base - 0xC]]
    "base": ("F8 01 74 04 83 65", 0),
    # song time (ms) = [[play_time + 0x5]]
    "play_time": ("5E 5F 5D C3 A1 ?? ?? ?? ?? 89 ?? 04", 0),
    # mod bitmask = [[menu_mods]]
    "menu_mods": ("C8 FF ?? ?? ?? ?? ?? 81 0D ?? ?? ?? ?? ?? 08 00 00", 0x9),
}

BEATMAP_FOLDER = 0x78
BEATMAP_FILE = 0x90

MOD_DT = 64
MOD_HT = 256
MOD_NC = 512

MAX_STRING_UNITS = 4096
SCAN_CHUNK = 16 * 1024 * 1024
# a failed scan is a full pass over osu!'s memory, so don't retry it straight away.
RESCAN_WAIT_S = 3.0

if AVAILABLE:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    PROCESS_VM_READ = 0x0010
    PROCESS_QUERY_INFORMATION = 0x0400
    TH32CS_SNAPPROCESS = 0x2
    STILL_ACTIVE = 259
    MEM_COMMIT = 0x1000
    PAGE_GUARD = 0x100
    # read-write and read-write-execute: osu!'s JIT code and its globals live here.
    WRITABLE = (0x04, 0x40)

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("PartitionId", wintypes.WORD),
            ("RegionSize", ctypes.c_size_t),
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        ]

    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    _k32.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W))
    _k32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W))
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _k32.ReadProcessMemory.argtypes = (
        wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    )
    _k32.VirtualQueryEx.restype = ctypes.c_size_t
    _k32.VirtualQueryEx.argtypes = (
        wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(MEMORY_BASIC_INFORMATION), ctypes.c_size_t,
    )
    _k32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    )


def _compile(pattern):
    parts = [b"." if p == "??" else re.escape(bytes([int(p, 16)])) for p in pattern.split()]
    return re.compile(b"".join(parts), re.DOTALL)


def _find_osu_pid():
    snap = _k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return None
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = _k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == "osu!.exe":
                return entry.th32ProcessID
            ok = _k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _k32.CloseHandle(snap)
    return None


def _mods_rate(mask):
    """Get the clock rate from stable's fixed mods: DT/NC 1.5, HT 0.75."""
    if mask & (MOD_DT | MOD_NC):
        return 1.5
    if mask & MOD_HT:
        return 0.75
    return 1.0


def _songs_folder(game_dir):
    """Get BeatmapDirectory from the newest osu!.<user>.cfg, relative to the game folder."""
    value = "Songs"
    try:
        cfgs = [
            os.path.join(game_dir, f) for f in os.listdir(game_dir)
            if f.lower().startswith("osu!.") and f.lower().endswith(".cfg") and f.lower() != "osu!.cfg"
        ]
        if cfgs:
            with open(max(cfgs, key=os.path.getmtime), encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    key, _, v = line.partition("=")
                    if key.strip() == "BeatmapDirectory" and v.strip():
                        value = v.strip()
                        break
    except OSError:
        pass
    return value if os.path.isabs(value) else os.path.join(game_dir, value)


class StableReader:
    """Find osu! and its anchors with attach() (a few seconds), then poll read()."""

    def __init__(self):
        self.handle = None
        self.anchors = {}
        self.songs = None
        self._last_failed_scan = -RESCAN_WAIT_S

    # --- Process ---

    def attach(self):
        if not AVAILABLE:
            return False
        if self.handle and self._alive():
            return len(self.anchors) == len(ANCHORS)
        self.detach()

        pid = _find_osu_pid()
        if pid is None:
            return False
        if time.monotonic() - self._last_failed_scan < RESCAN_WAIT_S:
            return False
        handle = _k32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            return False
        self.handle = handle

        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if _k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            self.songs = _songs_folder(os.path.dirname(buf.value))

        started = time.monotonic()
        self.anchors = self._scan()
        scan_s = time.monotonic() - started
        # osu! JITs the code the anchors sit in after it starts; scan again on the next attach.
        if len(self.anchors) != len(ANCHORS):
            missing = sorted(set(ANCHORS) - set(self.anchors))
            print(f"[osu!] Anchors not found yet: {', '.join(missing)}")
            self._last_failed_scan = time.monotonic()
            self.detach()
            return False
        print(f"[osu!] Attached to osu! (pid {pid}) in {scan_s:.1f}s, songs folder: {self.songs}")
        return True

    def detach(self):
        if self.handle:
            _k32.CloseHandle(self.handle)
        self.handle = None
        self.anchors = {}

    def _alive(self):
        code = wintypes.DWORD()
        return bool(_k32.GetExitCodeProcess(self.handle, ctypes.byref(code))) and code.value == STILL_ACTIVE

    # --- Memory ---

    def _read(self, addr, size):
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_size_t()
        if not _k32.ReadProcessMemory(self.handle, ctypes.c_void_p(addr), buf, size, ctypes.byref(got)):
            raise OSError("read failed")
        if got.value != size:
            raise OSError("short read")
        return buf.raw

    def _u32(self, addr):
        return int.from_bytes(self._read(addr, 4), "little")

    def _i32(self, addr):
        return int.from_bytes(self._read(addr, 4), "little", signed=True)

    def _ptr(self, addr):
        return self._u32(self._u32(addr))

    def _string(self, addr):
        """Read a 32-bit .NET string: length (UTF-16 units) at +0x4, text from +0x8."""
        if not addr:
            return None
        n = self._i32(addr + 0x4)
        if n <= 0 or n > MAX_STRING_UNITS:
            return None
        return self._read(addr + 0x8, n * 2).decode("utf-16-le", errors="replace")

    def _regions(self):
        mbi = MEMORY_BASIC_INFORMATION()
        addr = 0
        while addr < 0xFFFFFFFF:
            if not _k32.VirtualQueryEx(self.handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
                break
            base = mbi.BaseAddress or 0
            if (
                mbi.State == MEM_COMMIT
                and not (mbi.Protect & PAGE_GUARD)
                and (mbi.Protect & 0xFF) in WRITABLE
            ):
                yield base, mbi.RegionSize, mbi.Protect & 0xFF
            addr = base + mbi.RegionSize

    def _scan(self):
        patterns = {k: (_compile(p), off) for k, (p, off) in ANCHORS.items()}
        found = {}
        overlap = 64
        # the anchors are in JIT code, which is read-write-execute, so scan that first.
        regions = sorted(self._regions(), key=lambda r: r[2] != 0x40)
        for base, size, _ in regions:
            pos = 0
            while pos < size and len(found) < len(patterns):
                n = min(SCAN_CHUNK + overlap, size - pos)
                try:
                    data = self._read(base + pos, n)
                except OSError:
                    break
                for key, (rx, off) in patterns.items():
                    if key in found:
                        continue
                    m = rx.search(data)
                    if m:
                        found[key] = base + pos + m.start() + off
                pos += SCAN_CHUNK
            if len(found) == len(patterns):
                break
        return found

    # --- Snapshot ---

    def read(self):
        """Get the fields Daniel uses, shaped like tosu's v2 API, or None if osu! has gone."""
        if not self.handle or not self._alive():
            self.detach()
            return None
        a = self.anchors
        folder = file = None
        time_ms = 0
        rate = 1.0
        try:
            beatmap = self._ptr(a["base"] - 0xC)
            if beatmap:
                folder = self._string(self._u32(beatmap + BEATMAP_FOLDER))
                file = self._string(self._u32(beatmap + BEATMAP_FILE))
        except OSError:
            pass
        try:
            time_ms = self._ptr(a["play_time"] + 0x5)
            if time_ms >= 0x80000000:
                time_ms -= 0x100000000
        except OSError:
            pass
        try:
            rate = _mods_rate(self._ptr(a["menu_mods"]))
        except OSError:
            pass
        return {
            "folders": {"songs": self.songs or "", "beatmap": folder},
            "files": {"beatmap": file},
            "beatmap": {"time": {"live": time_ms}},
            "play": {"mods": {"rate": rate}},
        }
