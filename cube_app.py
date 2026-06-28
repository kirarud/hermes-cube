#!/usr/bin/env python3
"""Hermes Cube — Desktop Particle Avatar
System tray app with animated 3D particle cube.

Single-instance guard: предотвращает запуск второго окна.
"""

from __future__ import annotations

import sys
import os
import tempfile
import atexit

# ── Single-instance lock ───────────────────────────────────────────────
_LOCK_FILE: str = os.path.join(
    tempfile.gettempdir(), 'hermes_cube.lock',
)

def _check_single_instance() -> None:
    """Exit silently if another Hermes Cube is already running."""
    try:
        # Try to create lock file exclusively
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        # Remove lock on normal exit
        atexit.register(lambda: os.unlink(_LOCK_FILE))
    except FileExistsError:
        # Lock exists — check if PID still alive
        try:
            with open(_LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            if sys.platform == 'win32':
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(
                    0x0400, False, old_pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    sys.exit(0)
            else:
                os.kill(old_pid, 0)
                sys.exit(0)
        except (ValueError, OSError, ProcessLookupError):
            try:
                os.unlink(_LOCK_FILE)
            except OSError:
                pass
            _check_single_instance()
            return

_check_single_instance()

from collections import deque
from typing import Any, Dict, Deque, List, Optional, Tuple

import json
import threading

import math

import numpy as np
from numpy.typing import NDArray

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

# Renderer — единый слой отрисовки облака точек (numpy-буфер → 1 картинка)
from renderer import PointCloudRenderer

# PixelGrid — framebuffer agent overlay
from pixel_grid import PixelGrid, PixelGridWindow

# AI-ядро (чат, настроения, буквы, ввод)
from ai_module import AiCore, AI_MOODS

# CubeAgents — pixel UI agents
from cube_agents import AgentManager
from particle_agents import ParticleAgentManager
from cube_agent_demo import run_demo

# Char cube — symbols on cube faces
from char_cube import SYMBOL_SETS

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.GetWindowLongW.restype = ctypes.c_long
    _user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    _user32.SetWindowLongW.restype = ctypes.c_long
    _user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    _user32.SetWindowPos.restype = ctypes.c_bool
    _user32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, ctypes.c_uint32, ctypes.c_byte,
                                                   ctypes.c_uint32]
    _user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
    _user32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_bool]
    _user32.InvalidateRect.restype = ctypes.c_bool
    _user32.UpdateWindow.argtypes = [wintypes.HWND]
    _user32.UpdateWindow.restype = ctypes.c_bool
else:
    _user32 = None  # type: ignore[assignment]

try:
    import pystray
except ImportError:
    pystray = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

APPDATA_DIR: str = os.environ.get('APPDATA', os.path.expanduser('~'))
CONFIG_DIR: str = os.path.join(APPDATA_DIR, 'HermesCube')
CONFIG_FILE: str = os.path.join(CONFIG_DIR, 'config.json')
os.makedirs(CONFIG_DIR, exist_ok=True)

TRANSPARENT_COLOR: str = '#000001'
TRANSPARENT_RGB: int = 0x000100  # BGR little-endian: (0, 1, 0)

#: Particle tile size constraints (px)
MIN_CELL_SIZE: int = 2
MAX_CELL_SIZE: int = 12
MIN_DENSITY: int = 6
MAX_DENSITY: int = 20

#: Rendering interval (~24 fps)
FRAME_MS: int = 42

#: How long the hint overlay stays visible (ms)
HINT_DURATION_MS: int = 5000

DEFAULT_CONFIG: Dict[str, Any] = {
    'window_width': 600,
    'window_height': 600,
    'rotation_speed': 0.28,
    'pulse_rate': 1.8,
    'pulse_amplitude': 0.12,
    'particle_density': 12,
    'cell_size': 6,
    'cube_scale': 0.27,
    'symbol': 'square',         # 'square' | 'circle' | 'dot'
    'shape_preset': 'cube',     # 'cube' | 'sphere' | 'torus' | 'dna' | 'metaball'
    'morph_progress': 0.0,      # 0 = cube, 1 = target shape
    'particle_mode': 'off',     # 'off' | 'wave' | 'breathe' | 'orbit' | 'geyser'
    'wave_speed': 1.5,          # speed multiplier for particle animation
    'wave_amp': 0.12,           # displacement amplitude for particle animation
    'always_on_top': True,
    'x': None,
    'y': None,
    # Symbol / char mode
    'char_mode': 'dots',          # 'dots' | 'symbols' | 'words' | 'glow'
    'symbol_set': 'default',      # from SYMBOL_SETS keys
}

#: Predefined colour palette for agent sprites
AGENT_PALETTE: Dict[str, str] = {
    'bg': TRANSPARENT_COLOR,
    'skin': '#ffcc00',
    'eye': '#cc8800',
    'mouth': '#ff3300',
    'accent': '#ff6600',
}

# ---------------------------------------------------------------------------
# Config manager (simple repository pattern)
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load config from disk, filling missing keys with defaults."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg: Dict[str, Any] = json.load(f)
        for key, value in DEFAULT_CONFIG.items():
            cfg.setdefault(key, value)
        return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist config to disk."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Shape generators — pure functions mapping cube grid → target shape
# ---------------------------------------------------------------------------

PointArray = NDArray[np.float64]  # shape (N, 3)


def _gen_cube(points: PointArray) -> PointArray:
    """Identity — points stay as unit cube [-1, 1]."""
    return points.copy()


def _gen_sphere(points: PointArray) -> PointArray:
    """Normalise cube points onto unit sphere surface."""
    norms: NDArray[np.float64] = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.clip(norms, 1e-8, None)


def _gen_torus(points: PointArray) -> PointArray:
    """Map cube grid onto a torus (major radius R, minor r)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    R: float = 1.5
    r: float = 0.5
    theta: NDArray[np.float64] = np.arctan2(z, x)
    phi: NDArray[np.float64] = np.arcsin(np.clip(y, -1, 1)) * 2.0
    result = np.zeros_like(points)
    result[:, 0] = (R + r * np.cos(phi)) * np.cos(theta)
    result[:, 1] = r * np.sin(phi)
    result[:, 2] = (R + r * np.cos(phi)) * np.sin(theta)
    return result


def _gen_dna(points: PointArray) -> PointArray:
    """Double-helix twist with variable radius."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    twists: float = 3.0
    angle: NDArray[np.float64] = z * twists * np.pi
    radius: NDArray[np.float64] = 0.7 + 0.3 * np.abs(np.sin(angle * 0.5))
    result = np.zeros_like(points)
    result[:, 0] = radius * np.cos(angle)
    result[:, 1] = y * 0.5
    result[:, 2] = radius * np.sin(angle)
    return result


def _gen_metaball(points: PointArray) -> PointArray:
    """Organic blob via 4 attractor points (metaball isosurface approximation)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    attractors: List[Tuple[float, float, float, float]] = [
        (0.5, 0.5, 0.5, 0.6),
        (-0.5, -0.5, 0.5, 0.6),
        (0.5, -0.5, -0.5, 0.6),
        (-0.5, 0.5, -0.5, 0.6),
    ]
    field: NDArray[np.float64] = np.zeros(len(points))
    for ax, ay, az, strength in attractors:
        dist: NDArray[np.float64] = np.sqrt(
            (x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2
        )
        field += strength / (dist + 0.1)
    field /= field.max()
    scale: NDArray[np.float64] = 0.6 + 0.4 * field
    result = points.copy()
    result[:, 0] *= scale
    result[:, 1] *= scale
    result[:, 2] *= scale
    return result


#: Lookup table: name → generator function
SHAPE_GENERATORS: Dict[str, Any] = {
    'cube': _gen_cube,
    'sphere': _gen_sphere,
    'torus': _gen_torus,
    'dna': _gen_dna,
    'metaball': _gen_metaball,
}

# ---------------------------------------------------------------------------
# Particle animation — numpy-vectorised displacement routines
# ---------------------------------------------------------------------------


def _animate_wave(points: PointArray, t: float, speed: float, amp: float) -> PointArray:
    """3D Lissajous wave field across all particles."""
    result = points.copy()
    if amp < 0.001:
        return result
    w1: NDArray[np.float64] = np.sin(points[:, 1] * 3.0 + t * speed * 2.5) * amp
    w2: NDArray[np.float64] = np.cos(points[:, 0] * 2.5 + t * speed * 1.7) * amp * 0.7
    w3: NDArray[np.float64] = np.sin(points[:, 2] * 3.2 + t * speed * 2.0) * amp * 0.5
    result[:, 0] += w1 * 0.5 + np.cos(points[:, 2] * 2.0 + t * speed * 1.3) * amp * 0.3
    result[:, 1] += w2 + np.sin(points[:, 0] * 3.0 + t * speed * 1.1) * amp * 0.4
    result[:, 2] += w3 + np.cos(points[:, 1] * 2.8 + t * speed * 1.9) * amp * 0.3
    return result


def _animate_breathe(points: PointArray, t: float, speed: float, amp: float) -> PointArray:
    """Each particle oscillates along its own phase axis."""
    if amp < 0.001:
        return points.copy()
    phase: NDArray[np.float64] = (
        points[:, 0] * 1.7 + points[:, 1] * 2.3 + points[:, 2] * 1.1
    )
    result = points.copy()
    result[:, 0] += np.sin(phase + t * speed * 1.5) * amp
    result[:, 1] += np.cos(phase * 1.3 + t * speed * 1.1) * amp
    result[:, 2] += np.sin(phase * 0.7 + t * speed * 1.8) * amp
    return result


def _animate_orbit(points: PointArray, t: float, speed: float, amp: float) -> PointArray:
    """Particles orbit their rest positions in 3D spirals."""
    if amp < 0.001:
        return points.copy()
    phase: NDArray[np.float64] = (
        points[:, 0] * 2.7 + points[:, 1] * 3.1 + points[:, 2] * 1.9
    )
    result = points.copy()
    result[:, 0] += (np.cos(phase + t * speed) * amp
                     + np.sin(phase * 0.5 + t * speed * 0.9) * amp * 0.4)
    result[:, 1] += np.sin(phase * 1.3 + t * speed * 0.7) * amp
    result[:, 2] += (np.cos(phase * 0.7 + t * speed * 1.4) * amp
                     + np.cos(phase * 0.9 + t * speed * 1.1) * amp * 0.4)
    return result


def _animate_geyser(points: PointArray, t: float, speed: float, amp: float) -> PointArray:
    """Particles stream upward, spreading at the top like a geyser."""
    if amp < 0.001:
        return points.copy()
    height: NDArray[np.float64] = (points[:, 1] + 1.0) * 0.5  # 0 = bottom, 1 = top
    spray: NDArray[np.float64] = np.sin(
        t * speed * 2.5 + points[:, 0] * 4.0 + points[:, 2] * 4.0
    )
    spread: NDArray[np.float64] = spray * amp * (0.3 + height * 0.7)
    result = points.copy()
    result[:, 0] += spread
    result[:, 2] += spread
    wobble: NDArray[np.float64] = np.sin(
        t * speed * 3.0 + points[:, 0] * 5.0 + points[:, 2] * 5.0
    )
    result[:, 1] += wobble * amp * 0.25 * height
    return result


#: Lookup table: animation name → displacement function
PARTICLE_ANIMATORS: Dict[str, Any] = {
    'off': lambda p, t, s, a: p.copy(),
    'wave': _animate_wave,
    'breathe': _animate_breathe,
    'orbit': _animate_orbit,
    'geyser': _animate_geyser,
}

# ---------------------------------------------------------------------------
# Rotation utilities (3×3 matrices, numpy-vectorised)
# ---------------------------------------------------------------------------


def _rotate_x(points: PointArray, angle: float) -> PointArray:
    c, s = math.cos(angle), math.sin(angle)
    y = points[:, 1] * c - points[:, 2] * s
    z = points[:, 1] * s + points[:, 2] * c
    result = points.copy()
    result[:, 1] = y
    result[:, 2] = z
    return result


def _rotate_y(points: PointArray, angle: float) -> PointArray:
    c, s = math.cos(angle), math.sin(angle)
    x = points[:, 0] * c + points[:, 2] * s
    z = -points[:, 0] * s + points[:, 2] * c
    result = points.copy()
    result[:, 0] = x
    result[:, 2] = z
    return result


def _rotate_z(points: PointArray, angle: float) -> PointArray:
    c, s = math.cos(angle), math.sin(angle)
    x = points[:, 0] * c - points[:, 1] * s
    y = points[:, 0] * s + points[:, 1] * c
    result = points.copy()
    result[:, 0] = x
    result[:, 1] = y
    return result


# ---------------------------------------------------------------------------
# CubeEngine — orchestrates shape → morph → animate → rotate → colour
# ---------------------------------------------------------------------------


class CubeEngine:
    """
    Pure-computation engine: no Tkinter or windowing.

    Responsibilities:
      1. Generate cube grid points with per-vertex RGB colour
      2. Cache target shape positions
      3. Apply morph interpolation, particle animation, and 3D rotation
      4. Return projected 2D positions, depth, and colours for the renderer
    """

    def __init__(self, density: int = MIN_DENSITY) -> None:
        self.density: int = density
        self.pts: Optional[PointArray] = None
        self.r0: Optional[NDArray[np.float64]] = None
        self.g0: Optional[NDArray[np.float64]] = None
        self.b0: Optional[NDArray[np.float64]] = None
        self.jx: Optional[NDArray[np.float64]] = None
        self.jy: Optional[NDArray[np.float64]] = None
        self.shape_cache: Dict[str, PointArray] = {}
        self._rebuild()

    # ── Grid generation ────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """(Re)generate cube points, vertex colours, jitter, and shape cache."""
        self.pts = self._generate_cube_grid(self.density)
        self.r0 = ((self.pts[:, 0] + 1.0) / 2.0 * 255.0)
        self.g0 = ((self.pts[:, 1] + 1.0) / 2.0 * 255.0)
        self.b0 = ((self.pts[:, 2] + 1.0) / 2.0 * 255.0)

        rng_jx = np.random.default_rng(42)
        self.jx = (rng_jx.random(len(self.pts)) - 0.5) * 0.3
        rng_jy = np.random.default_rng(43)
        self.jy = (rng_jy.random(len(self.pts)) - 0.5) * 0.3

        self._refresh_shape_cache()

    @staticmethod
    def _generate_cube_grid(n: int) -> PointArray:
        """
        Build an (N*N*6, 3) array of evenly-spaced points on the 6 faces
        of a unit cube [-1, 1].
        """
        pts: List[Tuple[float, float, float]] = []
        u: NDArray[np.float64] = np.linspace(-1.0, 1.0, n)
        v: NDArray[np.float64] = np.linspace(-1.0, 1.0, n)
        for ui in u:
            for vi in v:
                pts.append((1.0, ui, vi))    # +X
                pts.append((-1.0, vi, ui))   # -X
                pts.append((ui, 1.0, vi))    # +Y
                pts.append((ui, -1.0, vi))   # -Y
                pts.append((ui, vi, 1.0))    # +Z
                pts.append((ui, vi, -1.0))   # -Z
        return np.array(pts, dtype=np.float64)

    # ── Shape cache ────────────────────────────────────────────────────

    def _refresh_shape_cache(self) -> None:
        """Pre-compute all registered shapes from current cube grid."""
        assert self.pts is not None
        self.shape_cache.clear()
        for name, gen in SHAPE_GENERATORS.items():
            self.shape_cache[name] = gen(self.pts)

    # ── Public interface ───────────────────────────────────────────────

    def recalc(self, cfg: Dict[str, Any]) -> None:
        """Rebuild particle grid if density changed."""
        new_density: int = int(cfg.get('particle_density', MIN_DENSITY))
        if new_density != self.density:
            self.density = new_density
            self._rebuild()

    def get_frame(
        self,
        t: float,
        cfg: Dict[str, Any],
    ) -> Tuple[PointArray, float]:
        """
        Compute the current frame: shape → morph → animate → rotate.

        Returns:
            (projected_3d, pulse): (N, 3) array where columns are
            (screen_x, screen_y, depth_z), and pulse is a 0-1 amplitude
            multiplier.
        """
        speed: float = cfg.get('rotation_speed', 0.28)
        pulse_rate: float = cfg.get('pulse_rate', 1.8)
        pulse_amp: float = cfg.get('pulse_amplitude', 0.12)
        morph: float = cfg.get('morph_progress', 0.0)
        shape_name: str = cfg.get('shape_preset', 'cube')
        anim_name: str = cfg.get('particle_mode', 'off')
        anim_speed: float = cfg.get('wave_speed', 1.5)
        anim_amp: float = cfg.get('wave_amp', 0.12)

        # Pulse
        pulse: float = 1.0 + pulse_amp * math.sin(t * pulse_rate)

        # Shape morph
        assert self.pts is not None
        target: PointArray = self.shape_cache.get(shape_name, self.pts)
        if morph > 0.0:
            points: PointArray = self.pts * (1.0 - morph) + target * morph
        else:
            points = self.pts.copy()

        # Particle animation (before rotation, in local space)
        anim_fn = PARTICLE_ANIMATORS.get(anim_name, PARTICLE_ANIMATORS['off'])
        points = anim_fn(points, t, anim_speed, anim_amp)

        # 3D rotation
        ang_x: float = t * 0.20 * (speed / 0.28)
        ang_y: float = t * speed
        ang_z: float = t * 0.08 * (speed / 0.28)
        points = _rotate_x(points, ang_x)
        points = _rotate_y(points, ang_y)
        points = _rotate_z(points, ang_z)

        return points, pulse


# ---------------------------------------------------------------------------
# Tray icon (pystray, optional)
# ---------------------------------------------------------------------------


def _create_tray_image() -> Image.Image:
    """Load 64×64 tray icon from disk, or fallback to generated pixel icon."""
    tray_path: str = os.path.join(
        os.environ.get('APPDATA', os.path.expanduser('~')),
        'HermesCube', 'tray_icon.png',
    )
    if os.path.isfile(tray_path):
        try:
            img: Image.Image = Image.open(tray_path).convert('RGBA')
            return img.resize((64, 64), Image.LANCZOS)
        except Exception:
            pass
    # Fallback: old pixel icon
    img: Image.Image = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pixel_data: List[Tuple[int, int, int, int, int]] = [
        (16, 8, 255, 50, 50), (18, 8, 255, 100, 50), (20, 8, 50, 255, 50),
        (22, 8, 50, 200, 100), (24, 8, 50, 50, 255), (26, 8, 100, 50, 200),
        (28, 8, 200, 50, 100), (14, 10, 255, 80, 80), (16, 10, 255, 150, 50),
        (18, 10, 100, 255, 100), (20, 10, 80, 200, 120), (22, 10, 80, 80, 255),
        (24, 10, 150, 50, 200), (26, 10, 200, 80, 150), (28, 10, 200, 100, 100),
        (30, 10, 150, 150, 50), (12, 12, 255, 100, 100), (14, 12, 255, 200, 80),
        (16, 12, 150, 255, 150), (18, 12, 100, 255, 200), (20, 12, 100, 100, 255),
        (22, 12, 200, 80, 255), (24, 12, 255, 100, 200), (26, 12, 255, 150, 100),
        (28, 12, 200, 200, 80), (30, 12, 150, 200, 100),
    ]
    for px, py, r, g, b in pixel_data:
        draw.rectangle([px, py, px + 3, py + 3], fill=(r, g, b, 255))
    draw.text((2, 52), '♢', fill=(150, 150, 255, 200))
    return img


def _setup_tray_icon(
    app_ref: Any,
) -> Optional[Any]:
    """
    Create a system-tray icon with menu.

    Returns the pystray Icon instance, or None if pystray is unavailable.
    """
    if pystray is None:
        print("pystray not available", flush=True)
        return None
    try:
        image = _create_tray_image()
        menu = pystray.Menu(
            pystray.MenuItem('♢ Показать/Скрыть', lambda i, m: app_ref.toggle_window()),
            pystray.MenuItem('↕ Переместить', lambda i, m: app_ref._toggle_draggable()),
            pystray.MenuItem('💬 Ввод (C)', lambda i, m: app_ref.ai.toggle_input()),
            pystray.MenuItem('🌠 Трейлы', lambda i, m: app_ref._toggle_trails()),
            pystray.MenuItem('▤ PixelGrid', lambda i, m: app_ref._toggle_pixel_grid()),
            pystray.MenuItem('⚙ Настройки', lambda i, m: app_ref.show_settings()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('✕ Выход', lambda i, m: app_ref.quit_app()),
        )
        icon = pystray.Icon('HermesCube', image, '♢ Hermes Cube', menu)
        if hasattr(icon, 'run_detached'):
            icon.run_detached()
        else:
            threading.Thread(target=icon.run, daemon=False).start()
        # Store the icon's GUID for forced cleanup on crash
        app_ref._tray_guid = 'HermesCube'
        return icon
    except Exception as e:
        print(f"Tray icon failed: {e}", flush=True)
        return None


def _remove_tray_icon_force(guid: str = 'HermesCube') -> None:
    """
    Force-remove tray icon via Win32 Shell_NotifyIconW(NIM_DELETE).
    Работает даже если pystray не успел почистить (SIGKILL/crash).
    """
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        from ctypes import wintypes, c_wchar, byref, create_unicode_buffer

        shell32 = ctypes.windll.shell32
        # GUID struct for NOTIFYICONDATAW
        class GUID(ctypes.Structure):
            _fields_ = [
                ('Data1', wintypes.DWORD),
                ('Data2', wintypes.WORD),
                ('Data3', wintypes.WORD),
                ('Data4', wintypes.BYTE * 8),
            ]

        class NOTIFYICONDATAW(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('hWnd', wintypes.HANDLE),
                ('uID', wintypes.UINT),
                ('uFlags', wintypes.UINT),
                ('uCallbackMessage', wintypes.UINT),
                ('hIcon', wintypes.HANDLE),
                ('szTip', c_wchar * 128),
                ('dwState', wintypes.DWORD),
                ('dwStateMask', wintypes.DWORD),
                ('szInfo', c_wchar * 256),
                ('uVersion', wintypes.UINT),
                ('szInfoTitle', c_wchar * 64),
                ('dwInfoFlags', wintypes.DWORD),
                ('guidItem', GUID),
                ('hBalloonIcon', wintypes.HANDLE),
            ]

        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.uID = 1
        nid.szTip = guid

        # NIM_DELETE = 2
        shell32.Shell_NotifyIconW(2, byref(nid))

        # Also try with different uIDs (some icons use uID=0)
        nid.uID = 0
        shell32.Shell_NotifyIconW(2, byref(nid))
    except Exception:
        pass

    # Fallback: notify explorer to refresh notification area
    try:
        ctypes.windll.user32.PostMessageW(
            0xFFFF,  # HWND_BROADCAST
            0x001A,  # WM_SETTINGCHANGE
            0, 0,
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Tray icon — Win32 API (no pystray dependency)
# ---------------------------------------------------------------------------

#: Register forced cleanup on abnormal exit
atexit.register(_remove_tray_icon_force)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Settings window — real-time controls with scrollable layout
# ---------------------------------------------------------------------------

#: Shared colour values for settings UI
UI_BG: str = '#1a1a2e'
UI_FG: str = '#e0e0e0'
UI_ACCENT: str = '#e94560'
UI_ACTIVE: str = '#0f3460'


class SettingsWindow:
    """Modal settings panel with instant-apply sliders and dropdowns.
    Contains a scrollable frame so all controls fit on small screens (1280×720).
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.config: Dict[str, Any] = dict(app.config)

        self.window = tk.Toplevel(app.root)
        self.window.title('⚙ Hermes Cube — Настройки')
        self.window.geometry('400x420')
        self.window.resizable(True, True)
        self.window.configure(bg=UI_BG)
        self.window.transient(app.root)
        self.window.grab_set()
        self.window.minsize(380, 300)

        # ─── Scrollable frame ───────────────────────────────────────
        canvas = tk.Canvas(self.window, bg=UI_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.window, orient='vertical', command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=UI_BG)

        self.scroll_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')),
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw', width=380)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mousewheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        def _on_mousewheel_linux(event: tk.Event) -> None:
            canvas.yview_scroll(-1 if event.num == 4 else 1, 'units')

        canvas.bind_all('<MouseWheel>', _on_mousewheel, add='+')
        canvas.bind_all('<Button-4>', _on_mousewheel_linux, add='+')
        canvas.bind_all('<Button-5>', _on_mousewheel_linux, add='+')

        # Cleanup bindings on destroy
        self.window.bind(
            '<Destroy>',
            lambda e: (
                canvas.unbind_all('<MouseWheel>'),
                canvas.unbind_all('<Button-4>'),
                canvas.unbind_all('<Button-5>'),
            ),
            add='+',
        )

        # ─── Content frame (parent = scroll_frame) ──────────────────
        parent = self.scroll_frame

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background=UI_BG, foreground=UI_FG,
                        font=('Segoe UI', 10))
        style.configure('TScale', background=UI_BG)

        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)

        row: int = 0
        row = self._add_title(parent, row)
        row = self._add_slider(parent, 'cube_scale', 'Размер куба', 0.08, 0.5, row)
        row = self._add_slider(parent, 'rotation_speed', 'Скорость вращения', 0.05, 1.0, row)
        row = self._add_slider(parent, 'pulse_rate', 'Частота пульсации', 0.3, 5.0, row)
        row = self._add_slider(parent, 'pulse_amplitude', 'Амплитуда пульсации', 0.0, 0.35, row)

        row = self._add_section(parent, 'Форма', row)
        row = self._add_dropdown(parent, 'shape_preset', 'Пресет формы',
                                 ['cube', 'sphere', 'torus', 'dna', 'metaball'], row)
        row = self._add_slider(parent, 'morph_progress', 'Морфинг (куб → форма)', 0.0, 1.0, row)

        row = self._add_section(parent, 'Анимация частиц', row)
        row = self._add_dropdown(parent, 'particle_mode', 'Режим',
                                 ['off', 'wave', 'breathe', 'orbit', 'geyser'], row)
        row = self._add_slider(parent, 'wave_speed', 'Скорость анимации', 0.2, 5.0, row)
        row = self._add_slider(parent, 'wave_amp', 'Амплитуда смещения', 0.0, 0.5, row)

        row = self._add_section(parent, 'Частицы', row)
        row = self._add_slider(parent, 'particle_density', 'Плотность', MIN_DENSITY, MAX_DENSITY, row, 0)
        row = self._add_slider(parent, 'cell_size', 'Размер (px)', MIN_CELL_SIZE, MAX_CELL_SIZE, row, 0)

        row = self._add_section(parent, 'Стиль', row)
        row = self._add_dropdown(parent, 'symbol', 'Форма частиц', ['square', 'circle', 'dot'], row)
        row = self._add_dropdown(parent, 'char_mode', 'Режим символов',
                                 ['dots', 'symbols', 'words', 'glow'], row)
        row = self._add_dropdown(parent, 'symbol_set', 'Набор символов',
                                 list(SYMBOL_SETS.keys()), row)

        row = self._add_topmost_checkbox(parent, row)
        self._add_buttons(parent, row)

    # ── UI helpers ─────────────────────────────────────────────────

    @staticmethod
    def _add_title(parent: tk.Frame, row: int) -> int:
        tk.Label(parent, text='♢ Hermes Cube', fg=UI_ACCENT, bg=UI_BG,
                 font=('Segoe UI', 14, 'bold')).grid(
            row=row, column=0, columnspan=3, pady=(15, 5))
        row += 1
        tk.Label(parent, text='Настройки аватара', fg='#888', bg=UI_BG,
                 font=('Segoe UI', 9)).grid(
            row=row, column=0, columnspan=3)
        row += 1
        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', padx=15, pady=8)
        return row + 1

    @staticmethod
    def _add_section(parent: tk.Frame, title: str, row: int) -> int:
        tk.Label(parent, text=title, fg=UI_FG, bg=UI_BG,
                 font=('Segoe UI', 10, 'bold')).grid(
            row=row, column=0, sticky='w', padx=15, pady=(10, 2))
        return row + 1

    def _add_slider(self, parent: tk.Frame, key: str, label: str,
                    min_v: float, max_v: float, row: int,
                    digits: int = 2) -> int:
        tk.Label(parent, text=label, fg=UI_FG, bg=UI_BG,
                 font=('Segoe UI', 9)).grid(
            row=row, column=0, sticky='w', padx=(15, 5))
        var = tk.DoubleVar(value=self.config.get(key, 1.0))

        def on_change(val: str, k: str = key, v: tk.DoubleVar = var) -> None:
            self.config[k] = float(val)
            self.app.config[k] = float(val)
            if k == 'particle_density':
                self.app.config['particle_density'] = int(self.config[k])
                self.app.engine.recalc(self.app.config)
            elif k == 'cell_size':
                self.app.config['cell_size'] = max(MIN_CELL_SIZE, int(float(val)))

        scale = tk.Scale(parent, from_=min_v, to=max_v,
                         resolution=10 ** -digits,
                         orient=tk.HORIZONTAL, variable=var, command=on_change,
                         length=180, bg=UI_BG, fg=UI_FG,
                         highlightbackground=UI_BG,
                         troughcolor='#16213e', activebackground=UI_ACTIVE)
        val_label = tk.Label(parent, textvariable=var, fg=UI_ACCENT,
                             bg=UI_BG, font=('Segoe UI', 9, 'bold'), width=4)
        scale.grid(row=row, column=1, sticky='ew', padx=(3, 3), pady=2)
        val_label.grid(row=row, column=2, sticky='w', padx=(0, 15))
        return row + 1

    def _add_dropdown(self, parent: tk.Frame, key: str, label: str,
                      options: List[str], row: int) -> int:
        tk.Label(parent, text=label, fg=UI_FG, bg=UI_BG,
                 font=('Segoe UI', 9)).grid(
            row=row, column=0, sticky='w', padx=(15, 5))
        initial: str = self.config.get(key, options[0])
        var = tk.StringVar(value=initial)
        dropdown = ttk.Combobox(parent, textvariable=var,
                                values=options, state='readonly', width=14)
        dropdown.grid(row=row, column=1, sticky='w', padx=5, pady=2)

        def on_change(*_a: Any, k: str = key) -> None:
            self.config[k] = var.get()
            self.app.config[k] = var.get()

        var.trace_add('write', on_change)
        return row + 1

    def _add_topmost_checkbox(self, parent: tk.Frame, row: int) -> int:
        var = tk.BooleanVar(value=self.config.get('always_on_top', True))
        cb = tk.Checkbutton(parent, text='Поверх всех окон',
                            variable=var, bg=UI_BG, fg=UI_FG,
                            selectcolor='#16213e',
                            activebackground=UI_BG, activeforeground=UI_FG,
                            font=('Segoe UI', 9))
        cb.grid(row=row, column=0, columnspan=3, sticky='w', padx=15, pady=8)

        def on_change() -> None:
            self.config['always_on_top'] = var.get()
            self.app.config['always_on_top'] = var.get()
            self.app.root.attributes('-topmost', var.get())

        cb.configure(command=on_change)
        return row + 1

    def _add_buttons(self, parent: tk.Frame, row: int) -> None:
        frame = tk.Frame(parent, bg=UI_BG)
        frame.grid(row=row, column=0, columnspan=3, pady=15)

        def save() -> None:
            # Coerce & round numeric values
            for key in ('cell_size', 'particle_density'):
                self.config[key] = int(self.config[key])
            for key in ('rotation_speed', 'pulse_rate', 'pulse_amplitude'):
                self.config[key] = round(self.config[key], 2)
            self.config['cube_scale'] = round(self.config['cube_scale'], 3)
            # Sync all to app
            for k, v in self.config.items():
                self.app.config[k] = v
            self.app.engine.recalc(self.app.config)
            self.app._auto_resize_window()
            save_config(self.app.config)
            self.window.destroy()

        def cancel() -> None:
            self.window.destroy()

        for text, cmd, col in [
            ('💾 Сохранить', save, 0),
            ('✕ Отмена', cancel, 1),
        ]:
            btn = tk.Button(frame, text=text, command=cmd,
                            bg=UI_ACTIVE, fg=UI_FG,
                            activebackground=UI_ACCENT, activeforeground='#fff',
                            relief=tk.FLAT, padx=12, pady=4,
                            font=('Segoe UI', 9))
            btn.grid(row=0, column=col, padx=5)


# ---------------------------------------------------------------------------
# Convex hull helpers for drag handle
# ---------------------------------------------------------------------------


def _convex_hull_2d(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Monotone chain convex hull. Returns hull vertices in CCW order.

    Args:
        pts: (N, 2) array of 2D points.

    Returns:
        (M, 2) array of hull vertices (M >= 3), or pts if N < 3.
    """
    if len(pts) < 3:
        return pts
    # Sort by x then y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(ox: float, oy: float, ax: float, ay: float,
              bx: float, by: float) -> float:
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(
                lower[-2][0], lower[-2][1],
                lower[-1][0], lower[-1][1],
                p[0], p[1]) <= 0:
            lower.pop()
        lower.append((float(p[0]), float(p[1])))

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(
                upper[-2][0], upper[-2][1],
                upper[-1][0], upper[-1][1],
                p[0], p[1]) <= 0:
            upper.pop()
        upper.append((float(p[0]), float(p[1])))

    pts_out = lower[:-1] + upper[:-1]
    return np.array(pts_out, dtype=np.float64)


def _expand_hull(hull: NDArray[np.float64], pad: float = 24.0) -> NDArray[np.float64]:
    """Expand convex hull vertices outward from centroid by *pad* pixels.

    Args:
        hull: (M, 2) convex hull vertices.
        pad:  outward padding in pixels.

    Returns:
        Expanded (M, 2) hull.
    """
    centroid: NDArray[np.float64] = hull.mean(axis=0)
    vecs: NDArray[np.float64] = hull - centroid
    dists: NDArray[np.float64] = np.linalg.norm(vecs, axis=1, keepdims=True)
    dists = np.where(dists == 0, 1, dists)
    return hull + (vecs / dists) * pad


# ---------------------------------------------------------------------------
# CubeApp — main application controller
# ---------------------------------------------------------------------------


class CubeApp:
    """
    Tkinter application driving the Hermes Cube overlay window.

    Responsibilities:
      - Window creation & configuration (transparent, overrideredirect)
      - Canvas-based particle rendering pipeline
      - Input binding (keyboard, mouse, tray)
      - Animation loop
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = load_config()
        self.engine = CubeEngine(self.config['particle_density'])

        self.running: bool = True
        self.anim_running: bool = False
        self.t0: float = 0.0
        self.frame_count: int = 0
        self._show_hint: bool = True

        # --- Window ---
        self.root = tk.Tk()
        self.root.title('♢ Hermes Cube')
        self.root.protocol('WM_DELETE_WINDOW', self._hide_window)

        w: int = self.config['window_width']
        h: int = self.config['window_height']
        x: Optional[int] = self.config.get('x')
        y: Optional[int] = self.config.get('y')
        pos_x: int = x if x is not None else 100
        pos_y: int = y if y is not None else 100

        self.root.geometry(f'{w}x{h}+{pos_x}+{pos_y}')
        self.root.resizable(True, True)
        self.root.overrideredirect(True)
        self.root.configure(bg=TRANSPARENT_COLOR)
        self.root.attributes('-transparentcolor', TRANSPARENT_COLOR)
        if self.config.get('always_on_top', True):
            self.root.attributes('-topmost', True)

        # --- Canvas ---
        self.canvas = tk.Canvas(
            self.root, bg=TRANSPARENT_COLOR, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- GPU/векторный рендерер облака точек ---
        # Заменяет по одной отрисовку N частиц (coords+itemconfig) на 1 картинку.
        self.renderer: PointCloudRenderer = PointCloudRenderer()
        self.renderer.attach(self.canvas)

        # ─── Fullscreen overlay ─────────────────────────────────────
        # Stretch to whole screen so cube never clips regardless of
        # scale, morph, animation, or pulse.
        # WS_EX_TRANSPARENT makes mouse clicks pass through to windows
        # underneath — keyboard shortcuts still work (S, H, Q).
        sw: int = self.root.winfo_screenwidth()
        sh: int = self.root.winfo_screenheight()
        self.root.geometry(f'{sw}x{sh}+0+0')
        if sys.platform == 'win32' and _user32 is not None:
            hwnd = ctypes.c_void_p(self.root.winfo_id())
            GWL_EXSTYLE = -20
            ex_style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex_style |= 0x00000020  # WS_EX_TRANSPARENT
            _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

        self.root.update()

        # Particle display items
        self.particle_items: List[int] = []
        self._current_symbol: str = 'square'
        self._using_chars: bool = False  # toggle when char_mode != 'dots'

        # Drag state
        self._drag_data: Dict[str, int] = {'grab_x': 0, 'grab_y': 0, 'start_ox': 0, 'start_oy': 0}
        self.draggable: bool = False  # T/tray toggle: click-through vs draggable
        self._dragging: bool = False
        self._cube_ox: float = 0.0   # cube offset from center (x)
        self._cube_oy: float = 0.0   # cube offset from center (y)
        self._drag_handle: Optional[int] = None  # canvas polygon item ID
        self._handle_color: str = '#000000'  # true black ≠ TRANSPARENT_COLOR (#000001)

        # PixelGrid animation state
        self._pixel_anim_active: bool = False
        self._pixel_anim_frame: int = 0

        # AI-ядро
        self.ai: AiCore = AiCore(self.root)
        self._ai_color_shift: float = 0.0
        self._last_mood: str = 'idle'

        # Trails
        self._trail_enabled: bool = False
        self._trail_history: Deque[dict] = deque(maxlen=12)
        # Слой трейлов для рендерера: (tx, ty, trgb) или None.
        # Рисуется ПОД кубом как одиночные пиксели (cell=1).
        self._trail_layer: Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]] = None
        self._trail_items: List[int] = []  # legacy Canvas-items (больше не используется)

        # Tray reference
        self.tray_icon: Optional[Any] = None

        # ─── Bindings ───────────────────────────────────────────────
        # WS_EX_TRANSPARENT blocks mouse events — toggle with T
        self.root.bind('<Configure>', self._on_resize)
        self.root.bind('<Button-1>', self._drag_start)
        self.root.bind('<B1-Motion>', self._drag_move)
        self.root.bind('<ButtonRelease-1>', self._drag_end)
        self.root.bind('<Escape>', lambda e: self._hide_window())
        self.root.bind('q', lambda e: self._hide_window())
        self.root.bind('h', lambda e: self._hide_window())
        self.root.bind('s', lambda e: self.show_settings())
        self.root.bind('c', lambda e: self.ai.toggle_input())
        self.root.bind('C', lambda e: self.ai.toggle_input())
        self.root.bind('t', lambda e: self._toggle_draggable())
        self.root.bind('T', lambda e: self._toggle_draggable())
        self.root.bind('r', lambda e: self._toggle_trails())
        self.root.bind('R', lambda e: self._toggle_trails())
        self.root.bind('g', lambda e: self._toggle_pixel_grid())
        self.root.bind('G', lambda e: self._toggle_pixel_grid())
        self.root.bind('a', lambda e: self._spawn_agent())
        self.root.bind('A', lambda e: self._spawn_agent())
        # Settings window also opens via tray icon

        # ─── Context menu ───────────────────────────────────────────
        self.context_menu = tk.Menu(
            self.root, tearoff=0, bg=UI_BG, fg=UI_FG,
            activebackground=UI_ACTIVE, activeforeground='#fff',
        )
        self.context_menu.add_command(
            label='♢ Показать/Скрыть', command=self.toggle_window)
        self.context_menu.add_command(
            label='↕ Переместить', command=self._toggle_draggable)
        self.context_menu.add_command(
            label='💬 Ввод (C)', command=self.ai.toggle_input)
        self.context_menu.add_command(
            label='🌠 Трейлы', command=self._toggle_trails)
        self.context_menu.add_command(
            label='▤ PixelGrid', command=self._toggle_pixel_grid)
        self.context_menu.add_command(
            label='👾 Агент (A)', command=self._spawn_agent)
        self.context_menu.add_command(
            label='⚙ Настройки', command=self.show_settings)
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label='✕ Выход', command=self.quit_app)

        # ─── Tray ───────────────────────────────────────────────────
        threading.Thread(target=lambda: setattr(
            self, 'tray_icon', _setup_tray_icon(self)), daemon=False).start()

        # ─── PixelGrid — framebuffer agent overlay ──────────────────
        sw_grid: int = self.root.winfo_screenwidth()
        sh_grid: int = self.root.winfo_screenheight()
        self.pixel_grid = PixelGrid(sw_grid, sh_grid)
        self.pixel_win = PixelGridWindow(self.pixel_grid)
        self.pixel_win._hide()  # start hidden
        self._draw_test_pixel_ui()

        # ─── Agent managers ───────────────────────────────────────────
        self.agent_mgr: AgentManager = AgentManager(self.pixel_grid)
        self.particle_mgr: ParticleAgentManager = ParticleAgentManager(self.pixel_grid)

        # ─── Start animation ────────────────────────────────────────
        self.root.after(100, self._start_anim)

    # ── Window visibility ──────────────────────────────────────────────

    # ── Window auto-resize ─────────────────────────────────────────────

    def _auto_resize_window(self) -> None:
        """Expand window to fit all particles.
        Safe: uses direct scale calculation, no get_frame call,
        no nested update(), hard-capped at screen-safe size.
        Includes pulse amplitude as a multiplier on scale.
        """
        MAX_PX: int = 1280
        try:
            scale_val: float = float(self.config.get('cube_scale', 0.27))
            pulse_amp: float = float(self.config.get('pulse_amplitude', 0.12))
            anim_amp: float = float(self.config.get('wave_amp', 0.12))
            # Pulse makes scale oscillate up to (1 + pulse_amp) × base scale
            pulse_peak: float = 1.0 + pulse_amp
            max_radius: float = 2.0 + anim_amp * 4.0
            w_cur: int = max(10, self.root.winfo_width())
            h_cur: int = max(10, self.root.winfo_height())
            base: float = float(min(w_cur, h_cur))
            if base < 10:
                return
            scale: float = base * scale_val / (1.0 + pulse_amp) * pulse_peak
            needed: float = 2.0 * (max_radius * scale + 60.0)
            if math.isnan(needed) or math.isinf(needed) or needed > MAX_PX:
                needed = float(MAX_PX)
            needed_int: int = max(300, int(needed))
            if needed_int > max(w_cur, h_cur):
                self.root.geometry(f'{needed_int}x{needed_int}')
        except Exception:
            pass

    def show_window(self) -> None:
        """Restore the overlay window."""
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.lift()
        self.root.lift()
        self.root.lift()
        if self.config.get('always_on_top', True):
            self.root.attributes('-topmost', True)
        self.root.update()

    def _hide_window(self) -> None:
        """Hide the overlay window (keeps tray icon alive)."""
        self.root.withdraw()

    def toggle_window(self) -> None:
        """Flip between hidden and visible."""
        if self.root.state() == 'withdrawn':
            self.show_window()
        else:
            self._hide_window()

    # ── Drag & context menu ────────────────────────────────────────────

    def _drag_start(self, event: tk.Event) -> None:
        if not self.draggable:
            return
        self._dragging = True
        self._drag_data['grab_x'] = event.x_root
        self._drag_data['grab_y'] = event.y_root
        self._drag_data['start_ox'] = self._cube_ox
        self._drag_data['start_oy'] = self._cube_oy

    def _drag_move(self, event: tk.Event) -> None:
        if not self.draggable or not self._dragging:
            return
        dx: int = event.x_root - self._drag_data['grab_x']
        dy: int = event.y_root - self._drag_data['grab_y']
        self._cube_ox = self._drag_data['start_ox'] + dx
        self._cube_oy = self._drag_data['start_oy'] + dy

    def _drag_end(self, event: tk.Event) -> None:
        self._dragging = False

    def _toggle_draggable(self) -> None:
        """Toggle between click-through and draggable mode.

        In draggable mode:
          - WS_EX_TRANSPARENT is removed (window catches clicks)
          - A near-black drag-handle rect is shown behind the cube
          - Cursor becomes fleur
        In click-through mode:
          - WS_EX_TRANSPARENT restored
          - Drag-handle rect hidden
          - Cursor reset
        Background stays TRANSPARENT_COLOR at all times.
        """
        self.draggable = not self.draggable
        if sys.platform == 'win32' and _user32 is not None:
            hwnd = ctypes.c_void_p(self.root.winfo_id())
            GWL_EXSTYLE = -20
            ex_style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if self.draggable:
                ex_style &= ~0x00000020  # Remove WS_EX_TRANSPARENT
                self.canvas.config(cursor='fleur')
                # Create drag handle polygon (convex hull, updated every frame)
                if self._drag_handle is None:
                    self._drag_handle = self.canvas.create_polygon(
                        0, 0, 0, 0,  # dummy coords, updated in _render_frame
                        fill=self._handle_color, outline='',
                        width=0, tags='drag_handle',
                    )
                self.canvas.tag_lower('drag_handle')  # behind particles
                self.canvas.itemconfig('drag_handle', state='normal')
            else:
                ex_style |= 0x00000020   # Restore WS_EX_TRANSPARENT
                self.canvas.config(cursor='')
                if self._drag_handle is not None:
                    self.canvas.itemconfig('drag_handle', state='hidden')
            _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)
        # Visual feedback
        mode_text: str = '↕ РЕЖИМ ПЕРЕМЕЩЕНИЯ' if self.draggable else 'ПРОЗРАЧНЫЙ'
        self._show_mode_overlay(mode_text)
        # Update context menu label
        label = '↕ Перемещение: вкл' if self.draggable else '↕ Переместить'
        self.context_menu.entryconfig(1, label=label)

    def _show_mode_overlay(self, text: str) -> None:
        """Brief overlay text in top-left corner."""
        overlay = self.canvas.create_text(
            10, 10, anchor='nw', text=text,
            fill='#ffffff', font=('Segoe UI', 14, 'bold'),
            tags='mode_overlay',
        )
        self.root.after(1200, lambda: self.canvas.delete('mode_overlay'))

    def _toggle_trails(self) -> None:
        """Toggle particle trails on/off (R key)."""
        self._trail_enabled = not self._trail_enabled
        if not self._trail_enabled:
            self._trail_history.clear()
            for item in self._trail_items:
                self.canvas.coords(item, 0, 0, 0, 0)
        mode_text: str = '🌠 ТРЕЙЛЫ ВКЛ' if self._trail_enabled else 'ТРЕЙЛЫ ВЫКЛ'
        self._show_mode_overlay(mode_text)

    def set_trails(self, enabled: bool) -> None:
        """Enable or disable particle trails programmatically (for AiCore)."""
        if enabled == self._trail_enabled:
            return
        self._trail_enabled = enabled
        if not enabled:
            self._trail_history.clear()
            for item in self._trail_items:
                self.canvas.coords(item, 0, 0, 0, 0)

    # ── Agent methods ──────────────────────────────────────────────────

    def _spawn_agent(self) -> None:
        """Spawn a particle-agent from a random cube particle (A key)."""
        if not hasattr(self, 'particle_mgr'):
            return
        t: float = self.root.tk.call('clock', 'milliseconds') / 1000.0
        pts3d, pulse = self.engine.get_frame(t, self.config)
        w: int = max(10, self.canvas.winfo_width())
        h: int = max(10, self.canvas.winfo_height())
        scale: float = (min(w, h) * self.config.get('cube_scale', 0.27)
                        / 1.12 * pulse)
        cx_s: float = w / 2.0 + self._cube_ox
        cy_s: float = h / 2.0 + self._cube_oy
        px: NDArray[np.float64] = pts3d[:, 0] * scale + cx_s
        py: NDArray[np.float64] = pts3d[:, 1] * scale + cy_s

        if len(px) < 1:
            return
        idx: int = int(np.random.rand() * len(px))
        r: int = int(self.engine.r0[idx])
        g: int = int(self.engine.g0[idx])
        b: int = int(self.engine.b0[idx])

        import random as _rnd
        tx: int = _rnd.randint(50, max(51, w - 50))
        ty: int = _rnd.randint(50, max(51, h - 50))
        self.particle_mgr.spawn(
            idx, float(px[idx]), float(py[idx]),
            (r, g, b), 'cursor', float(tx), float(ty),
        )

    def _render_pixel_agents(self) -> None:
        """Periodic render loop for pixel agents (called from after-loop)."""
        if not self._pixel_anim_active:
            return
        if hasattr(self, 'pixel_win') and self.pixel_win.running:
            self.agent_mgr.render_all()
            self.particle_mgr.update_all()
            self.particle_mgr.render_all()
            self.pixel_grid._modified = True
        self.root.after(66, self._render_pixel_agents)

    def _show_context_menu(self, event: tk.Event) -> None:
        self.context_menu.tk_popup(
            self.root.winfo_rootx() + event.x,
            self.root.winfo_rooty() + event.y,
        )

    # ── Rendering pipeline ─────────────────────────────────────────────

    def _start_anim(self) -> None:
        self.t0 = self.root.tk.call('clock', 'milliseconds')
        self.anim_running = True
        self.root.lift()
        self.root.lift()
        self.root.lift()
        self.root.update()
        self._render_frame()

    def _on_resize(self, event: tk.Event) -> None:
        """Cache canvas dimensions to avoid repeated winfo calls."""
        pass  # Dimensions are read fresh each frame via winfo_width/height

    def _render_frame(self) -> None:
        """Main render tick: compute → project → draw → schedule next."""
        if not self.anim_running or not self.running:
            return

        now: float = self.root.tk.call('clock', 'milliseconds')
        elapsed: float = (now - self.t0) / 1000.0
        w: int = max(10, self.canvas.winfo_width())
        h: int = max(10, self.canvas.winfo_height())

        if w < 10 or h < 10:
            self.root.after(FRAME_MS, self._render_frame)
            return

        # ── AI mood override ─────────────────────────────────────────
        mood_params: Dict[str, float] = self.ai.get_mood_override()
        if mood_params:
            self.config['pulse_rate'] = mood_params['pulse_rate']
            self.config['pulse_amplitude'] = mood_params['pulse_amp']
            self.config['rotation_speed'] = mood_params['speed']
        self._ai_color_shift = self.ai.get_color_shift()

        # ── Mood change overlay ────────────────────────────────────────
        current_mood: str = self.ai.mood
        if current_mood != self._last_mood:
            self._last_mood = current_mood
            mood_labels: Dict[str, str] = {
                'idle':     '😐 Бездействие',
                'thinking': '🤔 Размышляет',
                'speaking': '💬 Говорит',
                'happy':    '😊 Радостный',
                'sad':      '😢 Грустный',
            }
            self._show_mode_overlay(
                mood_labels.get(current_mood, current_mood))

        pts3d, pulse = self.engine.get_frame(elapsed, self.config)

        # Project 3D → 2D
        scale: float = (min(w, h) * self.config.get('cube_scale', 0.27)
                        / (1.0 + self.config.get('pulse_amplitude', 0.12))
                        * pulse)
        cx_s: float = w / 2.0 + self._cube_ox
        cy_s: float = h / 2.0 + self._cube_oy

        px: NDArray[np.float64] = pts3d[:, 0] * scale + cx_s
        py: NDArray[np.float64] = pts3d[:, 1] * scale + cy_s
        pz: NDArray[np.float64] = pts3d[:, 2]

        # ── Convex hull drag handle (follows cube shape in real-time) ──
        if self._drag_handle is not None and self.draggable and len(px) >= 3:
            # Stack projected (x, y) and compute convex hull with padding
            xy: NDArray[np.float64] = np.column_stack((px, py))
            hull: NDArray[np.float64] = _convex_hull_2d(xy)
            if len(hull) >= 3:
                hull = _expand_hull(hull, pad=24.0)
                # Flatten to [x1, y1, x2, y2, ...] for canvas polygon coords
                flat_coords: list[float] = hull.ravel().tolist()
                self.canvas.coords(self._drag_handle, *flat_coords)
                self.canvas.tag_lower('drag_handle')
        elif self._drag_handle is not None:
            # Keep handle hidden / zero-area when not in drag mode
            self.canvas.coords(self._drag_handle, 0, 0, 0, 0)

        # Depth sort (painter's algorithm)
        order: NDArray[np.int64] = np.argsort(pz)
        px, py, pz = px[order], py[order], pz[order]

        # Per-particle colour with depth shading
        depth_factor: float = 0.6 + 0.4 * (pz + 1.0) / 2.0
        r_p: NDArray[np.float64]
        g_p: NDArray[np.float64]
        b_p: NDArray[np.float64]
        r_p = np.clip(self.engine.r0[order] * depth_factor, 0, 255)
        g_p = np.clip(self.engine.g0[order] * depth_factor, 0, 255)
        b_p = np.clip(self.engine.b0[order] * depth_factor, 0, 255)

        # ── AI mood HSV color shift (fully vectorized numpy) ─────────
        shift: float = self._ai_color_shift
        if shift > 0.01:
            # Normalize to [0, 1]
            r = r_p / 255.0
            g = g_p / 255.0
            b = b_p / 255.0

            # RGB → HSV (vectorized)
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            delta = mx - mn

            h = np.zeros_like(r)
            mask = delta > 1e-6
            rm = mask & (mx == r)
            gm = mask & (mx == g)
            bm = mask & (mx == b)
            h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6.0
            h[gm] = ((b[gm] - r[gm]) / delta[gm]) + 2.0
            h[bm] = ((r[bm] - g[bm]) / delta[bm]) + 4.0
            h = h / 6.0

            s = np.zeros_like(r)
            s[mask] = delta[mask] / mx[mask]
            v = mx

            # Apply hue shift (wrap around)
            h = (h + shift) % 1.0

            # HSV → RGB (vectorized)
            h6 = h * 6.0
            hi = np.floor(h6).astype(np.int32)
            f = h6 - hi.astype(np.float64)
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))

            r_p = np.clip(np.select(
                [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
                [v, q, p, p, t, v]
            ) * 255.0, 0, 255)
            g_p = np.clip(np.select(
                [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
                [t, v, v, q, p, p]
            ) * 255.0, 0, 255)
            b_p = np.clip(np.select(
                [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
                [p, p, t, v, v, q]
            ) * 255.0, 0, 255)

        cell: int = max(MIN_CELL_SIZE, int(self.config.get('cell_size', MIN_CELL_SIZE)))
        half: int = cell // 2
        symbol: str = self.config.get('symbol', 'square')

        # Symbol-specific size adjustment
        cell_actual: int = cell
        half_actual: int = half
        if symbol == 'dot':
            cell_actual = max(MIN_CELL_SIZE, cell // 2)
            half_actual = cell_actual // 2

        count: int = len(px)

        # ── Trails — light trail effect ───────────────────────────────────
        # Push current frame to history (before drawing trails)
        if self._trail_enabled:
            self._trail_history.append({
                'px': px.copy(),
                'py': py.copy(),
                'r': r_p.copy(),
                'g': g_p.copy(),
                'b': b_p.copy(),
                'cell': cell_actual,
                'half': half_actual,
                'count': count,
            })

        # Рисуем трейлы одним векторным слоем в общий буфер рендерера
        # (раньше: (trail_len-1)*count отдельных Canvas-айтемов, ≈26000 при
        # density=20 → система висла). Теперь: cell=1 (одиночные пиксели),
        # это ~36× быстрее, а шлейф выглядит даже чище.
        trail_len: int = len(self._trail_history)
        if trail_len > 1 and self._trail_enabled:
            # Соберём все пиксели трейлов в плоские массивы, с учётом fade.
            chunks_x: List[NDArray[np.float64]] = []
            chunks_y: List[NDArray[np.float64]] = []
            chunks_rgb: List[NDArray[np.uint8]] = []
            for age in range(1, trail_len):
                frame_data = self._trail_history[trail_len - 1 - age]
                fade: float = 1.0 - (age / self._trail_history.maxlen)
                if fade < 0.05:
                    continue
                fc = frame_data['count']
                if fc == 0:
                    continue
                fade_arr: NDArray[np.float64] = np.asarray(fade, dtype=np.float64)
                col = np.stack([
                    frame_data['r'] * fade_arr,
                    frame_data['g'] * fade_arr,
                    frame_data['b'] * fade_arr,
                ], axis=1).clip(0, 255).astype(np.uint8)
                chunks_x.append(frame_data['px'])
                chunks_y.append(frame_data['py'])
                chunks_rgb.append(col)
            if chunks_x:
                all_tx = np.concatenate(chunks_x)
                all_ty = np.concatenate(chunks_y)
                all_trgb = np.concatenate(chunks_rgb)
                self._trail_layer = (all_tx, all_ty, all_trgb)
        elif not self._trail_enabled:
            # Снять шлейф: очистить историю и флаг слоя
            self._trail_history.clear()
            self._trail_layer = None

        char_mode: str = self.config.get('char_mode', 'dots')
        symbol_set_name: str = self.config.get('symbol_set', 'default')
        symbols_set: List[str] = SYMBOL_SETS.get(symbol_set_name, SYMBOL_SETS['default'])

        # Rebuild particle items if symbol or char_mode changed
        mode_changed: bool = (
            self._current_symbol != symbol
            or (char_mode != 'dots' and not self._using_chars)
            or (char_mode == 'dots' and self._using_chars)
        )
        if mode_changed:
            for item in self.particle_items:
                self.canvas.delete(item)
            self.particle_items.clear()
            self._current_symbol = symbol
            self._using_chars = (char_mode != 'dots')

        # ── Единый рендер-конвейер (и геометрия, и символы через рендерер) ─
        # Раньше: 2400–26000 вызовов canvas.coords()+itemconfig() за кадр.
        # Теперь: 1 картинка через numpy-буфер + 1 blit.
        rgb_arr: NDArray[np.uint8] = np.column_stack(
            (r_p, g_p, b_p),
        ).astype(np.uint8)
        self.renderer.begin_frame()
        # Сначала трейлы (слой ПОД кубом), потом сам куб.
        if self._trail_layer is not None:
            tx, ty, trgb = self._trail_layer
            self.renderer.add_points(tx, ty, trgb, 1, 'square')

        if self._using_chars:
            # Символьный режим: глифы растеризуются ОДИН РАЗ (кеш),
            # штампуются векторно через add_chars.
            char_list: List[str] = [
                symbols_set[i % len(symbols_set)] for i in range(count)
            ]
            self.renderer.add_chars(px, py, rgb_arr, char_list, cell_actual)
        else:
            self.renderer.add_points(px, py, rgb_arr, cell_actual, symbol)
        self.renderer.finish_frame()

        # Startup hint overlay
        if self._show_hint:
            hint_id = self.canvas.create_text(
                w // 2, h - 20,
                text='C — ввод  |  S — настройки  |  R — трейлы  |  G — PixelGrid  |  H — скрыть',
                fill=UI_ACCENT, font=('Segoe UI', 9), anchor='center',
            )
            self.root.after(
                HINT_DURATION_MS,
                lambda: self.canvas.delete(hint_id) if self.canvas.winfo_exists() else None,
            )
            self._show_hint = False

        # ── AI update — spawn response letters from cube center ─────────
        self.ai.update(FRAME_MS / 1000.0)
        if (self.ai.last_response
                and not self.ai._thinking
                and len(self.ai.text_overlay.particles) == 0):
            # Spawn letters at cube center (screen position)
            response_text: str = self.ai.last_response
            self.ai.spawn_response(response_text, int(cx_s), int(cy_s))
            self.ai.last_response = ''  # clear to avoid re-spawn

        self.frame_count += 1
        self.root.after(FRAME_MS, self._render_frame)

    # ── Settings ───────────────────────────────────────────────────────

    def show_settings(self) -> None:
        """Open the settings panel."""
        SettingsWindow(self)

    # ── PixelGrid ───────────────────────────────────────────────────

    def _toggle_pixel_grid(self) -> None:
        """Toggle the PixelGrid framebuffer overlay (G key / tray)."""
        if not hasattr(self, 'pixel_win'):
            return
        if self.pixel_win.root.state() == 'withdrawn':
            self.pixel_win.show()
            self._pixel_anim_active = True
            self._pixel_anim_frame = 0
            self.root.after(100, self._pixel_anim_loop)
        else:
            self._pixel_anim_active = False
            self.pixel_win._hide()
            if hasattr(self, 'pixel_grid'):
                self.pixel_grid.clear()

    def _draw_test_pixel_ui(self) -> None:
        """Draw a demo pixel-art panel on the PixelGrid."""
        if not hasattr(self, 'pixel_grid'):
            return
        pg = self.pixel_grid
        sw = pg.w
        sh = pg.h

        # ── Bottom-right control panel ──────────────────────────────
        panel_w: int = 140
        panel_h: int = 60
        px: int = sw - panel_w - 20
        py: int = sh - panel_h - 20

        # Background
        pg.paint_rect(px, py, px + panel_w, py + panel_h, (20, 20, 40))
        # Border
        pg.paint_outline(px, py, px + panel_w, py + panel_h, (60, 60, 120))

        # "OK" button (left half)
        btn_x: int = px + 6
        btn_y: int = py + 6
        btn_w: int = 58
        btn_h: int = 24
        pg.paint_rect(btn_x, btn_y, btn_x + btn_w, btn_y + btn_h, (40, 80, 40))
        pg.paint_outline(btn_x, btn_y, btn_x + btn_w, btn_y + btn_h, (80, 160, 80))
        # Text centered in button
        text_x: int = btn_x + (btn_w - 5 * 6) // 2
        text_y: int = btn_y + (btn_h - 7) // 2
        pg.paint_text(text_x, text_y, 'OK', (180, 255, 180))
        # Hit zone
        pg.register_zone(btn_x, btn_y, btn_x + btn_w, btn_y + btn_h,
                         on_click=lambda mx, my, action: self._pg_btn_ok())

        # "X" button (right half)
        btn2_x: int = px + panel_w - 6 - 58
        btn2_y: int = py + 6
        pg.paint_rect(btn2_x, btn2_y, btn2_x + btn_w, btn2_y + btn_h, (80, 30, 30))
        pg.paint_outline(btn2_x, btn2_y, btn2_x + btn_w, btn2_y + btn_h, (160, 60, 60))
        text2_x = btn2_x + (btn_w - 5 * 3) // 2
        pg.paint_text(text2_x, text_y, '✕', (255, 160, 160))
        pg.register_zone(btn2_x, btn2_y, btn2_x + btn_w, btn2_y + btn_h,
                         on_click=lambda mx, my, action: self._pg_btn_close())

        # ── Smiley face (left of panel) ──────────────────────────────
        smiley_x: int = px - 40
        smiley_y: int = py + 5
        smiley: List[List[int]] = [
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ]
        pg.paint_sprite(smiley_x, smiley_y, smiley, [(0, 0, 1), (255, 200, 50)])

        # ── Top-right status text ────────────────────────────────────
        pg.paint_text(sw - 120, 10, 'PixelGrid', (100, 100, 180))
        pg.paint_text(sw - 120, 20, 'G — скрыть', (80, 80, 140))

    def _pixel_anim_loop(self) -> None:
        """Animated pixel demo — доказывает что пиксели реально работают."""
        if not self._pixel_anim_active:
            return
        if not hasattr(self, 'pixel_grid'):
            return

        import math
        pg = self.pixel_grid
        sw, sh = pg.w, pg.h
        frame = self._pixel_anim_frame

        # Clear and redraw static UI
        pg.clear()
        self._draw_test_pixel_ui()

        # ── 1. Спиннер — вращающиеся точки в центре ──
        cx = sw // 2
        cy = sh // 2 - 40
        angle = frame * 0.06
        for i in range(14):
            a = angle + i * (2.0 * math.pi / 14)
            r = 35 + 12 * math.sin(frame * 0.1 + i * 0.9)
            px = int(cx + r * math.cos(a))
            py = int(cy + r * math.sin(a))
            # Цвет меняется по кругу
            ri = int(128 + 127 * math.sin(i * 0.5 + frame * 0.05))
            gi = int(128 + 127 * math.sin(i * 0.7 + frame * 0.07))
            bi = int(128 + 127 * math.sin(i * 0.3 + frame * 0.11))
            pg.paint(px, py, ri, gi, bi)

        # ── 2. Бегущая синусоида — волна пикселей ──
        t = frame * 0.12
        step = max(1, sw // 200)
        for x in range(0, sw, step):
            y = int(cy + 70 + 20 * math.sin(x * 0.04 + t))
            brightness = int(100 + 155 * (0.5 + 0.5 * math.sin(x * 0.03 + t)))
            pg.paint(x, y, brightness // 2, brightness, brightness)

        # ── 3. Счётчик кадров (живые цифры) ──
        counter = f'Frame: {frame}'
        pg.paint_text(sw - 200, sh - 130, counter, (0, 255, 200))

        # ── 4. Пульсирующая точка ──
        bx = int(cx + 100 * math.cos(t * 0.7))
        by = int(cy + 80 * math.sin(t * 0.5))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                pg.paint(bx + dx, by + dy, 255, 80, 80)

        # ── 5. Шкала загрузки ──
        bar_x = cx - 80
        bar_y = cy + 100
        bar_w = 160
        fill = int((frame % 100) * bar_w / 100)
        pg.paint_rect(bar_x, bar_y, bar_x + bar_w, bar_y + 8, (30, 30, 60))
        pg.paint_rect(bar_x, bar_y, bar_x + fill, bar_y + 8, (0, 200, 100))
        pg.paint_outline(bar_x, bar_y, bar_x + bar_w, bar_y + 8, (80, 80, 120))

        # ── 6. Agent rendering ────────────────────────────────────────
        if hasattr(self, 'agent_mgr') and hasattr(self, 'particle_mgr'):
            self.agent_mgr.render_all()
            self.particle_mgr.update_all()
            self.particle_mgr.render_all()

        self._pixel_anim_frame += 1
        self.root.after(66, self._pixel_anim_loop)

    def _pg_btn_ok(self) -> None:
        """Test PixelGrid OK button — flash a message."""
        self._show_mode_overlay('PixelGrid OK!')

    def _pg_btn_close(self) -> None:
        """Test PixelGrid close button — hide overlay."""
        self._toggle_pixel_grid()

    # ── Tray callbacks ─────────────────────────────────────────────────

    def quit_app(self) -> None:
        """Graceful shutdown: save config, stop tray, destroy window."""
        self.config['x'] = self.root.winfo_x()
        self.config['y'] = self.root.winfo_y()
        self.config['window_width'] = self.root.winfo_width()
        self.config['window_height'] = self.root.winfo_height()
        save_config(self.config)

        self.anim_running = False
        self.running = False

        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass

        try:
            self.pixel_win.close()
        except Exception:
            pass

        try:
            self.ai.close()
        except Exception:
            pass

        self.root.destroy()
        os._exit(0)

    def run(self) -> None:
        """Enter the Tkinter event loop."""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = CubeApp()
    app.run()
