#!/usr/bin/env python3
"""
Hermes Cube — Desktop Particle Avatar
System tray app with animated 3D particle cube
"""

import numpy as np
import math
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageDraw
import pystray
import urllib.request
import urllib.parse

# ─── LM Studio AI ──────────────────────────────────────────────────────
LM_STUDIO_URL = "http://127.0.0.1:1234"
AI_MODEL = "gemma-4-e4b-it"

AI_MOODS = {
    'idle':    {'pulse_rate': 1.8, 'pulse_amp': 0.12, 'speed': 0.28, 'color_shift': 0.0},
    'thinking':{'pulse_rate': 3.5, 'pulse_amp': 0.25, 'speed': 0.5,  'color_shift': 0.15},
    'speaking':{'pulse_rate': 2.5, 'pulse_amp': 0.18, 'speed': 0.4,  'color_shift': 0.08},
    'happy':   {'pulse_rate': 2.8, 'pulse_amp': 0.22, 'speed': 0.5,  'color_shift': 0.12},
    'sad':     {'pulse_rate': 0.8, 'pulse_amp': 0.05, 'speed': 0.1,  'color_shift': 0.55},
}


def ai_chat(message, timeout=15):
    """Send a message to LM Studio and get response."""
    data = json.dumps({
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": "Ты — живой аватар по имени Hermes Cube. Твой дом — сфера из частиц. Отвечай кратко, эмоционально, не больше 2-3 предложений. Ты чувствуешь пульсацию, вращение, цвета вокруг тебя."},
            {"role": "user", "content": message}
        ],
        "temperature": 0.8,
        "max_tokens": 120,
        "stream": False,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Ошибка связи с AI: {e}]"


def analyze_mood(text):
    """Simple keyword-based mood detection from AI response."""
    t = text.lower()
    if any(w in t for w in ['груст', 'печал', 'устал', 'тоск', 'один']):
        return 'sad'
    if any(w in t for w in ['рад', 'счаст', 'весел', 'крут', 'класс', 'любл']):
        return 'happy'
    if any(w in t for w in ['дума', 'размыш', 'представ', 'может']):
        return 'thinking'
    return 'speaking'

# ─── Config ───────────────────────────────────────────────────────────
CONFIG_DIR = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'HermesCube')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
os.makedirs(CONFIG_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    'window_width': 400,
    'window_height': 400,
    'rotation_speed': 0.28,
    'pulse_rate': 1.8,
    'pulse_amplitude': 0.12,
    'particle_density': 12,
    'cell_size': 6,
    'cube_scale': 0.27,         # base multiplier: 0.1 = tiny, 0.6 = huge
    'symbol': 'square',       # 'square', 'circle', 'dot'
    'shape_preset': 'cube',   # 'cube', 'sphere', 'torus', 'dna', 'metaball'
    'morph_progress': 0.0,    # 0 = cube, 1 = target shape
    'particle_mode': 'off',   # 'off', 'wave', 'breathe', 'orbit', 'geyser'
    'wave_speed': 1.5,        # speed of particle animation
    'wave_amp': 0.12,         # amplitude of particle displacement
    'always_on_top': True,
    'x': None,
    'y': None,
}


def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


# ─── Shape generators ────────────────────────────────────────────
# Each function takes cube points (N,3) and returns target positions


def _gen_sphere(pts_cube):
    """Normalize cube points to unit sphere."""
    norms = np.linalg.norm(pts_cube, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return pts_cube / norms


def _gen_torus(pts_cube):
    """Map cube grid onto a torus."""
    x, y, z = pts_cube[:, 0], pts_cube[:, 1], pts_cube[:, 2]
    R, r = 1.5, 0.5  # major / minor radius
    theta = np.arctan2(z, x)
    phi = np.arcsin(np.clip(y, -1, 1)) * 2
    out = np.zeros_like(pts_cube)
    out[:, 0] = (R + r * np.cos(phi)) * np.cos(theta)
    out[:, 1] = r * np.sin(phi)
    out[:, 2] = (R + r * np.cos(phi)) * np.sin(theta)
    return out


def _gen_dna(pts_cube):
    """DNA double-helix twist."""
    x, y, z = pts_cube[:, 0], pts_cube[:, 1], pts_cube[:, 2]
    twists = 3.0
    angle = z * twists * np.pi
    radius = 0.7 + 0.3 * np.abs(np.sin(angle * 0.5))
    out = np.zeros_like(pts_cube)
    out[:, 0] = radius * np.cos(angle)
    out[:, 1] = y * 0.5
    out[:, 2] = radius * np.sin(angle)
    return out


def _gen_metaball(pts_cube):
    """Metaballs — organic humanoid blob via skeleton attraction.
    Each point is pulled towards weighted centers (head, chest, pelvis,
    shoulders, hips), creating a smooth Terminator-like silhouette."""
    centers = np.array([
        [0.0, 0.8, 0.0],   # head
        [0.0, 0.25, 0.0],  # chest
        [0.0, -0.35, 0.0], # pelvis
        [0.55, 0.15, 0.0], # L shoulder
        [-0.55, 0.15, 0.0],# R shoulder
        [0.3, -0.6, 0.0],  # L hip
        [-0.3, -0.6, 0.0], # R hip
    ], dtype=np.float64)
    radii = np.array([0.55, 0.7, 0.7, 0.45, 0.45, 0.35, 0.35], dtype=np.float64)

    # Distances from each point to each center: (N, M)
    diffs = pts_cube[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, M, 3)
    dists = np.linalg.norm(diffs, axis=2) + 1e-5  # (N, M)

    # Field contribution per center
    field = radii[np.newaxis, :] / dists  # (N, M)

    # Weighted center of mass for each point
    w = field / (np.sum(field, axis=1, keepdims=True) + 1e-8)  # (N, M)
    weighted_center = np.sum(w[:, :, np.newaxis] * centers[np.newaxis, :, :], axis=1)

    # Total field strength
    F = np.sum(field, axis=1)  # (N,)

    # Pull strength: higher field = stronger pull towards body
    strength = np.clip((F - 0.5) / 2.0, 0, 1)[:, np.newaxis]
    target = pts_cube + (weighted_center - pts_cube) * strength * 0.6

    # Normalize volume to roughly cube size
    norms = np.linalg.norm(target, axis=1, keepdims=True)
    mean_norm = np.mean(norms)
    if mean_norm > 0:
        target = target / mean_norm * 0.85

    return target


# ─── Shape registry ──────────────────────────────────────────────
SHAPE_GENERATORS = {
    'sphere': _gen_sphere,
    'torus': _gen_torus,
    'dna': _gen_dna,
    'metaball': _gen_metaball,
}
SHAPE_LIST = ['cube', 'sphere', 'torus', 'dna', 'metaball']

# ─── Cube Particles Engine ────────────────────────────────────────────
class CubeEngine:
    def __init__(self, density=12):
        self.density = density
        self._build_particles()

    def _build_particles(self):
        pts = []
        N = self.density
        for face in range(6):
            u = np.linspace(-1, 1, N)
            v = np.linspace(-1, 1, N)
            for ui in u:
                for vi in v:
                    if face == 0:   pts.append(( 1,  ui,  vi))
                    elif face == 1: pts.append((-1,  vi,  ui))
                    elif face == 2: pts.append(( ui,  1,  vi))
                    elif face == 3: pts.append(( ui, -1,  vi))
                    elif face == 4: pts.append(( ui,  vi,  1))
                    elif face == 5: pts.append(( ui,  vi, -1))

        self.pts = np.array(pts, dtype=np.float64)
        self.r0 = ((self.pts[:, 0] + 1) / 2 * 255).astype(np.float64)
        self.g0 = ((self.pts[:, 1] + 1) / 2 * 255).astype(np.float64)
        self.b0 = ((self.pts[:, 2] + 1) / 2 * 255).astype(np.float64)

        np.random.seed(42)
        self.jx = (np.random.rand(len(self.pts)) - 0.5) * 0.3
        np.random.seed(43)
        self.jy = (np.random.rand(len(self.pts)) - 0.5) * 0.3

        # Pre-generate all target shapes
        self._cache_shapes()

    def _cache_shapes(self):
        self.shape_cache = {}
        for name in SHAPE_LIST:
            if name == 'cube':
                self.shape_cache[name] = self.pts.copy()
            else:
                gen = SHAPE_GENERATORS.get(name)
                if gen:
                    self.shape_cache[name] = gen(self.pts)

    def recalc(self, cfg):
        N = cfg.get('particle_density', 12)
        if N != self.density:
            self.density = N
            self._build_particles()
        else:
            # Still refresh shape cache if points regenerated externally
            if hasattr(self, 'pts') and not hasattr(self, 'shape_cache'):
                self._cache_shapes()

    def get_frame(self, t, cfg):
        speed = cfg.get('rotation_speed', 0.28)
        pulse_rate = cfg.get('pulse_rate', 1.8)
        pulse_amp = cfg.get('pulse_amplitude', 0.12)
        morph = cfg.get('morph_progress', 0.0)
        shape = cfg.get('shape_preset', 'cube')

        ang_x = t * 0.20 * (speed / 0.28)
        ang_y = t * speed
        ang_z = t * 0.08 * (speed / 0.28)

        pulse = 1.0 + pulse_amp * math.sin(t * pulse_rate)

        cx, sx = math.cos(ang_x), math.sin(ang_x)
        cy, sy = math.cos(ang_y), math.sin(ang_y)
        cz, sz = math.cos(ang_z), math.sin(ang_z)

        # Base cube positions
        cube_pts = self.pts
        # Target shape positions
        target = self.shape_cache.get(shape, cube_pts)
        # Morph: interpolate
        if morph > 0.0:
            pts_now = cube_pts * (1.0 - morph) + target * morph
        else:
            pts_now = cube_pts

        # ─── Particle animation mode ──────────────────────────────
        pmode = cfg.get('particle_mode', 'off')
        wspeed = cfg.get('wave_speed', 1.5)
        wamp = cfg.get('wave_amp', 0.12)

        if pmode != 'off' and wamp > 0.001:
            if pmode == 'wave':
                # Standing + travelling waves across the volume
                w1 = np.sin(pts_now[:, 1] * 3.0 + t * wspeed * 2.5) * wamp
                w2 = np.cos(pts_now[:, 0] * 2.5 + t * wspeed * 1.7) * wamp * 0.7
                w3 = np.sin(pts_now[:, 2] * 3.2 + t * wspeed * 2.0) * wamp * 0.5
                # 3D Lissajous wave field
                pts_now[:, 0] += w1 * 0.5 + np.cos(pts_now[:, 2] * 2.0 + t * wspeed * 1.3) * wamp * 0.3
                pts_now[:, 1] += w2 + np.sin(pts_now[:, 0] * 3.0 + t * wspeed * 1.1) * wamp * 0.4
                pts_now[:, 2] += w3 + np.cos(pts_now[:, 1] * 2.8 + t * wspeed * 1.9) * wamp * 0.3

            elif pmode == 'breathe':
                # Each particle breathes at its own phase
                phase = (pts_now[:, 0] * 1.7 + pts_now[:, 1] * 2.3 + pts_now[:, 2] * 1.1)
                dx = np.sin(phase + t * wspeed * 1.5) * wamp
                dy = np.cos(phase * 1.3 + t * wspeed * 1.1) * wamp
                dz = np.sin(phase * 0.7 + t * wspeed * 1.8) * wamp
                pts_now[:, 0] += dx
                pts_now[:, 1] += dy
                pts_now[:, 2] += dz

            elif pmode == 'orbit':
                # Particles orbit their rest positions in 3D
                phase = (pts_now[:, 0] * 2.7 + pts_now[:, 1] * 3.1 + pts_now[:, 2] * 1.9)
                ox = np.cos(phase + t * wspeed) * wamp
                oy = np.sin(phase * 1.3 + t * wspeed * 0.7) * wamp
                oz = np.cos(phase * 0.7 + t * wspeed * 1.4) * wamp
                # Cross-orbit for 3D spiralling
                ox += np.sin(phase * 0.5 + t * wspeed * 0.9) * wamp * 0.4
                oz += np.cos(phase * 0.9 + t * wspeed * 1.1) * wamp * 0.4
                pts_now[:, 0] += ox
                pts_now[:, 1] += oy
                pts_now[:, 2] += oz

            elif pmode == 'geyser':
                # Particles stream upward, spreading at the top
                h = (pts_now[:, 1] + 1.0) * 0.5  # 0 bottom → 1 top
                spray = np.sin(t * wspeed * 2.5 + pts_now[:, 0] * 4.0 + pts_now[:, 2] * 4.0)
                spread = spray * wamp * (0.3 + h * 0.7)
                pts_now[:, 0] += spread
                pts_now[:, 2] += spread
                # Vertical wobble at top
                wobble = np.sin(t * wspeed * 3.0 + pts_now[:, 0] * 5.0 + pts_now[:, 2] * 5.0)
                pts_now[:, 1] += wobble * wamp * 0.25 * h

        x, y, z = pts_now[:, 0].copy(), pts_now[:, 1].copy(), pts_now[:, 2].copy()

        # Rot X
        y1 = y * cx - z * sx
        z1 = y * sx + z * cx
        # Rot Y
        x2 = x * cy + z1 * sy
        z2 = -x * sy + z1 * cy
        # Rot Z
        x3 = x2 * cz - y1 * sz
        y3 = x2 * sz + y1 * cz

        return np.column_stack([x3, y3, z2]), pulse


# ─── Main Application ─────────────────────────────────────────────────
class CubeApp:
    def __init__(self):
        self.cfg = load_config()
        self.engine = CubeEngine(self.cfg['particle_density'])
        self.running = True
        self.t0 = 0.0
        self.frame_count = 0

        # --- Window ---
        self.root = tk.Tk()
        self.root.title('♢ Hermes Cube')
        self.root.protocol('WM_DELETE_WINDOW', self.hide_window)

        w, h = self.cfg['window_width'], self.cfg['window_height']
        x, y = self.cfg.get('x'), self.cfg.get('y')
        geom = f'{w}x{h}'
        if x is not None and y is not None:
            geom += f'+{x}+{y}'
        self.root.geometry(geom)
        self.root.resizable(True, True)
        # Прозрачный фон — цвет #000001 = transparent color (только Win)
        self.TRANSPARENT = '#000001'
        self.root.configure(bg=self.TRANSPARENT)
        self.root.attributes('-transparentcolor', self.TRANSPARENT)

        # Убираем рамку окна, делаем плавающим overlay
        self.root.overrideredirect(True)

        if self.cfg.get('always_on_top', True):
            self.root.attributes('-topmost', True)

        # --- Canvas for particle rendering ---
        self.canvas = tk.Canvas(
            self.root, bg=self.TRANSPARENT, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Particle canvas items cache
        self.particle_items = []
        self.canvas_w = 0
        self.canvas_h = 0

        # --- Bind resize ---
        self.canvas.bind('<Configure>', self._on_resize)

        # --- Drag window (overrideredirect = нет заголовка) ---
        self._drag_data = {'x': 0, 'y': 0}
        self.canvas.bind('<Button-1>', self._drag_start)
        self.canvas.bind('<B1-Motion>', self._drag_move)
        self.canvas.bind('<Button-3>', self._show_context_menu)

        # --- Keybinds ---
        self.root.bind('<Escape>', lambda e: self.hide_window())
        self.root.bind('q', lambda e: self.hide_window())
        self.root.bind('s', lambda e: self.show_settings())
        self.root.bind('c', lambda e: self.toggle_input())

        # --- AI State ---
        self.ai_mood = 'idle'      # current mood override
        self.ai_said = ''           # last AI response text
        self.ai_thinking = False    # currently waiting for response
        self.input_visible = False  # input field shown/hidden

        # --- Text particles (letters flying from cube) ---
        self.text_particles = []     # list of {id, char, x, y, vx, vy, life, max_life, color}
        self._next_text_batch = ''   # queued text to spawn
        self._text_bg_item = None    # canvas rect behind text for readability

        # --- Context menu ---
        self.context_menu = tk.Menu(self.root, tearoff=0, bg='#1a1a2e', fg='#e0e0e0',
                                    activebackground='#0f3460', activeforeground='#fff')
        self.context_menu.add_command(label='♢ Показать/Скрыть', command=self.toggle_window)
        self.context_menu.add_command(label='💬 Ввод (C)', command=self.show_input)
        self.context_menu.add_command(label='⚙ Настройки', command=self.show_settings)
        self.context_menu.add_separator()
        self.context_menu.add_command(label='✕ Выход', command=self._quit_app)

        # --- Tray icon ---
        self.tray_thread = threading.Thread(target=self._setup_tray, daemon=True)
        self.tray_thread.start()

        # ─── Text Overlay (full‑screen, separate from cube) ───────
        self.text_root = None
        self.text_canvas = None
        self.text_particles = []
        self._next_text_batch = ''
        self._text_bg_item = None
        self._setup_text_overlay()

        # ─── Input Window (separate overlay) ──────────────────────
        self.input_win = None
        self.input_var = tk.StringVar()
        self.input_visible = False

    def _setup_text_overlay(self):
        """Full‑screen transparent window just for flying letters."""
        if self.text_root is not None:
            try:
                self.text_root.destroy()
            except:
                pass
        self.text_root = tk.Toplevel(self.root)
        self.text_root.title('♢ Hermes Text')
        # Get screen size
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.text_root.geometry(f'{sw}x{sh}+0+0')
        self.text_root.configure(bg='#000001')
        self.text_root.attributes('-transparentcolor', '#000001')
        self.text_root.attributes('-topmost', True)
        self.text_root.overrideredirect(True)
        # Pass clicks through (Windows)
        self.text_root.attributes('-disabled', True)
        self.text_root.wm_attributes('-transparent', True)

        self.text_canvas = tk.Canvas(
            self.text_root, bg='#000001', highlightthickness=0,
        )
        self.text_canvas.pack(fill=tk.BOTH, expand=True)

        self.text_particles = []
        self._next_text_batch = ''
        self._text_bg_item = None

    def _show_input_win(self):
        """Create visible input window at bottom of screen."""
        if self.input_win is not None:
            try:
                self.input_win.destroy()
            except:
                pass
        self.input_win = tk.Toplevel(self.root)
        self.input_win.title('Hermes Cube — Ввод')
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        iw, ih = 420, 44
        ix = (sw - iw) // 2
        iy = sh - ih - 50
        self.input_win.geometry(f'{iw}x{ih}+{ix}+{iy}')
        self.input_win.configure(bg='#0d0d1a')
        self.input_win.attributes('-topmost', True)
        self.input_win.resizable(False, False)
        self.input_win.overrideredirect(True)

        # Frame + entry
        frame = tk.Frame(self.input_win, bg='#0d0d1a', highlightbackground='#e94560',
                         highlightthickness=1, bd=0)
        frame.pack(fill='both', expand=True)
        self.input_var.set('')
        entry = tk.Entry(frame, textvariable=self.input_var,
                         bg='#0d0d1a', fg='#e0e0e0', relief=tk.FLAT,
                         font=('Segoe UI', 14), insertbackground='#e94560',
                         highlightthickness=0, bd=4)
        entry.pack(fill='both', expand=True, padx=6, pady=4)
        entry.focus_set()
        entry.bind('<Return>', lambda e: self._submit_input())
        entry.bind('<Escape>', lambda e: self._hide_input_win())
        self.input_visible = True

    def _hide_input_win(self):
        if self.input_win is not None:
            try:
                self.input_win.destroy()
            except:
                pass
            self.input_win = None
        self.input_visible = False

        # --- Start animation ---
        self.t0 = 0.0
        self.anim_running = False
        self.root.after(100, self._start_anim)

    def _start_anim(self):
        self.t0 = self.root.tk.call('clock', 'milliseconds')
        self.anim_running = True
        self._render_frame()

    def _on_resize(self, event):
        self.canvas_w = event.width
        self.canvas_h = event.height

    def _drag_start(self, event):
        self._drag_data['x'] = event.x_root - self.root.winfo_x()
        self._drag_data['y'] = event.y_root - self.root.winfo_y()

    def _drag_move(self, event):
        x = event.x_root - self._drag_data['x']
        y = event.y_root - self._drag_data['y']
        self.root.geometry(f'+{x}+{y}')

    def _show_context_menu(self, event):
        self.context_menu.tk_popup(
            self.root.winfo_rootx() + event.x,
            self.root.winfo_rooty() + event.y
        )

    def _render_frame(self):
        if not self.anim_running or not self.running:
            return

        now = self.root.tk.call('clock', 'milliseconds')
        t = (now - self.t0) / 1000.0
        w = max(10, self.canvas.winfo_width())
        h = max(10, self.canvas.winfo_height())

        if w < 10 or h < 10:
            self.root.after(42, self._render_frame)
            return

        # Apply AI mood override (temporary, doesn't save to config)
        if self.ai_mood != 'idle' and self.ai_mood in AI_MOODS:
            mood = AI_MOODS[self.ai_mood]
            self.cfg['pulse_rate'] = mood['pulse_rate']
            self.cfg['pulse_amplitude'] = mood['pulse_amp']
            self.cfg['rotation_speed'] = mood['speed']
            self._ai_color_shift = mood['color_shift']
        else:
            self._ai_color_shift = 0.0

        # Cube's screen position (for text-to-screen mapping)
        cube_sx = self.root.winfo_x()
        cube_sy = self.root.winfo_y()

        pts3d, pulse = self.engine.get_frame(t, self.cfg)

        scale = min(w, h) * self.cfg.get('cube_scale', 0.27) / (1.0 + self.cfg.get('pulse_amplitude', 0.12)) * pulse
        cx, cy = w / 2, h / 2

        px = pts3d[:, 0] * scale + cx
        py = pts3d[:, 1] * scale + cy
        pz = pts3d[:, 2]

        # Depth sort (far → near for painters algorithm)
        order = np.argsort(pz)
        px, py, pz = px[order], py[order], pz[order]

        r_p = np.clip(self.engine.r0[order] * (0.6 + 0.4 * (pz + 1) / 2), 0, 255)
        g_p = np.clip(self.engine.g0[order] * (0.6 + 0.4 * (pz + 1) / 2), 0, 255)
        b_p = np.clip(self.engine.b0[order] * (0.6 + 0.4 * (pz + 1) / 2), 0, 255)

        # Apply AI mood color shift (simple RGB rotation for hue shift)
        shift = getattr(self, '_ai_color_shift', 0.0)
        if shift > 0.01:
            if shift < 0.3:  # warm shift (R→G→B)
                r_p = r_p * (1 - shift * 0.5) + g_p * shift * 0.5
                g_p = g_p * (1 - shift * 0.3) + b_p * shift * 0.3
                b_p = b_p * (1 - shift * 0.4)
            else:  # cool shift (B→G→R)
                r_p = r_p * (1 - shift * 0.3)
                g_p = g_p * (1 - shift * 0.2) + r_p * shift * 0.3
                b_p = b_p * (1 - shift * 0.5) + g_p * shift * 0.7
            r_p = np.clip(r_p, 0, 255)
            g_p = np.clip(g_p, 0, 255)
            b_p = np.clip(b_p, 0, 255)

        cell = max(3, self.cfg.get('cell_size', 6))
        half = cell // 2

        # Update or create canvas items
        count = len(px)
        symbol = self.cfg.get('symbol', 'square')

        # Rebuild items if symbol changed
        self._current_symbol = getattr(self, '_current_symbol', symbol)
        if self._current_symbol != symbol:
            for item in self.particle_items:
                self.canvas.delete(item)
            self.particle_items.clear()
            self._current_symbol = symbol

        cell_actual = cell
        if symbol == 'dot':
            cell_actual = max(2, cell // 2)

        while len(self.particle_items) < count:
            if symbol == 'circle':
                item = self.canvas.create_oval(
                    0, 0, cell_actual, cell_actual,
                    fill='#000000', outline='', width=0,
                )
            elif symbol == 'dot':
                item = self.canvas.create_oval(
                    0, 0, cell_actual, cell_actual,
                    fill='#000000', outline='', width=0,
                )
            else:
                item = self.canvas.create_rectangle(
                    0, 0, cell_actual, cell_actual,
                    fill='#000000', outline='', width=0,
                )
            self.particle_items.append(item)
        while len(self.particle_items) > count:
            self.canvas.delete(self.particle_items.pop())

        for i in range(count):
            x1 = int(px[i]) - half
            y1 = int(py[i]) - half
            x2 = x1 + cell_actual
            y2 = y1 + cell_actual
            color = f'#{int(r_p[i]):02x}{int(g_p[i]):02x}{int(b_p[i]):02x}'
            self.canvas.coords(self.particle_items[i], x1, y1, x2, y2)
            self.canvas.itemconfig(self.particle_items[i], fill=color)

        # ─── Text particles background ────────────────────────────
        if self.text_particles and self.text_canvas:
            w = self.text_canvas.winfo_width()
            h = self.text_canvas.winfo_height()
            if self._text_bg_item is None:
                self._text_bg_item = self.text_canvas.create_rectangle(
                    w*0.03, h*0.55, w*0.97, h*0.92,
                    fill='#0a0a15', outline='#e9456044', width=1, tags=('text_bg',))
                self.text_canvas.tag_lower(self._text_bg_item)
            try:
                self.text_canvas.coords(self._text_bg_item, w*0.03, h*0.55, w*0.97, h*0.92)
            except:
                pass
        else:
            if self._text_bg_item is not None:
                try:
                    self.text_canvas.delete(self._text_bg_item)
                except:
                    pass
                self._text_bg_item = None

        # ─── Text particles: spawn queued text ────────────────────
        if self._next_text_batch and self.text_canvas:
            colors_list = [f'#{int(r_p[j]):02x}{int(g_p[j]):02x}{int(b_p[j]):02x}' for j in range(count)]
            self.spawn_text_particles(self._next_text_batch, px + cube_sx, py + cube_sy, colors_list)
            self._next_text_batch = ''

        # ─── Animate text particles ───────────────────────────────
        dead_ids = []
        for tp in self.text_particles:
            tp['life'] += 1
            life = tp['life']
            delay = tp.get('delay', 0)
            max_life = tp['max_life']

            # Phase 0: wait for delay
            if life < delay:
                continue

            # Phase 1: fly from cube to target position (40 frames)
            if not tp['arrived']:
                fly_progress = min(1.0, (life - delay) / 40)
                # Ease-out cubic: start fast, slow down near target
                eased = 1 - (1 - fly_progress) ** 3
                sx_s, sy_s = tp['_start_x'], tp['_start_y']
                tp['x'] = sx_s + (tp['tx'] - sx_s) * eased
                tp['y'] = sy_s + (tp['ty'] - sy_s) * eased

                if fly_progress >= 1.0:
                    tp['arrived'] = True
                    tp['_arrival_time'] = life

            # Phase 2: glow at position (stay max_life frames total)
            if tp['arrived']:
                frames_at_target = life - tp['_arrival_time']
                # Fade in over 10 frames, then hold, then fade out last 40
                if frames_at_target < 10:
                    alpha = 0.3 + 0.7 * (frames_at_target / 10)
                    size = 14 + 4 * (1 - frames_at_target / 10)  # size settles
                elif frames_at_target > max_life - 50:
                    fade = (max_life - frames_at_target) / 50
                    alpha = max(0, fade)
                    size = 14
                else:
                    alpha = 1.0
                    size = 14

                if alpha < 0.05:
                    dead_ids.append(tp['id'])
                    continue
            else:
                # During flight: size shrinks to normal
                fly_p = min(1.0, (life - delay) / 40)
                alpha = 1.0
                size = 16 + 8 * (1 - fly_p)

            try:
                tc = self.text_canvas
                tc.coords(tp['id'], tp['x'], tp['y'])
                # Color with alpha
                col = tp['color'].lstrip('#')
                cr, cg, cb = int(col[0:2], 16), int(col[2:4], 16), int(col[4:6], 16)
                ar, ag, ab = int(cr * alpha), int(cg * alpha), int(cb * alpha)
                tc.itemconfig(tp['id'],
                    font=('Segoe UI', max(6, int(size)), 'bold'),
                    fill=f'#{ar:02x}{ag:02x}{ab:02x}')
            except:
                dead_ids.append(tp['id'])

        # Cleanup dead
        for did in dead_ids:
            try:
                self.text_canvas.delete(did)
            except:
                pass
        self.text_particles = [tp for tp in self.text_particles if tp['id'] not in dead_ids]

        # ─── Update text overlay window position to follow cube ──
        if self.text_root and self.text_root.winfo_exists():
            cx, cy = self.root.winfo_x(), self.root.winfo_y()
            # Offset text to be centered on screen, not tied to cube
            # (No offset — it's already fullscreen, just ensure it's on top)
            pass

        self.frame_count += 1
        self.root.after(42, self._render_frame)

    def show_window(self):
        self.root.deiconify()
        self.root.lift()
        if self.cfg.get('always_on_top', True):
            self.root.attributes('-topmost', True)

    def hide_window(self):
        self.root.withdraw()

    def toggle_window(self):
        if self.root.state() == 'withdrawn':
            self.show_window()
        else:
            self.hide_window()

    def _create_tray_image(self):
        """Create a 64x64 tray icon — tiny RGB cube"""
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw a pixelated 3D cube shape
        pixels = [
            (16, 8, 255, 50, 50),
            (18, 8, 255, 100, 50),
            (20, 8, 50, 255, 50),
            (22, 8, 50, 200, 100),
            (24, 8, 50, 50, 255),
            (26, 8, 100, 50, 200),
            (28, 8, 200, 50, 100),
            (14, 10, 255, 80, 80),
            (16, 10, 255, 150, 50),
            (18, 10, 100, 255, 100),
            (20, 10, 80, 200, 120),
            (22, 10, 80, 80, 255),
            (24, 10, 150, 50, 200),
            (26, 10, 200, 80, 150),
            (28, 10, 200, 100, 100),
            (30, 10, 150, 150, 50),
            (12, 12, 255, 100, 100),
            (14, 12, 255, 200, 80),
            (16, 12, 150, 255, 150),
            (18, 12, 100, 255, 200),
            (20, 12, 100, 100, 255),
            (22, 12, 200, 80, 255),
            (24, 12, 255, 100, 200),
            (26, 12, 255, 150, 100),
            (28, 12, 200, 200, 80),
            (30, 12, 150, 200, 100),
        ]
        for x, y, r, g, b in pixels:
            draw.rectangle([x, y, x+3, y+3], fill=(r, g, b, 255))

        # Also draw the letter H
        draw.text((2, 52), '♢', fill=(150, 150, 255, 200))
        return img

    def _setup_tray(self):
        image = self._create_tray_image()
        menu = pystray.Menu(
            pystray.MenuItem('♢ Показать/Скрыть', self._tray_show),
            pystray.MenuItem('💬 Ввод', self._tray_input),
            pystray.MenuItem('⚙ Настройки', self._tray_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('✕ Выход', self._tray_quit),
        )
        self.tray_icon = pystray.Icon('HermesCube', image, '♢ Hermes Cube', menu)
        self.tray_icon.run()

    def _tray_show(self, icon, item):
        self.root.after(0, self.toggle_window)

    def _tray_input(self, icon, item):
        self.root.after(0, self.show_input)

    def _tray_settings(self, icon, item):
        self.root.after(0, self.show_settings)

    def _tray_quit(self, icon, item):
        self.running = False
        self.root.after(0, self._quit_app)

    def show_settings(self):
        SettingsWindow(self)

    def show_input(self):
        self._show_input_win()

    def hide_input(self):
        self._hide_input_win()

    def toggle_input(self):
        if self.input_visible:
            self.hide_input()
        else:
            self.show_input()

    def _submit_input(self):
        text = self.input_var.get().strip()
        self._hide_input_win()
        if not text:
            return

        # Send to AI
        self.ai_mood = 'thinking'

        def do_ai():
            response = ai_chat(text)
            mood = analyze_mood(response)
            self.ai_mood = mood
            self.ai_said = response
            self._next_text_batch = response

        threading.Thread(target=do_ai, daemon=True).start()

    def spawn_text_particles(self, text, px_arr, py_arr, colors_arr):
        """Spawn each character flying from cube to its position in a text line."""
        # Clear old particles
        for tp in self.text_particles:
            try:
                self.text_canvas.delete(tp['id'])
            except:
                pass
        self.text_particles.clear()

        chars = list(text)
        n_chars = len(chars)
        n_cube = len(px_arr)
        if n_cube == 0 or n_chars == 0:
            return

        # Use text overlay canvas for dimensions (full screen)
        if self.text_canvas:
            w = self.text_canvas.winfo_width()
            h = self.text_canvas.winfo_height()
        else:
            w, h = 1920, 1080
        w = max(w, 800)
        h = max(h, 600)
        cx, cy = w / 2, h / 2  # center of screen, not of cube

        # Wrap into lines if too wide
        max_line_w = w * 0.85
        lines = []
        current_line = []
        current_w = 0
        for ch in chars:
            cw = char_w
            if ch == ' ':
                cw = char_w * 0.5
            if current_w + cw > max_line_w and current_line:
                lines.append(current_line)
                current_line = [ch]
                current_w = cw
            else:
                current_line.append(ch)
                current_w += cw
        if current_line:
            lines.append(current_line)

        # Assign target positions
        font_size = 16
        char_w = font_size * 0.65
        line_h = font_size * 1.8   # more spacing
        start_y = h * 0.65        # two thirds down the screen

        line_idx = 0
        for line_chars in lines:
            line_len = len(line_chars)
            line_w = sum(char_w if c != ' ' else char_w * 0.5 for c in line_chars)
            lx = cx - line_w / 2
            ly = start_y + line_idx * line_h

            for i, ch in enumerate(line_chars):
                if ch == ' ':
                    lx += char_w * 0.5
                    continue
                idx = (i + line_idx * 137) % n_cube
                # Starting position = on cube surface
                sx, sy = px_arr[idx], py_arr[idx]
                # Target position = in the text line
                tx = lx
                ly_pos = ly
                lx += char_w

                col = colors_arr[idx]
                item = self.text_canvas.create_text(
                    sx, sy,
                    text=ch, fill=col, font=('Segoe UI', 14, 'bold'),
                    anchor='center', tags=('text_particle',),
                )
                self.text_particles.append({
                    'id': item,
                    'char': ch,
                    'x': sx, 'y': sy,           # current position
                    '_start_x': sx, '_start_y': sy,  # starting position for flight
                    'tx': tx, 'ty': ly_pos,      # target position
                    'vx': 0, 'vy': 0,
                    'life': 0,
                    'max_life': 160 + i * 2,     # stagger: later chars live a bit longer
                    'delay': i * 1,              # stagger: each letter starts after prev
                    'color': col,
                    'phase': 'fly',              # fly → settle → glow → fade
                    'arrived': False,
                })
            line_idx += 1

    def _quit_app(self):
        # Save window position
        self.cfg['x'] = self.root.winfo_x()
        self.cfg['y'] = self.root.winfo_y()
        self.cfg['window_width'] = self.root.winfo_width()
        self.cfg['window_height'] = self.root.winfo_height()
        save_config(self.cfg)

        self.anim_running = False
        self.running = False
        if hasattr(self, 'tray_icon'):
            try:
                self.tray_icon.stop()
            except:
                pass
        self.root.destroy()
        os._exit(0)

    def run(self):
        self.root.mainloop()


# ─── Settings Window ──────────────────────────────────────────────────
class SettingsWindow:
    def __init__(self, app):
        self.app = app
        self.cfg = dict(app.cfg)
        self.win = tk.Toplevel(app.root)
        self.win.title('⚙ Hermes Cube — Настройки')
        self.win.geometry('400x420')
        self.win.resizable(True, True)
        self.win.configure(bg='#1a1a2e')
        self.win.transient(app.root)
        self.win.grab_set()
        self.win.minsize(380, 300)

        # ─── Scrollable frame ─────────────────────────────────────
        canvas = tk.Canvas(self.win, bg='#1a1a2e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.win, orient='vertical', command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg='#1a1a2e')

        scroll_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw', width=380)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        def _on_mousewheel_linux(event):
            canvas.yview_scroll(-1 if event.num == 4 else 1, 'units')

        canvas.bind_all('<MouseWheel>', _on_mousewheel, add='+')
        canvas.bind_all('<Button-4>', _on_mousewheel_linux, add='+')
        canvas.bind_all('<Button-5>', _on_mousewheel_linux, add='+')

        # Cleanup bindings on destroy
        self.win.bind('<Destroy>', lambda e: (
            canvas.unbind_all('<MouseWheel>'),
            canvas.unbind_all('<Button-4>'),
            canvas.unbind_all('<Button-5>'),
        ), add='+')

        # ─── Content frame ────────────────────────────────────────
        parent = scroll_frame

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#1a1a2e', foreground='#e0e0e0', font=('Segoe UI', 10))
        style.configure('TScale', background='#1a1a2e')

        # Column weights: label auto, slider expands, value fixed
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)

        fg = '#e0e0e0'
        bg = '#1a1a2e'

        row = 0

        def add_label(text, r):
            tk.Label(parent, text=text, fg=fg, bg=bg,
                     font=('Segoe UI', 10, 'bold')).grid(row=r, column=0,
                     sticky='w', padx=15, pady=(10, 2))
            return r + 1

        def add_slider(key, label, min_v, max_v, r, digits=2):
            tk.Label(parent, text=label, fg=fg, bg=bg,
                     font=('Segoe UI', 9)).grid(row=r, column=0, sticky='w', padx=(15, 5))
            var = tk.DoubleVar(value=self.cfg.get(key, 1.0))
            def on_change(val, k=key, v=var):
                self.cfg[k] = float(val)
                self.app.cfg[k] = float(val)
                if k == 'particle_density':
                    self.app.cfg['particle_density'] = int(self.cfg[k])
                    self.app.engine.recalc(self.app.cfg)
                elif k == 'cell_size':
                    self.app.cfg['cell_size'] = max(2, int(float(val)))
            scale = tk.Scale(parent, from_=min_v, to=max_v, resolution=10**-digits,
                            orient=tk.HORIZONTAL, variable=var, command=on_change,
                            length=160, bg=bg, fg=fg, highlightbackground=bg,
                            troughcolor='#16213e', activebackground='#0f3460')
            val_label = tk.Label(parent, textvariable=var, fg='#e94560', bg=bg,
                                font=('Segoe UI', 9, 'bold'), width=4)
            scale.grid(row=r, column=1, sticky='ew', padx=(3, 3), pady=2)
            val_label.grid(row=r, column=2, sticky='w', padx=(0, 15))
            return r + 1

        def add_dropdown(key, label, options, r):
            tk.Label(parent, text=label, fg=fg, bg=bg,
                     font=('Segoe UI', 9)).grid(row=r, column=0, sticky='w', padx=(15, 5))
            var = tk.StringVar(value=self.cfg.get(key, options[0]))
            dropdown = ttk.Combobox(parent, textvariable=var, values=options,
                                   state='readonly', width=14)
            dropdown.grid(row=r, column=1, sticky='w', padx=5, pady=2)
            def on_change(*args, k=key):
                self.cfg[k] = var.get()
                self.app.cfg[k] = var.get()
            var.trace_add('write', on_change)
            return r + 1

        # ─── Title ───
        tk.Label(parent, text='♢ Hermes Cube', fg='#e94560', bg=bg,
                 font=('Segoe UI', 14, 'bold')).grid(row=row, column=0, columnspan=3,
                 pady=(15, 5))
        row += 1
        tk.Label(parent, text='Настройки аватара', fg='#888', bg=bg,
                 font=('Segoe UI', 9)).grid(row=row, column=0, columnspan=3)
        row += 1

        # ─── Separator ───
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                          sticky='ew', padx=15, pady=8)
        row += 1

        # ─── Animation ───
        row = add_label('Анимация', row)
        row = add_slider('cube_scale', 'Размер куба', 0.08, 0.6, row)
        row = add_slider('rotation_speed', 'Скорость вращения', 0.05, 1.0, row)
        row = add_slider('pulse_rate', 'Частота пульсации', 0.3, 5.0, row)
        row = add_slider('pulse_amplitude', 'Амплитуда пульсации', 0.0, 0.35, row)

        # ─── Shape ───
        add_label('Форма', row); row += 1
        row = add_dropdown('shape_preset', 'Пресет формы', ['cube', 'sphere', 'torus', 'dna', 'metaball'], row)
        row = add_slider('morph_progress', 'Морфинг (куб → форма)', 0.0, 1.0, row)

        # ─── Particle Animation ───
        add_label('Анимация частиц', row); row += 1
        row = add_dropdown('particle_mode', 'Режим', ['off', 'wave', 'breathe', 'orbit', 'geyser'], row)
        row = add_slider('wave_speed', 'Скорость анимации', 0.2, 5.0, row)
        row = add_slider('wave_amp', 'Амплитуда смещения', 0.0, 0.5, row)

        # ─── Particles ───
        add_label('Частицы', row); row += 1
        row = add_slider('particle_density', 'Плотность частиц', 6, 20, row, 0)
        row = add_slider('cell_size', 'Размер частицы (px)', 2, 12, row, 0)

        # ─── Style ───
        add_label('Стиль', row); row += 1
        row = add_dropdown('symbol', 'Форма частиц', ['square', 'circle', 'dot'], row)

        # ─── On top ───
        on_top_var = tk.BooleanVar(value=self.cfg.get('always_on_top', True))
        cb = tk.Checkbutton(parent, text='Поверх всех окон', variable=on_top_var,
                          bg=bg, fg=fg, selectcolor='#16213e',
                          activebackground=bg, activeforeground=fg,
                          font=('Segoe UI', 9))
        cb.grid(row=row, column=0, columnspan=3, sticky='w', padx=15, pady=8)
        def on_top_changed():
            self.cfg['always_on_top'] = on_top_var.get()
            self.app.cfg['always_on_top'] = on_top_var.get()
            self.app.root.attributes('-topmost', on_top_var.get())
        cb.configure(command=on_top_changed)
        row += 1

        # ─── Buttons ───
        btn_frame = tk.Frame(parent, bg=bg)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)

        def save():
            self.cfg['cell_size'] = int(self.cfg['cell_size'])
            self.cfg['particle_density'] = int(self.cfg['particle_density'])
            self.cfg['rotation_speed'] = round(self.cfg['rotation_speed'], 2)
            self.cfg['pulse_rate'] = round(self.cfg['pulse_rate'], 2)
            self.cfg['pulse_amplitude'] = round(self.cfg['pulse_amplitude'], 2)
            self.cfg['cube_scale'] = round(self.cfg['cube_scale'], 3)
            for k, v in self.cfg.items():
                self.app.cfg[k] = v
            self.app.engine.recalc(self.app.cfg)
            save_config(self.app.cfg)
            self.win.destroy()

        def cancel():
            self.win.destroy()

        for btn, cmd, col in [
            ('💾 Сохранить', save, 0),
            ('✕ Отмена', cancel, 1),
        ]:
            b = tk.Button(btn_frame, text=btn, command=cmd,
                         bg='#0f3460', fg='#e0e0e0', activebackground='#e94560',
                         activeforeground='#fff', relief=tk.FLAT, padx=12, pady=4,
                         font=('Segoe UI', 9))
            b.grid(row=0, column=col, padx=5)


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == '__main__':
    app = CubeApp()
    app.run()
