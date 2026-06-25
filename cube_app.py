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

# Windows-specific: принудительный показ окна через Win32 API
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    # Устанавливаем типы аргументов для Win32-функций
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetWindowLongW.restype = ctypes.c_long
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SetWindowPos.restype = ctypes.c_bool
    user32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, ctypes.c_uint32, ctypes.c_byte, ctypes.c_uint32]
    user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
    user32.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    user32.MoveWindow.restype = ctypes.c_bool
    user32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_bool]
    user32.InvalidateRect.restype = ctypes.c_bool
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.UpdateWindow.restype = ctypes.c_bool
else:
    user32 = None
try:
    import pystray
except ImportError:
    pystray = None

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
SHAPE_GENERATORS = {}

def _register_shape(name):
    def decorator(fn):
        SHAPE_GENERATORS[name] = fn
        return fn
    return decorator


def _gen_cube(pts_cube):
    """Identity — points stay as cube."""
    return pts_cube.copy()


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
    """Metaball-like organic blobs."""
    x, y, z = pts_cube[:, 0], pts_cube[:, 1], pts_cube[:, 2]
    # Multiple attractor points create organic shapes
    attractors = [
        (0.5, 0.5, 0.5, 0.6),
        (-0.5, -0.5, 0.5, 0.6),
        (0.5, -0.5, -0.5, 0.6),
        (-0.5, 0.5, -0.5, 0.6),
    ]
    field = np.zeros(len(pts_cube))
    for ax, ay, az, strength in attractors:
        dist = np.sqrt((x - ax)**2 + (y - ay)**2 + (z - az)**2)
        field += strength / (dist + 0.1)
    # Normalize field and push points toward isosurface
    field = field / np.max(field)
    scale = 0.6 + 0.4 * field
    out = pts_cube.copy()
    out[:, 0] *= scale
    out[:, 1] *= scale
    out[:, 2] *= scale
    return out


# ─── Register all shapes ─────────────────────────────────────────
SHAPE_LIST = ['cube', 'sphere', 'torus', 'dna', 'metaball']


# ─── Cube Particles Engine ────────────────────────────────────────────
class CubeEngine:
    def __init__(self, density=12):
        self.density = density
        self._build_particles()
        self._cache_shapes()

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

        self.TRANSPARENT = '#000001'

        # Точная копия рабочего теста — geometry + overrideredirect + transparent
        pos_x = x if x is not None else 100
        pos_y = y if y is not None else 100
        geom = f'{w}x{h}+{pos_x}+{pos_y}'
        self.root.geometry(geom)
        self.root.resizable(True, True)
        self.root.configure(bg=self.TRANSPARENT)
        self.root.overrideredirect(True)
        self.root.attributes('-transparentcolor', self.TRANSPARENT)
        if self.cfg.get('always_on_top', True):
            self.root.attributes('-topmost', True)

        # --- Canvas for particle rendering ---
        self.canvas = tk.Canvas(
            self.root, bg=self.TRANSPARENT, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Единственный update — он применяет геометрию и маппит окно
        self.root.update()

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
        # Double-click to show settings
        self.canvas.bind('<Double-Button-1>', lambda e: self.show_settings())
        # Show hint text on startup
        self._show_hint = True

        # --- Keybinds ---
        self.root.bind('<Escape>', lambda e: self.hide_window())
        self.root.bind('q', lambda e: self.hide_window())
        self.root.bind('s', lambda e: self.show_settings())
        self.root.bind('h', lambda e: self.hide_window())

        # --- Context menu ---
        self.context_menu = tk.Menu(self.root, tearoff=0, bg='#1a1a2e', fg='#e0e0e0',
                                    activebackground='#0f3460', activeforeground='#fff')
        self.context_menu.add_command(label='♢ Показать/Скрыть', command=self.toggle_window)
        self.context_menu.add_command(label='⚙ Настройки', command=self.show_settings)
        self.context_menu.add_separator()
        self.context_menu.add_command(label='✕ Выход', command=self._quit_app)

        # --- Tray icon ---
        self.tray_icon = None
        threading.Thread(target=self._setup_tray, daemon=False).start()

        # --- Start animation ---
        self.t0 = 0.0
        self.anim_running = False
        self.root.after(100, self._start_anim)

    def _start_anim(self):
        self.t0 = self.root.tk.call('clock', 'milliseconds')
        self.anim_running = True
        # Убедимся что окно показано
        self.root.lift()
        self.root.lift()
        self.root.lift()
        self.root.update()
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

        cell = max(3, self.cfg.get('cell_size', 6))
        half = cell // 2

        # Update or create canvas items
        count = len(px)
        symbol = self.cfg.get('symbol', 'square')

        cell_actual = cell
        half_actual = half
        if symbol == 'dot':
            cell_actual = max(2, cell // 2)
            half_actual = cell_actual // 2

        # Rebuild items if symbol changed
        self._current_symbol = getattr(self, '_current_symbol', symbol)
        if self._current_symbol != symbol:
            for item in self.particle_items:
                self.canvas.delete(item)
            self.particle_items.clear()
            self._current_symbol = symbol

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
            x1 = int(px[i]) - half_actual
            y1 = int(py[i]) - half_actual
            x2 = x1 + cell_actual
            y2 = y1 + cell_actual
            color = f'#{int(r_p[i]):02x}{int(g_p[i]):02x}{int(b_p[i]):02x}'
            self.canvas.coords(self.particle_items[i], x1, y1, x2, y2)
            self.canvas.itemconfig(self.particle_items[i], fill=color)

        # Show startup hint for 5 seconds
        if self._show_hint:
            hint = self.canvas.create_text(
                w//2, h - 20,
                text='Нажми S — настройки | H — скрыть | Двойной клик — меню',
                fill='#e94560', font=('Segoe UI', 9),
                anchor='center'
            )
            self.root.after(5000, lambda: (
                self.canvas.delete(hint) if self.canvas.winfo_exists() else None
            ))
            self._show_hint = False

        self.frame_count += 1
        self.root.after(42, self._render_frame)

    def show_window(self):
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.lift()
        self.root.lift()
        self.root.lift()
        if self.cfg.get('always_on_top', True):
            self.root.attributes('-topmost', True)
        self.root.update()

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
        if pystray is None:
            return  # pystray not installed, skip tray icon
        image = self._create_tray_image()
        menu = pystray.Menu(
            pystray.MenuItem('♢ Показать/Скрыть', self._tray_show),
            pystray.MenuItem('⚙ Настройки', self._tray_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('✕ Выход', self._tray_quit),
        )
        self.tray_icon = pystray.Icon('HermesCube', image, '♢ Hermes Cube', menu)
        # Try run_detached (pystray >= 0.19.3), fallback to run() in thread
        if hasattr(self.tray_icon, 'run_detached'):
            self.tray_icon.run_detached()
        else:
            # Remove daemon so icon persists
            self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=False)
            self.tray_thread.start()

    def _tray_show(self, icon, item):
        self.root.after(0, self.toggle_window)

    def _tray_settings(self, icon, item):
        self.root.after(0, self.show_settings)

    def _tray_quit(self, icon, item):
        self.running = False
        self.root.after(0, self._quit_app)

    def show_settings(self):
        SettingsWindow(self)

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
        self.win.geometry('420x500')
        self.win.resizable(True, True)
        self.win.configure(bg='#1a1a2e')
        self.win.transient(app.root)
        self.win.grab_set()

        # Column weights: label auto, slider expands, value fixed
        self.win.columnconfigure(0, weight=0)
        self.win.columnconfigure(1, weight=1)
        self.win.columnconfigure(2, weight=0)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#1a1a2e', foreground='#e0e0e0', font=('Segoe UI', 10))
        style.configure('TScale', background='#1a1a2e')

        fg = '#e0e0e0'
        bg = '#1a1a2e'
        entry_bg = '#16213e'

        row = 0

        def add_label(text, r):
            tk.Label(self.win, text=text, fg=fg, bg=bg,
                     font=('Segoe UI', 10, 'bold')).grid(row=r, column=0,
                     sticky='w', padx=15, pady=(10, 2))
            return r + 1

        def add_slider(key, label, min_v, max_v, r, digits=2):
            tk.Label(self.win, text=label, fg=fg, bg=bg,
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
            scale = tk.Scale(self.win, from_=min_v, to=max_v, resolution=10**-digits,
                            orient=tk.HORIZONTAL, variable=var, command=on_change,
                            length=180, bg=bg, fg=fg, highlightbackground=bg,
                            troughcolor='#16213e', activebackground='#0f3460')
            val_label = tk.Label(self.win, textvariable=var, fg='#e94560', bg=bg,
                                font=('Segoe UI', 9, 'bold'), width=4)
            scale.grid(row=r, column=1, sticky='ew', padx=(3, 3), pady=2)
            val_label.grid(row=r, column=2, sticky='w', padx=(0, 15))
            return r + 1

        def add_dropdown(key, label, options, r):
            tk.Label(self.win, text=label, fg=fg, bg=bg,
                     font=('Segoe UI', 9)).grid(row=r, column=0, sticky='w', padx=(15, 5))
            var = tk.StringVar(value=self.cfg.get(key, options[0]))
            dropdown = ttk.Combobox(self.win, textvariable=var, values=options,
                                   state='readonly', width=14)
            dropdown.grid(row=r, column=1, sticky='w', padx=5, pady=2)
            def on_change(*args, k=key):
                self.cfg[k] = var.get()
                self.app.cfg[k] = var.get()
            var.trace_add('write', on_change)
            return r + 1

        # ─── Title ───
        tk.Label(self.win, text='♢ Hermes Cube', fg='#e94560', bg=bg,
                 font=('Segoe UI', 14, 'bold')).grid(row=row, column=0, columnspan=3,
                 pady=(15, 5))
        row += 1
        tk.Label(self.win, text='Настройки аватара', fg='#888', bg=bg,
                 font=('Segoe UI', 9)).grid(row=row, column=0, columnspan=3)
        row += 1

        # ─── Separator ───
        ttk.Separator(self.win, orient='horizontal').grid(row=row, column=0, columnspan=3,
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

        # ─── Particles ───
        add_label('Частицы', row); row += 1
        row = add_slider('particle_density', 'Плотность частиц', 6, 20, row, 0)
        row = add_slider('cell_size', 'Размер частицы (px)', 2, 12, row, 0)

        # ─── Style ───
        add_label('Стиль', row); row += 1
        row = add_dropdown('symbol', 'Форма частиц', ['square', 'circle', 'dot'], row)

        # ─── On top ───
        on_top_var = tk.BooleanVar(value=self.cfg.get('always_on_top', True))
        cb = tk.Checkbutton(self.win, text='Поверх всех окон', variable=on_top_var,
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
        btn_frame = tk.Frame(self.win, bg=bg)
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
