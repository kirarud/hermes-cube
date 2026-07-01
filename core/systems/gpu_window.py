"""systems/gpu_window.py — OpenGL overlay window (Windows, standalone context).

Архитектура:
  - Win32 прозрачное окно с OpenGL-compatible pixel format
  - moderngl standalone контекст (FBO-based)
  - Рендер в FBO → PBO readback → Win32 DIB → UpdateLayeredWindow
  - Никакого Tk canvas, PIL, PhotoImage
  - Click-through (WS_EX_TRANSPARENT), topmost, прозрачность color-key
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Any, Optional, Callable

import numpy as np
from numpy.typing import NDArray


if sys.platform != 'win32':
    raise RuntimeError("GpuWindowSystem requires Windows")

# ═══════════════════════════════════════════════════════════════════════════
# Win32
# ═══════════════════════════════════════════════════════════════════════════

_GetModuleHandleW = ctypes.windll.kernel32.GetModuleHandleW
_GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
_GetModuleHandleW.restype = ctypes.c_void_p

_RegisterClassExW = ctypes.windll.user32.RegisterClassExW

_CreateWindowExW = ctypes.windll.user32.CreateWindowExW
_CreateWindowExW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_wchar_p,
    ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_CreateWindowExW.restype = ctypes.c_void_p

_DestroyWindow = ctypes.windll.user32.DestroyWindow
_DefWindowProcW = ctypes.windll.user32.DefWindowProcW
_DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
_DefWindowProcW.restype = ctypes.c_long

_GetDC = ctypes.windll.user32.GetDC
_ReleaseDC = ctypes.windll.user32.ReleaseDC
_SetWindowLongW = ctypes.windll.user32.SetWindowLongW
_GetWindowLongW = ctypes.windll.user32.GetWindowLongW
_SetLayeredWindowAttributes = ctypes.windll.user32.SetLayeredWindowAttributes
_ShowWindow = ctypes.windll.user32.ShowWindow
_GetSystemMetrics = ctypes.windll.user32.GetSystemMetrics
_PeekMessageW = ctypes.windll.user32.PeekMessageW
_TranslateMessage = ctypes.windll.user32.TranslateMessage
_DispatchMessageW = ctypes.windll.user32.DispatchMessageW
_UpdateLayeredWindow = ctypes.windll.user32.UpdateLayeredWindow
_UpdateLayeredWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
_UpdateLayeredWindow.restype = ctypes.c_bool

_SetWindowPos = ctypes.windll.user32.SetWindowPos
_SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_SetWindowPos.restype = ctypes.c_bool

_CreateCompatibleDC = ctypes.windll.gdi32.CreateCompatibleDC
_DeleteDC = ctypes.windll.gdi32.DeleteDC
_CreateDIBSection = ctypes.windll.gdi32.CreateDIBSection
_CreateDIBSection.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
_CreateDIBSection.restype = ctypes.c_void_p
_SelectObject = ctypes.windll.gdi32.SelectObject
_SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_SelectObject.restype = ctypes.c_void_p
_SwapBuffers = ctypes.windll.gdi32.SwapBuffers
_SwapBuffers.argtypes = [ctypes.c_void_p]
_SwapBuffers.restype = ctypes.c_bool

_DeleteObject = ctypes.windll.gdi32.DeleteObject
_DeleteObject.argtypes = [ctypes.c_void_p]
_DeleteObject.restype = ctypes.c_bool

_BitBlt = ctypes.windll.gdi32.BitBlt
_BitBlt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
_BitBlt.restype = ctypes.c_bool

WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
SW_HIDE = 0
SW_RESTORE = 9
SM_CXSCREEN = 0
SM_CYSCREEN = 1
LWA_COLORKEY = 0x00000001
HTTRANSPARENT = -1
WM_NCHITTEST = 0x0084
WM_CLOSE = 0x0010
WM_KEYDOWN = 0x0100
WM_SIZE = 0x0005
PM_REMOVE = 0x0001

TRANSPARENT_RGB = 0x000100  # BGR (0,0,1) = #000001 — прозрачный цвет для color-key


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.c_uint), ('style', ctypes.c_uint),
        ('lpfnWndProc', ctypes.c_void_p), ('cbClsExtra', ctypes.c_int),
        ('cbWndExtra', ctypes.c_int), ('hInstance', ctypes.c_void_p),
        ('hIcon', ctypes.c_void_p), ('hCursor', ctypes.c_void_p),
        ('hbrBackground', ctypes.c_void_p), ('lpszMenuName', ctypes.c_wchar_p),
        ('lpszClassName', ctypes.c_wchar_p), ('hIconSm', ctypes.c_void_p),
    ]


class POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


class SIZE(ctypes.Structure):
    _fields_ = [('cx', ctypes.c_long), ('cy', ctypes.c_long)]


class MSG(ctypes.Structure):
    _fields_ = [
        ('hwnd', ctypes.c_void_p), ('message', ctypes.c_uint),
        ('wParam', ctypes.c_void_p), ('lParam', ctypes.c_void_p),
        ('time', ctypes.c_ulong), ('pt', POINT),
        ('lPrivate', ctypes.c_uint),
    ]


BLENDFUNCTION = ctypes.c_ubyte * 4


# ═══════════════════════════════════════════════════════════════════════════
# GpuWindowSystem
# ═══════════════════════════════════════════════════════════════════════════

class GpuWindowSystem:
    """Прозрачное Win32 окно с OpenGL FBO-рендером через moderngl.

    Рендер:
      1. moderngl standalone контекст (не зависит от окна)
      2. FBO render
      3. PBO readback → numpy (уже shared с DIB)
      4. UpdateLayeredWindow на Win32 окно
    """

    WND_CLASS: str = 'HermesEngineGL'

    def __init__(self, width: Optional[int] = None, height: Optional[int] = None,
                 x: int = 0, y: int = 0, clickthrough: bool = True) -> None:
        self._hwnd: Optional[int] = None
        self._dc: Optional[int] = None  # Window DC
        self._mem_dc: Optional[int] = None  # Memory DC for DIB
        self._dib_bits: Optional[int] = None  # Pointer to DIB pixels
        self._dib_bitmap: Optional[int] = None
        self._gl_ctx: Any = None
        self._fbo: Any = None
        self._moderngl: Any = None

        self._width: int = width if width else _GetSystemMetrics(SM_CXSCREEN)
        self._height: int = height if height else _GetSystemMetrics(SM_CYSCREEN)
        self._start_x: int = x
        self._start_y: int = y
        self._clickthrough: bool = clickthrough
        self._visible: bool = False

        # Callbacks
        self.on_key: Optional[Callable[[int], None]] = None
        self.on_quit: Optional[Callable] = None

        self._create_window()

    def _create_window(self) -> None:
        hinst = _GetModuleHandleW(None)

        # Register class
        wnd_proc = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
            ctypes.c_void_p, ctypes.c_void_p,
        )(self._wnd_proc)

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = ctypes.cast(wnd_proc, ctypes.c_void_p)
        wc.hInstance = hinst
        wc.hCursor = ctypes.windll.user32.LoadCursorW(None, 32512)
        wc.lpszClassName = self.WND_CLASS
        _RegisterClassExW(ctypes.byref(wc))

        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE
        hwnd = _CreateWindowExW(
            ex_style, self.WND_CLASS, '♢ Hermes Cube',
            WS_POPUP | WS_VISIBLE,
            self._start_x, self._start_y, self._width, self._height,
            None, None, hinst, None,
        )
        if not hwnd:
            raise RuntimeError(f"CreateWindowEx failed: {ctypes.GetLastError()}")

        self._hwnd = hwnd
        _SetLayeredWindowAttributes(hwnd, TRANSPARENT_RGB, 0, LWA_COLORKEY)
        _ShowWindow(hwnd, SW_HIDE)

        # OpenGL setup
        self._init_opengl()
        self._init_dib()

    def _init_opengl(self) -> None:
        """Создать moderngl standalone контекст."""
        import moderngl
        self._moderngl = moderngl
        self._gl_ctx = moderngl.create_standalone_context(require=330)
        self._fbo = self._gl_ctx.simple_framebuffer((self._width, self._height))

    def _init_dib(self) -> None:
        """Создать DIB section для Win32 blit."""
        from ctypes import wintypes

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD),
                ('biWidth', ctypes.c_long),
                ('biHeight', ctypes.c_long),
                ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD),
                ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD),
                ('biXPelsPerMeter', ctypes.c_long),
                ('biYPelsPerMeter', ctypes.c_long),
                ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [('bmiHeader', BITMAPINFOHEADER)]

        self._dc = _GetDC(self._hwnd)
        self._mem_dc = _CreateCompatibleDC(self._dc)

        hdr = BITMAPINFOHEADER()
        hdr.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        hdr.biWidth = self._width
        hdr.biHeight = -self._height  # negative = top-down
        hdr.biPlanes = 1
        hdr.biBitCount = 32
        hdr.biCompression = 0  # BI_RGB

        bmi = BITMAPINFO()
        bmi.bmiHeader = hdr

        bits_ptr = ctypes.c_void_p()
        self._dib_bitmap = _CreateDIBSection(
            self._mem_dc, ctypes.byref(bmi), 0, ctypes.byref(bits_ptr), None, 0,
        )
        if not self._dib_bitmap:
            err = ctypes.GetLastError()
            raise RuntimeError(f"CreateDIBSection failed: {err}")
        self._dib_bits = bits_ptr.value
        _SelectObject(self._mem_dc, self._dib_bitmap)

    # ── Window proc ──────────────────────────────────────────────────

    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == WM_NCHITTEST and self._clickthrough:
            return HTTRANSPARENT
        if msg == WM_CLOSE:
            if self.on_quit:
                self.on_quit()
            return 0
        if msg == WM_KEYDOWN and self.on_key:
            self.on_key(wparam & 0xFF)
            return 0
        if msg == WM_SIZE:
            self._width = lparam & 0xFFFF
            self._height = (lparam >> 16) & 0xFFFF
            return 0
        return _DefWindowProcW(hwnd, msg, wparam, lparam)

    # ── Rendering ────────────────────────────────────────────────────

    @property
    def ctx(self) -> Any:
        return self._gl_ctx

    @property
    def w(self) -> int:
        return self._width

    @property
    def h(self) -> int:
        return self._height

    def make_current(self) -> None:
        """Установить FBO как текущую цель рендера.
        alpha=0 — прозрачный для ULW_ALPHA."""
        self._fbo.use()
        self._gl_ctx.clear(0.0, 0.0, 1.0/255.0, 0.0)

    def swap_buffers(self) -> None:
        """PBO readback → memmove → DIB → UpdateLayeredWindow (ULW_ALPHA).

        Шейдер уже выводит BGR, DIB получает правильные цвета напрямую.
        """
        if self._hwnd is None or not self._visible:
            return

        buf = self._fbo.read(components=4)
        ctypes.memmove(self._dib_bits, buf, self._width * self._height * 4)

        dc_screen = _GetDC(None)
        pt_src = POINT(0, 0)
        pt_dst = POINT(0, 0)
        size = SIZE(self._width, self._height)
        blend = BLENDFUNCTION(0, 0, 255, 1)  # AC_SRC_ALPHA
        _UpdateLayeredWindow(
            self._hwnd, dc_screen, ctypes.byref(pt_dst),
            ctypes.byref(size), self._mem_dc, ctypes.byref(pt_src),
            0, ctypes.byref(blend), 2,  # ULW_ALPHA
        )
        _ReleaseDC(None, dc_screen)

    def show(self) -> None:
        if not self._visible and self._hwnd:
            _ShowWindow(self._hwnd, SW_RESTORE)
            self._visible = True

    def hide(self) -> None:
        if self._visible and self._hwnd:
            _ShowWindow(self._hwnd, SW_HIDE)
            self._visible = False

    def toggle_visible(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def set_clickthrough(self, on: bool = True) -> None:
        self._clickthrough = on
        if self._hwnd:
            ex = _GetWindowLongW(self._hwnd, -20)
            ex = ex | WS_EX_TRANSPARENT if on else ex & ~WS_EX_TRANSPARENT
            _SetWindowLongW(self._hwnd, -20, ex)

    def resize(self, w: int, h: int, x: int = 0, y: int = 0) -> None:
        """Изменить размер окна, FBO и DIB."""
        self._width = w
        self._height = h
        self._start_x = x
        self._start_y = y
        if self._hwnd:
            _SetWindowPos(self._hwnd, None, x, y, w, h,
                          0x0004 | 0x0020)  # SWP_NOZORDER | SWP_NOACTIVATE
        # Recreate FBO
        self._fbo = self._gl_ctx.simple_framebuffer((w, h))
        # Recreate DIB
        if self._dib_bitmap:
            _DeleteObject(self._dib_bitmap)
            self._dib_bitmap = None
        if self._mem_dc:
            _DeleteDC(self._mem_dc)
            self._mem_dc = None
        self._init_dib()

    def pump_messages(self) -> None:
        msg = MSG()
        while _PeekMessageW(ctypes.byref(msg), self._hwnd, 0, 0, PM_REMOVE):
            _TranslateMessage(ctypes.byref(msg))
            _DispatchMessageW(ctypes.byref(msg))

    def destroy(self) -> None:
        self._fbo = None
        self._gl_ctx = None
        self._moderngl = None
        if self._dib_bitmap:
            _DeleteObject(self._dib_bitmap)
            self._dib_bitmap = None
        if self._mem_dc:
            _DeleteDC(self._mem_dc)
            self._mem_dc = None
        if self._dc:
            _ReleaseDC(self._hwnd, self._dc)
            self._dc = None
        if self._hwnd:
            _DestroyWindow(self._hwnd)
            self._hwnd = None
