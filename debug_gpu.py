#!/usr/bin/env python3
"""debug_gpu.py — тест OpenGL standalone через moderngl и PyOpenGL."""

import ctypes
import numpy as np

# Загружаем opengl32
gl = ctypes.windll.opengl32

# Создаём dummy window для контекста
import ctypes.wintypes

class PIXELFORMATDESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ('nSize', ctypes.c_ushort),
        ('nVersion', ctypes.c_ushort),
        ('dwFlags', ctypes.c_uint32),
        ('iPixelType', ctypes.c_byte),
        ('cColorBits', ctypes.c_byte),
        ('cRedBits', ctypes.c_byte),
        ('cRedShift', ctypes.c_byte),
        ('cGreenBits', ctypes.c_byte),
        ('cGreenShift', ctypes.c_byte),
        ('cBlueBits', ctypes.c_byte),
        ('cBlueShift', ctypes.c_byte),
        ('cAlphaBits', ctypes.c_byte),
        ('cAlphaShift', ctypes.c_byte),
        ('cAccumBits', ctypes.c_byte),
        ('cAccumRedBits', ctypes.c_byte),
        ('cAccumGreenBits', ctypes.c_byte),
        ('cAccumBlueBits', ctypes.c_byte),
        ('cAccumAlphaBits', ctypes.c_byte),
        ('cDepthBits', ctypes.c_byte),
        ('cStencilBits', ctypes.c_byte),
        ('cAuxBuffers', ctypes.c_byte),
        ('iLayerType', ctypes.c_byte),
        ('bReserved', ctypes.c_byte),
        ('dwLayerMask', ctypes.c_uint32),
        ('dwVisibleMask', ctypes.c_uint32),
        ('dwDamageMask', ctypes.c_uint32),
    ]

print("1. Creating window class...")
user32 = ctypes.windll.user32
WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)

wc = ctypes.c_ubyte * 80
wnd_class = wc()
wnd_proc = WNDPROC(lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l))

hinst = ctypes.windll.kernel32.GetModuleHandleW(None)
hinst = ctypes.c_void_p(hinst)

class_name = 'TestGL'
# Простейшая регистрация

print("2. Creating window...")
hwnd = user32.CreateWindowExW(
    0, class_name, 'Debug', 0x80000000,  # WS_POPUP
    0, 0, 100, 100, None, None, hinst, None
)
print(f"  HWND: {hwnd}")

print("3. Getting DC...")
dc = user32.GetDC(hwnd)
print(f"  DC: {dc}")

print("4. Setting pixel format...")
pfd = ctypes.create_string_buffer(40)
ctypes.memset(pfd, 0, 40)

import struct
data = struct.pack('HHI', 40, 1, 0x0025)  # PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
ctypes.memmove(pfd, data, 8)
pfd[8] = 32  # color bits
pfd[23] = 24  # depth bits

ChoosePixelFormat = ctypes.windll.gdi32.ChoosePixelFormat
pf = ChoosePixelFormat(dc, pfd)
SetPixelFormat = ctypes.windll.gdi32.SetPixelFormat
SetPixelFormat(dc, pf, pfd)

print(f"  PixelFormat: {pf}")

print("5. Creating GL context...")
glrc = gl.wglCreateContext(dc)
gl.wglMakeCurrent(dc, glrc)

print("6. Simple test: clear to red...")
gl.glClearColor(1.0, 0.0, 0.0, 1.0)
gl.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT

# Read pixel
gl.glReadPixels = ctypes.windll.opengl32.glReadPixels
gl.glReadPixels.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
pixel = (ctypes.c_ubyte * 4)()
gl.glReadPixels(50, 50, 1, 1, 0x1908, 0x1401, pixel)  # GL_RGBA, GL_UNSIGNED_BYTE
print(f"  Center pixel after GL clear: ({pixel[0]}, {pixel[1]}, {pixel[2]}, {pixel[3]})")

print("7. Drawing a quad...")
gl.glBegin(0x0007)  # GL_QUADS
gl.glColor3f(0.0, 1.0, 0.0)
gl.glVertex2f(-0.5, -0.5)
gl.glVertex2f(0.5, -0.5)
gl.glVertex2f(0.5, 0.5)
gl.glVertex2f(-0.5, 0.5)
gl.glEnd()
gl.glFinish()

gl.glReadPixels(50, 50, 1, 1, 0x1908, 0x1401, pixel)
print(f"  Center pixel after quad draw: ({pixel[0]}, {pixel[1]}, {pixel[2]}, {pixel[3]})")

# Cleanup
gl.wglMakeCurrent(None, None)
gl.wglDeleteContext(glrc)
user32.ReleaseDC(hwnd, dc)
user32.DestroyWindow(hwnd)

print("\nDone.")
