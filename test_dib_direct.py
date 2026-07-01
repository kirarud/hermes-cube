#!/usr/bin/env python3
"""Минимальный тест DIB — обычное окно без WS_EX_LAYERED, рисуем FBO."""
import ctypes, sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
os.environ['HERMES_LOCKED'] = '1'

from core.gpu import GpuRenderer

import moderngl
ctx = moderngl.create_standalone_context(require=330)
fbo = ctx.simple_framebuffer((400, 300))

renderer = GpuRenderer()
renderer.init_from_context(ctx)

# Яркие частицы
n = 500
rng = np.random.default_rng(42)
px = rng.uniform(50, 350, n).astype(np.float64)
py = rng.uniform(50, 250, n).astype(np.float64)
pz = np.zeros(n, dtype=np.float64)
rgb = rng.integers(100, 256, (n, 3), dtype=np.uint8)
renderer.upload(n)

# Рендер в FBO
fbo.use()
ctx.clear(0.0, 0.0, 0.0, 0.0)  # чёрный фон
renderer.render(px, py, pz, rgb, 400, 300, cell_size=12)

# Readback
buf = fbo.read(components=4)
arr = np.frombuffer(buf, dtype=np.uint8).reshape((300, 400, 4))
print(f'FBO center pixel: R={arr[150,200,0]} G={arr[150,200,1]} B={arr[150,200,2]} A={arr[150,200,3]}', flush=True)
has_pixels = np.any(arr[:,:,0] > 10) or np.any(arr[:,:,1] > 10) or np.any(arr[:,:,2] > 10)
print(f'FBO has visible content: {has_pixels}', flush=True)
if has_pixels:
    # Считаем сколько пикселей не-чёрных
    non_black = np.sum(np.any(arr[:,:,:3] > 5, axis=2))
    print(f'Non-black pixels: {non_black}/{400*300}', flush=True)

# Win32: полный экран, жёлтый паттерн
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

hinst = kernel32.GetModuleHandleW(None)
WC = 'TestDIB'

@ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)
def wndproc(hwnd, msg, wparam, lparam):
    if msg == 0x000F:  # WM_PAINT
        ps = ctypes.create_string_buffer(72)
        user32.BeginPaint(hwnd, ps)
        dc = user32.GetDC(hwnd)
        # DIB section
        from ctypes import wintypes
        class BMIH(ctypes.Structure):
            _fields_ = [('biSize', wintypes.DWORD), ('biWidth', ctypes.c_long),
                ('biHeight', ctypes.c_long), ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD)]
        hdr = BMIH()
        hdr.biSize = ctypes.sizeof(BMIH)
        hdr.biWidth = 400
        hdr.biHeight = -300  # top-down
        hdr.biPlanes = 1
        hdr.biBitCount = 32
        hdr.biCompression = 0
        class BMI(ctypes.Structure):
            _fields_ = [('bmiHeader', BMIH)]
        bmi = BMI(); bmi.bmiHeader = hdr
        
        bits_ptr = ctypes.c_void_p()
        dib = gdi32.CreateDIBSection(dc, ctypes.byref(bmi), 0, ctypes.byref(bits_ptr), None, 0)
        if dib:
            gdi32.SetDIBitsToDevice(dc, 0, 0, 400, 300, 0, 0, 0, 300,
                                    buf, ctypes.byref(bmi), 0)
            gdi32.DeleteObject(dib)
        user32.ReleaseDC(hwnd, dc)
        user32.EndPaint(hwnd, ps)
        return 0
    elif msg == 0x0002:  # WM_DESTROY
        user32.PostQuitMessage(0)
        return 0
    elif msg == 0x0010:  # WM_CLOSE
        user32.DestroyWindow(hwnd)
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

# Register
class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.c_uint), ('style', ctypes.c_uint),
        ('lpfnWndProc', ctypes.c_void_p), ('cbClsExtra', ctypes.c_int),
        ('cbWndExtra', ctypes.c_int), ('hInstance', ctypes.c_void_p),
        ('hIcon', ctypes.c_void_p), ('hCursor', ctypes.c_void_p),
        ('hbrBackground', ctypes.c_void_p), ('lpszMenuName', ctypes.c_wchar_p),
        ('lpszClassName', ctypes.c_wchar_p), ('hIconSm', ctypes.c_void_p),
    ]
wc = WNDCLASSEXW()
wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
wc.lpfnWndProc = ctypes.cast(wndproc, ctypes.c_void_p)
wc.hInstance = hinst
wc.hCursor = user32.LoadCursorW(None, 32512)
wc.hbrBackground = gdi32.GetStockObject(5)  # WHITE_BRUSH
wc.lpszClassName = WC
user32.RegisterClassExW(ctypes.byref(wc))

hwnd = user32.CreateWindowExW(
    0, WC, '♢ DIB TEST', 0x00CF0000, 200, 200, 400, 300,
    None, None, hinst, None)
user32.ShowWindow(hwnd, 5)  # SW_SHOW
user32.UpdateWindow(hwnd)
print(f'Window at (200,200) 400x300 — видишь цветные точки?', flush=True)

msg = (ctypes.c_void_p * 4)()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 30:
    while user32.PeekMessageW(msg, None, 0, 0, 1):
        user32.TranslateMessage(msg)
        user32.DispatchMessageW(msg)
    time.sleep(0.02)

user32.DestroyWindow(hwnd)
renderer.destroy()
print('DONE', flush=True)
