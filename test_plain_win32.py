#!/usr/bin/env python3
"""Тест: простое Win32 окно с синим фоном."""
import ctypes, sys

_GetModuleHandleW = ctypes.windll.kernel32.GetModuleHandleW
_RegisterClassExW = ctypes.windll.user32.RegisterClassExW
_CreateWindowExW = ctypes.windll.user32.CreateWindowExW
_ShowWindow = ctypes.windll.user32.ShowWindow
_UpdateWindow = ctypes.windll.user32.UpdateWindow
_GetDC = ctypes.windll.user32.GetDC
_ReleaseDC = ctypes.windll.user32.ReleaseDC
_DefWindowProcW = ctypes.windll.user32.DefWindowProcW
_PeekMessageW = ctypes.windll.user32.PeekMessageW
_TranslateMessage = ctypes.windll.user32.TranslateMessage
_DispatchMessageW = ctypes.windll.user32.DispatchMessageW
_DestroyWindow = ctypes.windll.user32.DestroyWindow
_PostQuitMessage = ctypes.windll.user32.PostQuitMessage
_FillRect = ctypes.windll.user32.FillRect
_GetStockObject = ctypes.windll.gdi32.GetStockObject

class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.c_uint), ('style', ctypes.c_uint),
        ('lpfnWndProc', ctypes.c_void_p), ('cbClsExtra', ctypes.c_int),
        ('cbWndExtra', ctypes.c_int), ('hInstance', ctypes.c_void_p),
        ('hIcon', ctypes.c_void_p), ('hCursor', ctypes.c_void_p),
        ('hbrBackground', ctypes.c_void_p), ('lpszMenuName', ctypes.c_wchar_p),
        ('lpszClassName', ctypes.c_wchar_p), ('hIconSm', ctypes.c_void_p),
    ]

class MSG(ctypes.Structure):
    _fields_ = [
        ('hwnd', ctypes.c_void_p), ('message', ctypes.c_uint),
        ('wParam', ctypes.c_void_p), ('lParam', ctypes.c_void_p),
        ('time', ctypes.c_ulong), ('pt', ctypes.c_ulong * 2),
    ]

class RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

hinst = _GetModuleHandleW(None)

@ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)
def wndproc(hwnd, msg, wparam, lparam):
    if msg == 0x000F:  # WM_PAINT
        ps = ctypes.create_string_buffer(64)
        ctypes.windll.user32.BeginPaint(hwnd, ps)
        dc = ctypes.windll.user32.GetDC(hwnd)
        brush = _GetStockObject(10)  # DC_BRUSH
        ctypes.windll.user32.SetDCBrushColor(dc, 0x00FF0000)  # BLUE
        rect = RECT(0, 0, 400, 300)
        _FillRect(dc, ctypes.byref(rect), brush)
        _ReleaseDC(hwnd, dc)
        ctypes.windll.user32.EndPaint(hwnd, ps)
        return 0
    elif msg == 0x0002:  # WM_DESTROY
        _PostQuitMessage(0)
        return 0
    return _DefWindowProcW(hwnd, msg, wparam, lparam)

wc = WNDCLASSEXW()
wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
wc.lpfnWndProc = ctypes.cast(wndproc, ctypes.c_void_p)
wc.hInstance = hinst
wc.hCursor = ctypes.windll.user32.LoadCursorW(None, 32512)
wc.hbrBackground = 6  # COLOR_WINDOW
wc.lpszClassName = 'TestWindow'
_RegisterClassExW(ctypes.byref(wc))

hwnd = _CreateWindowExW(
    0, 'TestWindow', 'Test Blue Window',
    0x00CF0000,  # WS_OVERLAPPEDWINDOW | WS_VISIBLE
    200, 200, 400, 300,
    None, None, hinst, None)
print(f'hwnd={hwnd}', flush=True)
_ShowWindow(hwnd, 5)  # SW_SHOW
_UpdateWindow(hwnd)

print('BLUE WINDOW at (200,200). Running 20 sec...', flush=True)
import time
t0 = time.perf_counter()
msg = MSG()
while time.perf_counter() - t0 < 20:
    while _PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
        _TranslateMessage(ctypes.byref(msg))
        _DispatchMessageW(ctypes.byref(msg))

_DestroyWindow(hwnd)
print('DONE', flush=True)
