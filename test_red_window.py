#!/usr/bin/env python3
"""Минимальный тест: обычное окно без прозрачности."""
import ctypes, time

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

hinst = kernel32.GetModuleHandleW(None)

@ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
                     ctypes.c_void_p, ctypes.c_void_p)
def wndproc(hwnd, msg, wparam, lparam):
    if msg == 0x000F:  # WM_PAINT
        ps = ctypes.create_string_buffer(64)  # PAINTSTRUCT
        user32.BeginPaint(hwnd, ps)
        dc = user32.GetDC(hwnd)
        
        # fill red
        rect = (ctypes.c_long * 4)(0, 0, 400, 300)
        brush = gdi32.CreateSolidBrush(0x000000FF)  # RED
        gdi32.FillRect(dc, ctypes.byref(rect), brush)
        gdi32.DeleteObject(brush)
        
        # draw text
        user32.SetTextColor(dc, 0x00FFFFFF)  # white
        user32.SetBkMode(dc, 1)  # TRANSPARENT
        user32.DrawTextW(dc, '♢ HERMES CUBE ♢', -1,
                         ctypes.byref(rect), 0x0024)  # DT_CENTER | DT_VCENTER
        
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

# Register class
wc = (ctypes.c_ubyte * 48)()  # WNDCLASSEXW
ctypes.memset(wc, 0, 48)
ctypes.memmove(wc, (ctypes.c_uint * 3)(48, 0, ctypes.cast(wndproc, ctypes.c_void_p)), 12)
hinst_bytes = (ctypes.c_void_p * 1)(hinst)
ctypes.memmove(ctypes.byref(wc, 20), hinst_bytes, 8)  # hInstance at offset 20
class_name = ctypes.c_wchar_p('TestClass')
ctypes.memmove(ctypes.byref(wc, 40), ctypes.byref(class_name), 8)  # lpszClassName at offset 40
# hbrBackground at offset 32
bg = (ctypes.c_void_p * 1)(gdi32.GetStockObject(5))  # WHITE_BRUSH
ctypes.memmove(ctypes.byref(wc, 32), bg, 8)

user32.RegisterClassExW(wc)

hwnd = user32.CreateWindowExW(
    0, 'TestClass', '♢ HERMES CUBE - TEST RED',
    0x00CF0000,  # WS_OVERLAPPEDWINDOW | WS_VISIBLE
    300, 200, 400, 300,
    None, None, hinst, None)

user32.UpdateWindow(hwnd)
print(f'hwnd={hwnd}  RED WINDOW at (300,200) 400x300 — видишь?', flush=True)

# Message loop for 30 seconds
msg = (ctypes.c_void_p * 4)()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 30:
    while user32.PeekMessageW(msg, None, 0, 0, 1):
        user32.TranslateMessage(msg)
        user32.DispatchMessageW(msg)
    time.sleep(0.01)

user32.DestroyWindow(hwnd)
print('DONE', flush=True)
