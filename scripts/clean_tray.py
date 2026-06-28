#!/usr/bin/env python3
"""Очистить все иконки Hermes Cube из системного трея.

Запускать когда иконка трея зависла после аварийного завершения.
"""
import ctypes
from ctypes import wintypes

class GUID(ctypes.Structure):
    _fields_ = [('Data1', wintypes.DWORD), ('Data2', wintypes.WORD),
                ('Data3', wintypes.WORD), ('Data4', wintypes.BYTE * 8)]

class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.DWORD), ('hWnd', wintypes.HANDLE), ('uID', wintypes.UINT),
        ('uFlags', wintypes.UINT), ('uCallbackMessage', wintypes.UINT), ('hIcon', wintypes.HANDLE),
        ('szTip', ctypes.c_wchar * 128), ('dwState', wintypes.DWORD), ('dwStateMask', wintypes.DWORD),
        ('szInfo', ctypes.c_wchar * 256), ('uVersion', wintypes.UINT), ('szInfoTitle', ctypes.c_wchar * 64),
        ('dwInfoFlags', wintypes.DWORD), ('guidItem', GUID), ('hBalloonIcon', wintypes.HANDLE),
    ]

shell32 = ctypes.windll.shell32
count = 0
for uid in range(100):
    nid = NOTIFYICONDATAW()
    nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
    nid.uID = uid
    if shell32.Shell_NotifyIconW(2, ctypes.byref(nid)):  # NIM_DELETE
        count += 1

ctypes.windll.user32.PostMessageW(0xFFFF, 0x001A, 0, 0)
print(f"Removed {count} ghost tray icons")
