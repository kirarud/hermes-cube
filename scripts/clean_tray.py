#!/usr/bin/env python3
"""clean_tray.py — Полная очистка иконок Hermes Cube из трея.

Перебирает все иконки в notification area (uID 0-1000),
удаляет те, у которых szTip содержит 'Hermes'.
Потом отправляет broadcast на обновление панели задач.
"""

import ctypes
from ctypes import wintypes

class GUID(ctypes.Structure):
    _fields_ = [('Data1', wintypes.DWORD), ('Data2', wintypes.WORD), 
                ('Data3', wintypes.WORD), ('Data4', wintypes.BYTE * 8)]

class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.DWORD),
        ('hWnd', wintypes.HANDLE),
        ('uID', wintypes.UINT),
        ('uFlags', wintypes.UINT),
        ('uCallbackMessage', wintypes.UINT),
        ('hIcon', wintypes.HANDLE),
        ('szTip', ctypes.c_wchar * 128),
        ('dwState', wintypes.DWORD),
        ('dwStateMask', wintypes.DWORD),
        ('szInfo', ctypes.c_wchar * 256),
        ('uVersion', wintypes.UINT),
        ('szInfoTitle', ctypes.c_wchar * 64),
        ('dwInfoFlags', wintypes.DWORD),
        ('guidItem', GUID),
        ('hBalloonIcon', wintypes.HANDLE),
    ]

shell32 = ctypes.windll.shell32
user32 = ctypes.windll.user32
NIM_DELETE = 2
WM_SETTINGCHANGE = 0x001A
HWND_BROADCAST = 0xFFFF

removed = 0
for uid in range(1000):
    nid = NOTIFYICONDATAW()
    nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
    nid.uID = uid
    nid.szTip = 'Hermes'
    ret = shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
    if ret:
        removed += 1

# Also try with szTip = '♢'
for uid in range(1000):
    nid = NOTIFYICONDATAW()
    nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
    nid.uID = uid
    nid.szTip = '♢'
    ret = shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))

# Broadcast to refresh notification area
user32.PostMessageW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, 0)

print(f"Removed {removed} Hermes tray icons")
