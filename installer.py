#!/usr/bin/env python3
"""
Hermes Cube Setup — Installer
Copies HermesCube.exe, creates shortcuts, optional autostart.
Single .exe via PyInstaller.
"""

import os
import sys
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─── Paths ────────────────────────────────────────────────────────────
SRC_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HermesCube.exe')
AUTOSTART_KEY = r'Software\Microsoft\Windows\CurrentVersion\Run'


def get_default_install_dir():
    return os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'HermesCube')


def get_desktop_path():
    return os.path.join(os.path.expanduser('~'), 'Desktop')


def get_startmenu_path():
    return os.path.join(os.environ.get('APPDATA', ''), r'Microsoft\Windows\Start Menu\Programs\Hermes Cube')


def create_shortcut(target_path, shortcut_path, description='♢ Hermes Cube'):
    """Create a .lnk shortcut using win32com."""
    try:
        import win32com.client
        shell = win32com.client.Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.TargetPath = target_path
        shortcut.WorkingDirectory = os.path.dirname(target_path)
        shortcut.Description = description
        shortcut.Save()
        return True
    except Exception as e:
        print(f"Shortcut error: {e}")
        return False


def add_autostart(exe_path):
    """Add to Windows autostart via registry."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY, 0,
                             winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, 'HermesCube', 0, winreg.REG_SZ, exe_path)
        winreg.CloseKey(key)
        return True
    except Exception as e:
        print(f"Autostart error: {e}")
        return False


def remove_autostart():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_KEY, 0,
                             winreg.KEY_SET_VALUE)
        try:
            winreg.DeleteValue(key, 'HermesCube')
        except OSError:
            pass
        winreg.CloseKey(key)
    except:
        pass


# ─── GUI Installer ────────────────────────────────────────────────────
class InstallerApp:
    BG = '#1a1a2e'
    FG = '#e0e0e0'
    ACCENT = '#e94560'
    ACCENT2 = '#0f3460'

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('♢ Hermes Cube — Установка')
        self.root.geometry('480x480')
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)

        self.install_dir = tk.StringVar(value=get_default_install_dir())
        self.create_desktop = tk.BooleanVar(value=True)
        self.create_startmenu = tk.BooleanVar(value=True)
        self.add_autostart = tk.BooleanVar(value=False)
        self.launch_after = tk.BooleanVar(value=True)

        self._build_ui()
        self._center_window()

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f'+{x}+{y}')

    def _build_ui(self):
        bg = self.BG
        fg = self.FG

        # ─── Title ───
        tk.Label(self.root, text='♢ Hermes Cube', fg=self.ACCENT, bg=bg,
                 font=('Segoe UI', 18, 'bold')).pack(pady=(25, 0))
        tk.Label(self.root, text='Установка аватара', fg='#888', bg=bg,
                 font=('Segoe UI', 10)).pack(pady=(0, 10))

        # ─── Separator ───
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=30, pady=5)

        # ─── Install path ───
        frame = tk.Frame(self.root, bg=bg)
        frame.pack(fill='x', padx=30, pady=15)

        tk.Label(frame, text='Путь установки:', fg=fg, bg=bg,
                 font=('Segoe UI', 9)).pack(anchor='w')
        path_frame = tk.Frame(frame, bg=bg)
        path_frame.pack(fill='x', pady=(5, 0))
        self.path_entry = tk.Entry(path_frame, textvariable=self.install_dir,
                                   bg='#16213e', fg=fg, relief=tk.FLAT,
                                   font=('Segoe UI', 9))
        self.path_entry.pack(side='left', fill='x', expand=True, ipady=3)
        tk.Button(path_frame, text='…', command=self._browse,
                  bg=self.ACCENT2, fg=fg, relief=tk.FLAT, width=3,
                  activebackground=self.ACCENT, activeforeground='#fff',
                  font=('Segoe UI', 9, 'bold')).pack(side='right', padx=(5, 0))

        # ─── Options ───
        opt_frame = tk.Frame(self.root, bg=bg)
        opt_frame.pack(fill='x', padx=30, pady=5)

        for var, text in [
            (self.create_desktop, 'Ярлык на рабочем столе'),
            (self.create_startmenu, 'Ярлык в меню Пуск'),
            (self.add_autostart, 'Автозапуск при входе в Windows'),
            (self.launch_after, 'Запустить после установки'),
        ]:
            cb = tk.Checkbutton(opt_frame, text=text, variable=var,
                                bg=bg, fg=fg, selectcolor='#16213e',
                                activebackground=bg, activeforeground=fg,
                                font=('Segoe UI', 9), anchor='w')
            cb.pack(fill='x', pady=2)

        # ─── Progress ───
        self.progress = ttk.Progressbar(self.root, mode='determinate', length=400)
        self.progress.pack(pady=(15, 5), padx=30)

        self.status_label = tk.Label(self.root, text='', fg='#aaa', bg=bg,
                                     font=('Segoe UI', 9))
        self.status_label.pack()

        # ─── Buttons ───
        btn_frame = tk.Frame(self.root, bg=bg)
        btn_frame.pack(pady=20)

        self.install_btn = tk.Button(btn_frame, text='⚡ Установить', command=self._do_install,
                                     bg=self.ACCENT2, fg=fg, relief=tk.FLAT, padx=20, pady=5,
                                     activebackground=self.ACCENT, activeforeground='#fff',
                                     font=('Segoe UI', 10, 'bold'))
        self.install_btn.pack(side='left', padx=5)

        self.cancel_btn = tk.Button(btn_frame, text='✕ Отмена', command=self.root.destroy,
                                    bg='#333', fg=fg, relief=tk.FLAT, padx=20, pady=5,
                                    activebackground='#555', activeforeground='#fff',
                                    font=('Segoe UI', 10))
        self.cancel_btn.pack(side='left', padx=5)

    def _browse(self):
        path = filedialog.askdirectory(initialdir=self.install_dir.get(),
                                       title='Выберите папку для установки')
        if path:
            self.install_dir.set(path)

    def _set_status(self, text, pct=None):
        self.status_label.config(text=text)
        if pct is not None:
            self.progress['value'] = pct
        self.root.update_idletasks()

    def _do_install(self):
        self.install_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
        threading.Thread(target=self._install_thread, daemon=True).start()

    def _install_thread(self):
        nonlocal_src = SRC_EXE
        try:
            install_path = self.install_dir.get()
            target_exe = os.path.join(install_path, 'HermesCube.exe')

            # Step 1: Verify source
            self.root.after(0, lambda: self._set_status('📁 Проверка исходного файла...', 5))
            src_paths = [
                nonlocal_src,
                os.path.join(os.path.dirname(__file__), 'HermesCube.exe'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HermesCube.exe'),
            ]
            found_src = None
            for p in src_paths:
                if os.path.exists(p):
                    found_src = p
                    break
            if not found_src:
                raise FileNotFoundError(
                    f'HermesCube.exe не найден рядом с установщиком!\n'
                    f'Искал:\n' + '\n'.join(f'  {p}' for p in src_paths))
            exe_size = os.path.getsize(found_src)
            size_mb = exe_size / (1024 * 1024)

            # Step 2: Create install dir
            self.root.after(0, lambda: self._set_status(f'📂 Создание папки: {install_path}', 15))
            os.makedirs(install_path, exist_ok=True)

            # Step 3: Copy HermesCube.exe
            self.root.after(0, lambda: self._set_status(f'📦 Копирование ({size_mb:.0f} МБ)...', 30))
            shutil.copy2(found_src, target_exe)

            # Step 4: Create shortcuts
            if self.create_desktop.get():
                desktop_shortcut = os.path.join(get_desktop_path(), '♢ Hermes Cube.lnk')
                self.root.after(0, lambda: self._set_status('🖥 Создание ярлыка на рабочем столе...', 50))
                ok = create_shortcut(target_exe, desktop_shortcut)
                if not ok:
                    # Fallback: create a .url file
                    url_path = os.path.join(get_desktop_path(), '♢ Hermes Cube.url')
                    with open(url_path, 'w') as f:
                        f.write(f'[InternetShortcut]\nURL=file:///{target_exe}\nIconFile={target_exe}\nIconIndex=0\n')

            if self.create_startmenu.get():
                sm_path = get_startmenu_path()
                os.makedirs(sm_path, exist_ok=True)
                sm_shortcut = os.path.join(sm_path, '♢ Hermes Cube.lnk')
                self.root.after(0, lambda: self._set_status('📋 Создание ярлыка в меню Пуск...', 65))
                create_shortcut(target_exe, sm_shortcut)

            if self.add_autostart.get():
                self.root.after(0, lambda: self._set_status('⚡ Добавление в автозагрузку...', 80))
                add_autostart(target_exe)
            else:
                remove_autostart()

            # Step 5: Done
            self.root.after(0, lambda: self._set_status('✅ Установка завершена!', 100))

            # Launch
            if self.launch_after.get():
                subprocess.Popen([target_exe], shell=True)

            self.root.after(200, lambda: messagebox.showinfo(
                '♢ Hermes Cube',
                f'Установка завершена!\n\n'
                f'Куб установлен в:\n{install_path}\n\n'
                f'{"✅ Запущен" if self.launch_after.get() else ""}'
            ))
            self.root.after(300, self.root.destroy)

        except Exception as e:
            self.root.after(0, lambda: self._set_status(f'❌ Ошибка: {e}', 0))
            self.root.after(0, lambda: messagebox.showerror(
                'Ошибка установки', str(e)))
            self.root.after(0, lambda: self.install_btn.config(state='normal'))
            self.root.after(0, lambda: self.cancel_btn.config(state='normal'))

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    # Check if source exe exists next to us
    app = InstallerApp()
    app.run()
