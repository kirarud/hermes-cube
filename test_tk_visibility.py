#!/usr/bin/env python3
"""Тест Tkinter окна — без всяких Win32 трюков."""
import tkinter as tk
import time

root = tk.Tk()
root.title('♢ Hermes Cube — TEST')
root.geometry('400x300+300+200')
root.configure(bg='#ff0000')
root.attributes('-topmost', True)

label = tk.Label(root, text='♢ HERMES CUBE ♢', 
                 font=('Segoe UI', 24, 'bold'), 
                 fg='#ffffff', bg='#ff0000')
label.pack(expand=True, fill='both')

print("TK WINDOW AT (300,200) 400x300 — RED BG, WHITE TEXT", flush=True)

t0 = time.perf_counter()
while time.perf_counter() - t0 < 60:
    root.update()
    time.sleep(0.03)

root.destroy()
print("DONE", flush=True)
