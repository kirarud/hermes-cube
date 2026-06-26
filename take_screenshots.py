#!/usr/bin/env python3
"""Take real screenshots of Hermes Cube in action."""
import os, time, sys, ctypes
from PIL import ImageGrab

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screenshots')
os.makedirs(OUT, exist_ok=True)

def shot(name, delay=0.5):
    time.sleep(delay)
    img = ImageGrab.grab()
    path = os.path.join(OUT, name)
    img.save(path)
    print(f"  Saved: {name} ({img.size[0]}x{img.size[1]})")

def press(key_code, mod=None):
    """Send a single key press via keybd_event."""
    if mod:
        ctypes.windll.user32.keybd_event(mod, 0, 0, 0)
    ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.05)
    ctypes.windll.user32.keybd_event(key_code, 0, 2, 0)
    if mod:
        ctypes.windll.user32.keybd_event(mod, 0, 2, 0)
    time.sleep(0.3)

# Wait for cube to initialize
time.sleep(3)

print("Taking screenshots...")

# 1. Cube in idle — full desktop
shot('01_idle.png', 0.5)

# 2. Settings window — press S
press(0x53)  # S
shot('02_settings.png', 0.5)

# 3. Change shape to sphere via settings (tab through) — just capture settings
# Close settings with Esc
press(0x1B)  # Escape
time.sleep(0.3)

# 4. Toggle draggable — press T
press(0x54)  # T
shot('03_draggable.png', 0.5)

# 5. Toggle trails — press R
press(0x52)  # R
shot('04_trails.png', 0.5)

# 6. Open AI input — press C
press(0x43)  # C
shot('05_ai_input.png', 0.5)

# Close input
press(0x1B)  # Escape
time.sleep(0.3)

# 7. PixelGrid — press G
press(0x47)  # G
shot('06_pixelgrid.png', 0.5)

# Close PixelGrid
press(0x47)  # G
time.sleep(0.3)

# 8. Spawn agent — press A
press(0x41)  # A
shot('07_agent.png', 1.0)

# 9. Context menu via keyboard simulation
# Show the cube again first
press(0x48)  # H (hide)
time.sleep(0.5)
press(0x48)  # H (show)
shot('08_overlay.png', 0.5)

# 10. Full screenshot with cube on desktop
time.sleep(1)
shot('00_hero.png', 0.5)

print(f"\nDone! {len([f for f in os.listdir(OUT) if f.endswith('.png')])} images")
