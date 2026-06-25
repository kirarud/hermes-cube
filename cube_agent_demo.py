#!/usr/bin/env python3
"""cube_agent_demo.py — Demo: spawn UI agents on the PixelGrid.

Usage:
  from cube_agent_demo import run_demo
  run_demo(pixel_grid, agent_manager)

Shows:
  - Button "Hello" that prints to stdout on click
  - Slider "Brightness" that prints value on drag
  - TextLabel with instructions
  - ButtonAgent to close/open PixelGrid
"""

from __future__ import annotations

from typing import Any

from pixel_grid import PixelGrid
from cube_agents import (
    AgentManager, ButtonAgent, SliderAgent, TextLabel,
    AGENT_COLORS,
)


def run_demo(grid: PixelGrid, mgr: AgentManager,
             screen_w: int = 1280, screen_h: int = 720) -> None:
    """Spawn a demo interface on the PixelGrid.

    Call this once after AgentManager is created.
    """
    # Clear existing agents
    mgr.clear()

    # ── Title ────────────────────────────────────────────────────────
    title = TextLabel(grid, 10, 10, '♢ Hermes Cube — Agents',
                      color='border', max_width=screen_w - 20)
    mgr.add(title)

    # ── Instruction label ────────────────────────────────────────────
    instructions = TextLabel(
        grid, 10, 60,
        'G — скрыть PixelGrid  |  C — ввод  |  S — настройки',
        color='info', max_width=screen_w - 20)
    mgr.add(instructions)

    # ── Status label (bottom-left) ───────────────────────────────────
    status = TextLabel(
        grid, 10, screen_h - 30,
        'Агенты активны • Hermes Cube',
        color='success')
    mgr.add(status)

    # ── Demo Button ──────────────────────────────────────────────────
    def on_hello(mx: int, my: int, action: str) -> None:
        print(f'[CubeAgent] Button clicked at ({mx}, {my})!')

    btn = ButtonAgent(grid, 20, 100, text='Привет!',
                      callback=on_hello, color='border')
    mgr.add(btn)

    # ── Close Button (hides PixelGrid) ───────────────────────────────
    # Note: this button needs a reference to pixel_win to close.
    # We'll wire it from CubeApp.
    # mgr.add(close_btn)

    # ── Demo Slider ──────────────────────────────────────────────────
    def on_slider(val: float) -> None:
        print(f'[CubeAgent] Slider value: {val:.3f}')

    slider = SliderAgent(grid, 20, 150, label='Тест',
                         min_val=0.0, max_val=1.0, default=0.5,
                         callback=on_slider, w=200)
    mgr.add(slider)


def create_close_button(grid: PixelGrid, mgr: AgentManager,
                        on_close: Any) -> None:
    """Create a close button for the PixelGrid overlay."""
    btn = ButtonAgent(grid, 20, 200, text='✕ Закрыть',
                      callback=lambda mx, my, a: on_close(),
                      color='warning')
    mgr.add(btn)
