#!/usr/bin/env python3
"""cube_agents.py — PixelGrid-based UI agents for Hermes Cube.

Architecture:
  CubeAgent        — abstract base: owns pixel area, renders into PixelGrid
  ButtonAgent      — clickable button from pixels
  SliderAgent      — draggable slider
  TextLabel        — static text label

Each agent:
  - Claims a rectangular area on the PixelGrid buffer
  - Renders itself via paint primitives (no tkinter widgets)
  - Registers/unregisters hit zones for mouse interaction
  - Has a unique key to avoid collisions

Integration:
  CubeApp (cube_app.py) imports CubeAgentManager which holds all active agents.
  G key / tray toggles the PixelGrid overlay and starts rendering agents.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from pixel_grid import PixelGrid, HitCallback

# ── Colour palette for pixel agents ─────────────────────────────────────

AGENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    'bg':        (10, 10, 30),     # dark background
    'border':    (200, 60, 80),    # red-orange accent
    'border_hi': (255, 100, 120),  # hover border
    'text':      (220, 220, 240),  # light text
    'track':     (40, 40, 60),     # slider track
    'fill':      (200, 60, 80),    # slider fill
    'handle':    (255, 255, 255),  # slider handle
    'success':   (80, 200, 80),    # green
    'warning':   (240, 200, 40),   # yellow
    'info':      (60, 150, 240),   # blue
}

AgentRenderFn = Callable[[], None]
AgentCallback = Callable[[int, int, str], None]
"""Callback signature: (mouse_x, mouse_y, action) where action is 'click'|'drag'|'release'."""


# ═══════════════════════════════════════════════════════════════════════════
# CubeAgent — abstract base
# ═══════════════════════════════════════════════════════════════════════════


class CubeAgent:
    """Abstract agent that paints into a PixelGrid buffer.

    Subclass and override `render()` to draw content.
    Call `remove()` to unregister hit zones and clear area.
    """

    def __init__(self, grid: PixelGrid, x: int, y: int, w: int, h: int,
                 key: str = '') -> None:
        self.grid: PixelGrid = grid
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        self.key: str = key or f'agent_{id(self)}'
        self.visible: bool = True
        self._registered: bool = False
        self._on_click: Optional[HitCallback] = None
        self._on_drag: Optional[HitCallback] = None
        self._on_release: Optional[HitCallback] = None

    def render(self) -> None:
        """Draw agent content into the PixelGrid buffer.
        Override in subclass.
        """
        raise NotImplementedError

    def _register(self) -> None:
        """Register hit zone for this agent's area."""
        if not self._registered:
            self.grid.register_zone(
                self.x, self.y, self.x + self.w, self.y + self.h,
                on_click=self._on_click,
                on_drag=self._on_drag,
                on_release=self._on_release,
                agent=self,
            )
            self._registered = True

    def _unregister(self) -> None:
        """Remove hit zone."""
        if self._registered:
            self.grid.unregister_zone(
                self.x, self.y, self.x + self.w, self.y + self.h)
            self._registered = False

    def remove(self) -> None:
        """Clean up: clear buffer area and unregister zone."""
        self.grid.clear_rect(self.x, self.y,
                             self.x + self.w, self.y + self.h)
        self._unregister()
        self.visible = False

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False
        self.grid.clear_rect(self.x, self.y,
                             self.x + self.w, self.y + self.h)


# ═══════════════════════════════════════════════════════════════════════════
# ButtonAgent
# ═══════════════════════════════════════════════════════════════════════════


class ButtonAgent(CubeAgent):
    """Pixel-art button with text label and click callback.

    Renders as a bordered rectangle with centered 5×7 text.
    """

    PAD: int = 4
    MIN_W: int = 30
    MIN_H: int = 20

    def __init__(self, grid: PixelGrid, x: int, y: int,
                 text: str = 'OK',
                 callback: Optional[AgentCallback] = None,
                 color: str = 'border',
                 w: int = 0, h: int = 0) -> None:
        # Auto-size from text length
        txt_w: int = len(text) * 6 + 10  # 5px char + 1 gap + 4px padding each side
        txt_h: int = 7 + 8               # 7px font + 4px padding top+bottom
        bw: int = max(w, txt_w, self.MIN_W)
        bh: int = max(h, txt_h, self.MIN_H)
        super().__init__(grid, x, y, bw, bh,
                         key=f'btn_{text}_{x}_{y}')
        self.text: str = text
        self.color_key: str = color
        self.callback: Optional[AgentCallback] = callback
        self._hover: bool = False

        # Wire click
        if callback:
            self._on_click = self._handle_click

    def _handle_click(self, mx: int, my: int, action: str) -> None:
        if self.callback:
            self.callback(mx, my, action)

    def render(self) -> None:
        if not self.visible:
            return

        color: Tuple[int, int, int] = AGENT_COLORS.get(
            self.color_key, AGENT_COLORS['border'])

        # Background
        self.grid.paint_rect(self.x, self.y,
                             self.x + self.w, self.y + self.h,
                             AGENT_COLORS['bg'])
        # Border
        bc: Tuple[int, int, int] = AGENT_COLORS['border_hi'] if self._hover else color
        self.grid.paint_outline(self.x, self.y,
                                self.x + self.w, self.y + self.h, bc)

        # Centered text
        txt_x: int = self.x + (self.w - len(self.text) * 6) // 2
        txt_y: int = self.y + (self.h - 7) // 2
        if txt_x > 0 and txt_y > 0:
            self.grid.paint_text(txt_x, txt_y, self.text,
                                 AGENT_COLORS['text'])

        self._register()


# ═══════════════════════════════════════════════════════════════════════════
# SliderAgent
# ═══════════════════════════════════════════════════════════════════════════


class SliderAgent(CubeAgent):
    """Draggable slider with value label.

    Renders a track (horizontal bar), filled portion, and a handle.
    """

    TRACK_H: int = 6
    HANDLE_R: int = 6
    MIN_W: int = 80

    def __init__(self, grid: PixelGrid, x: int, y: int,
                 label: str = '',
                 min_val: float = 0.0, max_val: float = 1.0,
                 default: float = 0.5,
                 callback: Optional[Callable[[float], None]] = None,
                 w: int = 120) -> None:
        h: int = 30  # fixed height: track + label space
        super().__init__(grid, x, y, w, h,
                         key=f'slider_{label}_{x}_{y}')
        self.label: str = label
        self.min_val: float = min_val
        self.max_val: float = max_val
        self.value: float = default
        self.callback: Optional[Callable[[float], None]] = callback
        self._dragging: bool = False

        self._on_drag = self._handle_drag
        self._on_release = self._handle_release

    @property
    def normalized(self) -> float:
        """Return 0..1 fraction."""
        if self.max_val == self.min_val:
            return 0.5
        return (self.value - self.min_val) / (self.max_val - self.min_val)

    @normalized.setter
    def normalized(self, val: float) -> None:
        clamped: float = max(0.0, min(1.0, val))
        self.value = self.min_val + clamped * (self.max_val - self.min_val)
        if self.callback:
            self.callback(self.value)

    def _handle_drag(self, mx: int, my: int, action: str) -> None:
        self._dragging = True
        rel_x: float = (mx - self.x - 10) / (self.w - 20)
        self.normalized = rel_x

    def _handle_release(self, mx: int, my: int, action: str) -> None:
        self._dragging = False
        rel_x: float = (mx - self.x - 10) / (self.w - 20)
        self.normalized = rel_x

    def render(self) -> None:
        if not self.visible:
            return

        # Track background
        track_y: int = self.y + (self.h - self.TRACK_H) // 2
        self.grid.paint_rect(self.x + 10, track_y,
                             self.x + self.w - 10, track_y + self.TRACK_H,
                             AGENT_COLORS['track'])

        # Filled portion
        fill_w: int = int((self.w - 20) * self.normalized)
        if fill_w > 0:
            self.grid.paint_rect(self.x + 10, track_y,
                                 self.x + 10 + fill_w, track_y + self.TRACK_H,
                                 AGENT_COLORS['fill'])

        # Handle
        hx: int = self.x + 10 + int((self.w - 20) * self.normalized)
        hy: int = track_y + self.TRACK_H // 2
        self.grid.paint_rect(hx - 3, hy - 4,
                             hx + 3, hy + 4,
                             AGENT_COLORS['handle'])

        # Label
        if self.label:
            lbl: str = f'{self.label}'
            self.grid.paint_text(self.x, self.y - 2, lbl, AGENT_COLORS['text'])

        # Value
        val_text: str = f'{self.value:.2f}'
        vx: int = self.x + self.w - len(val_text) * 6 - 5
        vy: int = self.y + self.h - 9
        if vx > 0 and vy > 0:
            self.grid.paint_text(vx, vy, val_text, AGENT_COLORS['text'])

        self._register()


# ═══════════════════════════════════════════════════════════════════════════
# TextLabel
# ═══════════════════════════════════════════════════════════════════════════


class TextLabel(CubeAgent):
    """Static pixel text label (non-interactive)."""

    def __init__(self, grid: PixelGrid, x: int, y: int,
                 text: str = '',
                 color: str = 'text',
                 max_width: int = 0) -> None:
        # Auto-compute height based on text length and max_width
        chars_per_line: int = max(1, max_width // 6) if max_width > 0 else len(text)
        lines: int = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
        h: int = lines * 8 + 4  # 7px font + 1 gap + 2 padding
        w: int = max_width if max_width > 0 else len(text) * 6 + 4
        super().__init__(grid, x, y, w, h,
                         key=f'label_{text[:10]}_{x}_{y}')
        self.text: str = text
        self.color_key: str = color
        self.max_width: int = max_width

    def render(self) -> None:
        if not self.visible:
            return
        color: Tuple[int, int, int] = AGENT_COLORS.get(
            self.color_key, AGENT_COLORS['text'])
        self.grid.paint_text_block(
            self.x + 2, self.y + 2, self.text, color, self.max_width)


# ═══════════════════════════════════════════════════════════════════════════
# AgentManager — holds and renders all agents
# ═══════════════════════════════════════════════════════════════════════════


class AgentManager:
    """Manages a collection of CubeAgents on one PixelGrid.

    render_all() is called each frame by the PixelGrid render loop.
    """

    def __init__(self, grid: PixelGrid) -> None:
        self.grid: PixelGrid = grid
        self.agents: List[CubeAgent] = []

    def add(self, agent: CubeAgent) -> None:
        """Register and add an agent."""
        self.agents.append(agent)

    def remove(self, agent: CubeAgent) -> None:
        """Remove and clean up an agent."""
        agent.remove()
        if agent in self.agents:
            self.agents.remove(agent)

    def remove_by_key(self, key: str) -> None:
        """Remove agent by its key."""
        for a in list(self.agents):
            if a.key == key:
                self.remove(a)
                break

    def clear(self) -> None:
        """Remove all agents."""
        for a in list(self.agents):
            a.remove()
        self.agents.clear()
        # Also clear the grid buffer
        self.grid.clear()

    def render_all(self) -> None:
        """Render all visible agents."""
        for a in self.agents:
            if a.visible:
                try:
                    a.render()
                except Exception:
                    pass  # isolate broken agents

    def find(self, key_fragment: str) -> Optional[CubeAgent]:
        """Find first agent whose key contains fragment."""
        for a in self.agents:
            if key_fragment in a.key:
                return a
        return None
