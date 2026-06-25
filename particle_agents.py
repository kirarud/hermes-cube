#!/usr/bin/env python3
"""particle_agents.py — Agent architecture for Hermes Cube particles.

Each particle in the cube can become a living agent with a role:

  CursorAgent   — flies to input area, bobs, indicates AI thinking
  SpawnerAgent  — multiplies from one particle into many
  MentorAgent   — guides user attention to active UI elements

Architecture:
  ParticleAgent       — base: position, state, behavior, children
  ParticleAgentManager — manages all active particle-agents

Lifecycle:
  1. ParticleAgent.activate(role) — wake from dormant
  2. render() — animate position, draw sprite on PixelGrid
  3. tick() — update position, state transitions
  4. deactivate() — return to cube or fade
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import random

import numpy as np
from numpy.typing import NDArray

from pixel_grid import PixelGrid
from cube_agents import CubeAgent, AgentManager

# ── States ──────────────────────────────────────────────────────────────

STATE_DORMANT: str = 'dormant'
STATE_ACTIVE: str = 'active'
STATE_FLYING: str = 'flying'
STATE_HOVERING: str = 'hovering'
STATE_SPAWNING: str = 'spawning'
STATE_FADING: str = 'fading'

# ── Agent roles ─────────────────────────────────────────────────────────

ROLE_CURSOR: str = 'cursor'
ROLE_SPAWNER: str = 'spawner'
ROLE_MENTOR: str = 'mentor'
ROLE_GUIDE: str = 'guide'


# ═══════════════════════════════════════════════════════════════════════════
# ParticleAgent
# ═══════════════════════════════════════════════════════════════════════════


class ParticleAgent:
    """A single particle that comes alive from the cube."""

    def __init__(self, particle_idx: int,
                 start_x: float, start_y: float,
                 color: Tuple[int, int, int]) -> None:
        self.particle_idx: int = particle_idx
        self.state: str = STATE_DORMANT
        self.role: str = ''

        # Position (screen coords)
        self.x: float = start_x
        self.y: float = start_y
        self.target_x: float = start_x
        self.target_y: float = start_y
        self.vx: float = 0.0
        self.vy: float = 0.0

        # Colour
        self.color: Tuple[int, int, int] = color
        self.alpha: float = 1.0  # 0..1 fade

        # Lifecycle
        self.life: int = 0
        self.max_life: int = 300  # frames
        self._phase_start: int = 0
        self._phase_duration: int = 0

        # Sprite / visual
        self.size: int = 6
        self._glow_phase: float = 0.0

        # Children (for spawners)
        self.children: List[ParticleAgent] = []

    def activate(self, role: str, target_x: float, target_y: float,
                 max_life: int = 300) -> None:
        """Wake up and start flying to target."""
        self.state = STATE_FLYING
        self.role = role
        self.target_x = target_x
        self.target_y = target_y
        self.max_life = max_life
        self.life = 0
        self.alpha = 1.0
        # Compute initial velocity (ease-out prep)
        dx: float = target_x - self.x
        dy: float = target_y - self.y
        dist: float = math.hypot(dx, dy)
        if dist > 0:
            speed: float = min(dist * 0.08, 15.0)
            self.vx = dx / dist * speed
            self.vy = dy / dist * speed
        self._phase_start = 0
        self._phase_duration = max(20, int(dist * 0.5))

    def deactivate(self) -> None:
        """Fade out and return to dormant."""
        self.state = STATE_FADING
        self.alpha = 1.0

    def tick(self) -> None:
        """Update position and state each frame."""
        self.life += 1

        if self.state == STATE_DORMANT:
            return

        # Handle spawning state
        if self.state == STATE_SPAWNING:
            # Children handle themselves
            for c in self.children:
                c.tick()
            # Clean dead children
            self.children = [c for c in self.children if c.state != STATE_DORMANT]
            return

        # Flying → ease-out toward target
        if self.state == STATE_FLYING:
            phase: int = self.life - self._phase_start
            progress: float = min(1.0, phase / max(1, self._phase_duration))
            # Ease-out cubic
            ease: float = 1.0 - (1.0 - progress) ** 3
            self.x = self.x + (self.target_x - self.x) * 0.12
            self.y = self.y + (self.target_y - self.y) * 0.12

            # Check arrival
            if math.hypot(self.target_x - self.x,
                          self.target_y - self.y) < 5.0:
                self.x = self.target_x
                self.y = self.target_y
                self.state = STATE_HOVERING
                self._phase_start = self.life

        # Hovering → bob with sine
        elif self.state == STATE_HOVERING:
            bob: float = math.sin(self.life * 0.1) * 3.0
            self.y = self.target_y + bob
            self._glow_phase = (math.sin(self.life * 0.15) + 1.0) * 0.5

            # Check if time to fade
            remaining: int = self.max_life - self.life
            if remaining < 50:
                self.alpha = max(0.0, remaining / 50.0)
            if remaining <= 0:
                self.deactivate()

        # Fading
        elif self.state == STATE_FADING:
            self.alpha -= 0.05
            if self.alpha <= 0.0:
                self.alpha = 0.0
                self.state = STATE_DORMANT

    def render(self, grid: PixelGrid) -> None:
        """Draw the agent on the PixelGrid."""
        if self.state == STATE_DORMANT or self.alpha <= 0:
            return

        if self.state == STATE_SPAWNING:
            for c in self.children:
                c.render(grid)
            return

        # Render as a coloured rectangle (simple) or sprite
        ix: int = int(self.x)
        iy: int = int(self.y)
        half: int = self.size // 2

        r: int = int(self.color[0] * self.alpha)
        g: int = int(self.color[1] * self.alpha)
        b: int = int(self.color[2] * self.alpha)

        # Glow effect in hovering state
        glow: float = 1.0
        if self.state == STATE_HOVERING:
            glow = 0.7 + 0.3 * self._glow_phase
            # Draw a glow ring (extra pixels around)
            glow_size: int = half + 2
            grid.paint_rect(ix - glow_size, iy - glow_size,
                            ix + glow_size, iy + glow_size,
                            (int(r * 0.3), int(g * 0.3), int(b * 0.3)))

        r = min(255, int(r * glow))
        g = min(255, int(g * glow))
        b = min(255, int(b * glow))

        grid.paint_rect(ix - half, iy - half,
                        ix + half, iy + half,
                        (r, g, b))


# ═══════════════════════════════════════════════════════════════════════════
# ParticleAgentManager
# ═══════════════════════════════════════════════════════════════════════════


class ParticleAgentManager:
    """Manages all active particle agents (born from cube particles)."""

    def __init__(self, grid: PixelGrid) -> None:
        self.grid: PixelGrid = grid
        self.agents: List[ParticleAgent] = []

    def spawn(self, particle_idx: int,
              start_x: float, start_y: float,
              color: Tuple[int, int, int],
              role: str, target_x: float, target_y: float,
              max_life: int = 300) -> ParticleAgent:
        """Create and activate a new particle agent."""
        agent = ParticleAgent(particle_idx, start_x, start_y, color)
        agent.activate(role, target_x, target_y, max_life)
        self.agents.append(agent)
        return agent

    def spawn_spawner(self, particle_idx: int,
                      x: float, y: float,
                      color: Tuple[int, int, int],
                      count: int = 5) -> ParticleAgent:
        """Create a SpawnerAgent that multiplies."""
        agent = ParticleAgent(particle_idx, x, y, color)
        agent.state = STATE_SPAWNING
        agent.role = ROLE_SPAWNER
        # Create children
        for i in range(count):
            child = ParticleAgent(
                particle_idx,
                x + random.uniform(-20, 20),
                y + random.uniform(-20, 20),
                color,
            )
            child.activate(
                ROLE_CURSOR,
                x + random.uniform(-100, 100),
                y + random.uniform(-100, 100),
                max_life=200,
            )
            agent.children.append(child)
        self.agents.append(agent)
        return agent

    def update_all(self) -> None:
        """Tick all agents and remove dead ones."""
        for a in self.agents:
            a.tick()
        self.agents = [a for a in self.agents if a.state != STATE_DORMANT]

    def render_all(self) -> None:
        """Render all active agents."""
        for a in self.agents:
            try:
                a.render(self.grid)
            except Exception:
                pass

    def clear(self) -> None:
        """Deactivate all agents."""
        self.agents.clear()
