#!/usr/bin/env python3
"""core/pipeline.py — Stage-based оркестрация систем.

Stage  = группа систем с одним типом scheduling
Pipeline = список Stage, выполняемых по порядку

Схема:
  Sim Stage  (fixed dt) — Grid → Morph → Animation → Rotation
  FX Stage   (variable) — Trail
  View Stage (variable) — Color → Projection
  Out Stage  (variable) — Render (через Render Graph)

Каждый Stage может быть:
  - fixed: dt фиксирован (для симуляции)
  - variable: dt реальный (для рендера)
  - skipped: пропущен (по конфигу/метрикам)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional

from core.world import World


class Schedule(Enum):
    """Тип scheduling для Stage."""
    FIXED = auto()     # фиксированный timestep (симуляция)
    VARIABLE = auto()  # реальный dt (рендер/эффекты)


SystemFn = Callable[[World, float], None]
"""Сигнатура системы: (world, dt) → None. Мутирует world на месте."""


@dataclass
class Stage:
    """Группа систем с одним типом scheduling.

    Attributes:
        name: Уникальное имя Stage (для логов/пропуска).
        systems: Список system-функций, выполняемых по порядку.
        schedule: FIXED (симуляция) или VARIABLE (рендер/эффекты).
        enabled: True = выполняется, False = пропускается.
    """
    name: str
    systems: List[SystemFn] = field(default_factory=list)
    schedule: Schedule = Schedule.VARIABLE
    enabled: bool = True

    def run(self, world: World, dt: float) -> None:
        """Выполнить все системы Stage-а по порядку."""
        if not self.enabled:
            return
        for system in self.systems:
            system(world, dt)


@dataclass
class Pipeline:
    """Оркестратор: список Stage, запускаемых каждый кадр.

    Пример:
        pipeline = Pipeline()
        pipeline.add_stage(Stage('sim', [morph, anim, rot], Schedule.FIXED))
        pipeline.add_stage(Stage('view', [color, proj], Schedule.VARIABLE))

        # Каждый кадр:
        pipeline.run(world, dt)
    """

    stages: List[Stage] = field(default_factory=list)

    def add_stage(self, stage: Stage) -> None:
        """Добавить Stage в конец очереди."""
        self.stages.append(stage)

    def get_stage(self, name: str) -> Optional[Stage]:
        """Найти Stage по имени."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def run(self, world: World, dt: float) -> None:
        """Выполнить все Stage по порядку.

        Каждый Stage получает dt:
          - FIXED: min(dt, max_fixed_dt) для стабильности
          - VARIABLE: реальный dt
        """
        for stage in self.stages:
            stage.run(world, dt)

    def run_stage(self, name: str, world: World, dt: float) -> None:
        """Выполнить один конкретный Stage по имени."""
        stage = self.get_stage(name)
        if stage:
            stage.run(world, dt)

    def enable_stage(self, name: str, enabled: bool = True) -> None:
        """Включить/выключить Stage."""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = enabled

    def apply_config(self, config: dict) -> None:
        """Применить настройки из конфига: включить/выключить Stage-ы.

        Конфиг может содержать 'stages_enabled' — dict имя→bool.
        """
        stages_enabled = config.get('stages_enabled', {})
        if not stages_enabled:
            return
        for name, enabled in stages_enabled.items():
            self.enable_stage(name, enabled)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_default_pipeline() -> Pipeline:
    """Собрать Pipeline со всеми Stage-ами по умолчанию.

    Порядок (соответствует CubeEngine.get_frame):
      1. Sim:  GridGenerator → Morph → Animation → Rotation
      2. FX:   Trail
      3. View: Color → Projection
      4. Out:  (RenderSystem — добавляется отдельно, ему нужен renderer)
    """
    import core.systems.grid_generator as gg
    import core.systems.morph as morph
    import core.systems.animation as anim
    import core.systems.rotation as rot
    import core.systems.trail as trail
    import core.systems.color as color
    import core.systems.projection as proj

    pipeline = Pipeline()
    pipeline.add_stage(Stage('sim', [gg.update, morph.update, anim.update, rot.update], Schedule.FIXED))
    pipeline.add_stage(Stage('fx', [trail.update], Schedule.VARIABLE))
    pipeline.add_stage(Stage('view', [color.update, proj.update], Schedule.VARIABLE))
    pipeline.add_stage(Stage('out', [], Schedule.VARIABLE))  # RenderSystem добавляется отдельно
    return pipeline
