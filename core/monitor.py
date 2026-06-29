#!/usr/bin/env python3
"""core/monitor.py — Полноценная система мониторинга и логирования.

Содержание:
  - FrameMonitor: кольцевой буфер таймингов + лог на диск
  - Визуализация: FPS, per-section бары, bbox, конфиг, алерты
  - Логирование: rolling file (~/.hermes/cube_monitor.log)

Использование в main.py:
    monitor = FrameMonitor()
    ...
    _t0 = time.perf_counter_ns()
    ... pipeline ...
    _t1 ...
    monitor.log_frame(fps=..., pipeline_us=..., sort_us=..., render_us=..., ...)
    monitor.draw(canvas, w, h)
"""

from __future__ import annotations

import os
import time
import json
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Rolling file logger
# ═══════════════════════════════════════════════════════════════════════════

_LOG_DIR: str = os.path.join(
    os.environ.get('APPDATA', os.path.expanduser('~')),
    'HermesCube',
)
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE: str = os.path.join(_LOG_DIR, 'monitor.log')

# Максимум строк в логе (rolling — дописываем пока не превысит, потом обрезаем)
_MAX_LOG_LINES: int = 20000


def _log_to_file(entry: Dict[str, Any]) -> None:
    """Дописать одну запись в лог. Если превышен лимит — обрезать половину."""
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    try:
        with open(_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line)
    except OSError:
        pass  # дисковая ошибка — не падаем


def _trim_log() -> None:
    """Обрезать лог до _MAX_LOG_LINES строк, сохранив последние записи."""
    try:
        with open(_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) > _MAX_LOG_LINES:
            keep = lines[-_MAX_LOG_LINES:]
            with open(_LOG_FILE, 'w', encoding='utf-8') as f:
                f.writelines(keep)
    except (OSError, FileNotFoundError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# FrameMonitor
# ═══════════════════════════════════════════════════════════════════════════


class FrameMonitor:
    """Собирает тайминги каждого кадра, рисует HUD, пишет лог.

    Хранит кольцевой буфер последних FRAME_HISTORY кадров.
    Выводит на canvas: FPS, per-section гистограммы, bbox, конфиг, алерты.
    """

    # ── Настройки монитора ────────────────────────────────────────────
    FRAME_HISTORY: int = 90        # сколько кадров храним в буфере
    LOG_INTERVAL_S: float = 5.0    # каждые N секунд пишем в файл
    ALERT_THRESHOLD_US: int = 2000  # предупреждение если секция > 2ms
    FPS_REFRESH_S: float = 0.5     # обновление fps-метрик

    # ── Цвета HUD ─────────────────────────────────────────────────────
    COLOR_PIPELINE: str = '#22c55e'   # зелёный
    COLOR_SORT: str = '#6366f1'       # индиго
    COLOR_RENDER: str = '#f59e0b'     # золото
    COLOR_TOTAL: str = '#ec4899'      # розовый
    COLOR_ALERT: str = '#ef4444'      # красный
    COLOR_OK: str = '#e94560'         # акцент (UI_ACCENT)
    COLOR_BG: str = '#1a1a2e'
    COLOR_FG: str = '#e0e0e0'

    def __init__(self) -> None:
        # ── Ring buffer ─────────────────────────────────────────────
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.FRAME_HISTORY)

        # ── Текущие значения ─────────────────────────────────────────
        self.fps: float = 0.0
        self.pipeline_us: float = 0.0
        self.sort_us: float = 0.0
        self.render_us: float = 0.0
        self.total_us: float = 0.0
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.bbox_w: int = 0
        self.bbox_h: int = 0
        self.n_particles: int = 0
        self.config: Dict[str, Any] = {}

        # ── FPS расчёт ──────────────────────────────────────────────
        self._fps_frame_count: int = 0
        self._fps_last_time: float = time.perf_counter()

        # ── Логирование ──────────────────────────────────────────────
        self._last_log_time: float = 0.0
        self._line_count_since_trim: int = 0
        self._log_meta: Dict[str, Any] = {}

        # ── Тик (для затухания алертов) ──────────────────────────────
        self._alert_visible: bool = False
        self._alert_text: str = ''

    # ── Логирование одного кадра ──────────────────────────────────────

    def log_frame(
        self,
        fps: float,
        pipeline_us: float,
        sort_us: float,
        render_us: float,
        total_us: float,
        n_particles: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Записать один кадр: сохранить в буфер и, если пора, на диск."""
        entry: Dict[str, float] = {
            'fps': fps,
            'pipeline': pipeline_us,
            'sort': sort_us,
            'render': render_us,
            'total': total_us,
            'n': n_particles,
            't': time.time(),
        }
        if bbox:
            entry['bx0'] = bbox[0]
            entry['by0'] = bbox[1]
            entry['bx1'] = bbox[2]
            entry['by1'] = bbox[3]

        self.history.append(entry)

        # Обновить текущие значения
        self.fps = fps
        self.pipeline_us = pipeline_us
        self.sort_us = sort_us
        self.render_us = render_us
        self.total_us = total_us
        self.n_particles = n_particles
        self.bbox = bbox
        if bbox:
            self.bbox_w = bbox[2] - bbox[0]
            self.bbox_h = bbox[3] - bbox[1]
        if config:
            self.config = config

        # Алерты
        pct = 100.0 * total_us / (FRAME_MS * 1000) if total_us > 0 else 0
        self._alert_visible = pct > 80  # warning при > 80% frame budget
        if self._alert_visible:
            culprits = []
            if pipeline_us > self.ALERT_THRESHOLD_US:
                culprits.append(f'pip {pipeline_us/1000:.1f}ms')
            if sort_us > self.ALERT_THRESHOLD_US:
                culprits.append(f'sort {sort_us/1000:.1f}ms')
            if render_us > self.ALERT_THRESHOLD_US:
                culprits.append(f'rdr {render_us/1000:.1f}ms')
            self._alert_text = f'⚠ {total_us/1000:.1f}ms ({pct:.0f}%)'
            if culprits:
                self._alert_text += '\n  ' + ', '.join(culprits)

        # FPS счётчик
        self._fps_frame_count += 1

        # Лог на диск (раз в LOG_INTERVAL_S)
        now = time.perf_counter()
        if now - self._last_log_time >= self.LOG_INTERVAL_S:
            self._flush_to_disk()
            self._last_log_time = now

    def set_meta(self, **kwargs: Any) -> None:
        """Задать мета-данные для следующего лога (версия, окно, резолюция)."""
        self._log_meta.update(kwargs)

    # ── Запись на диск ────────────────────────────────────────────────

    def _flush_to_disk(self) -> None:
        """Усреднить history за последнюю секунду и дописать в лог."""
        if not self.history:
            return
        # Берём последние 60 кадров (или сколько есть)
        recent = list(self.history)[-min(60, len(self.history)):]
        avg: Dict[str, float] = {
            'fps': sum(f['fps'] for f in recent) / len(recent),
            'pipeline': sum(f['pipeline'] for f in recent) / len(recent),
            'sort': sum(f['sort'] for f in recent) / len(recent),
            'render': sum(f['render'] for f in recent) / len(recent),
            'total': sum(f['total'] for f in recent) / len(recent),
            'n': max(f['n'] for f in recent),
            't': recent[-1]['t'],
        }
        avg.update(self._log_meta)

        # bbox из последнего
        if self.bbox:
            avg['bx0'] = self.bbox[0]
            avg['by0'] = self.bbox[1]
            avg['bx1'] = self.bbox[2]
            avg['by1'] = self.bbox[3]
            avg['bw'] = self.bbox_w
            avg['bh'] = self.bbox_h

        _log_to_file(avg)
        self._line_count_since_trim += 1
        if self._line_count_since_trim >= (_MAX_LOG_LINES // 4):
            _trim_log()
            self._line_count_since_trim = 0

    # ── Отрисовка HUD на canvas ───────────────────────────────────────

    def draw(self, canvas: Any, w: int, h: int) -> None:
        """Нарисовать HUD: fps, бары, bbox, алерты. Вызывать каждый кадр."""
        self._draw_stats_text(canvas, w, h)
        self._draw_timing_bars(canvas, h)
        self._draw_alert(canvas, w)

    def _draw_stats_text(self, canvas: Any, w: int, h: int) -> None:
        """Левая колонка: fps, частицы, bbox, конфиг."""
        lines: List[str] = []

        # FPS
        fps_color = self.COLOR_OK if self.fps > 20 else self.COLOR_ALERT
        lines.append(f'FPS: {self.fps:.0f}')

        # Частицы
        lines.append(f'ptcl: {self.n_particles}')

        # Тайминги
        lines.append(f'pipeline: {self.pipeline_us:.0f} µs')
        lines.append(f'sort:     {self.sort_us:.0f} µs')
        lines.append(f'render:   {self.render_us:.0f} µs')
        lines.append(f'total:    {self.total_us:.0f} µs  ({self.total_us/1000:.1f} ms)')

        # Bbox
        if self.bbox:
            bw = self.bbox[2] - self.bbox[0]
            bh = self.bbox[3] - self.bbox[1]
            lines.append(f'bbox: {bw}×{bh}')
            pct_w = 100.0 * bw / w if w > 0 else 0
            pct_h = 100.0 * bh / h if h > 0 else 0
            lines.append(f'      ({pct_w:.0f}% × {pct_h:.0f}% экрана)')

        # Конфиг (самое важное)
        if self.config:
            scale = self.config.get('cube_scale', 0.27)
            cell = self.config.get('cell_size', 6)
            density = self.config.get('particle_density', 12)
            shape = self.config.get('shape_preset', 'cube')
            anim = self.config.get('particle_mode', 'off')
            lines.append(f'scale={scale:.2f} cell={cell} den={density}')
            lines.append(f'{shape} {anim}')

        # Рисуем
        y = 8
        for i, line in enumerate(lines):
            color = fps_color if i == 0 else self.COLOR_FG
            canvas.create_text(
                8, y, anchor='nw', text=line,
                fill=color, font=('Consolas', 10), tags='monitor_hud',
            )
            y += 15

    def _draw_timing_bars(self, canvas: Any, h: int) -> None:
        """Правая колонка: вертикальная гистограмма таймингов."""
        if not self.history:
            return

        bar_w = 4
        gap = 1
        total_w = min(200, (bar_w + gap) * self.FRAME_HISTORY)
        x0 = 8
        y_base = h - 60

        # Группируем: каждый 3-й кадр для читаемости
        recent = list(self.history)
        step = max(1, len(recent) // 60)

        # Нормализуем высоту: макс = 45px
        max_us = max(f['total'] for f in recent) or 1

        for i, entry in enumerate(recent[::step]):
            x = x0 + i * (bar_w + gap)
            pip_h = max(1, int(entry['pipeline'] / max_us * 40))
            srt_h = max(1, int(entry['sort'] / max_us * 40))
            rdr_h = max(1, int(entry['render'] / max_us * 40))

            # Рендер (нижняя часть)
            y_b = y_base
            canvas.create_rectangle(
                x, y_b - rdr_h, x + bar_w, y_b,
                fill=self.COLOR_RENDER, outline='', tags='monitor_hud',
            )
            y_b -= rdr_h
            # Сорт
            canvas.create_rectangle(
                x, y_b - srt_h, x + bar_w, y_b,
                fill=self.COLOR_SORT, outline='', tags='monitor_hud',
            )
            y_b -= srt_h
            # Pipeline
            canvas.create_rectangle(
                x, y_b - pip_h, x + bar_w, y_b,
                fill=self.COLOR_PIPELINE, outline='', tags='monitor_hud',
            )

        # Label
        canvas.create_text(
            x0, y_base + 4, anchor='nw',
            text=f'{max_us/1000:.1f}ms',
            fill='#888', font=('Consolas', 8), tags='monitor_hud',
        )

    def _draw_alert(self, canvas: Any, w: int) -> None:
        """Красный баннер при превышении frame budget."""
        if self._alert_visible:
            lines = self._alert_text.split('\n')
            canvas.create_rectangle(
                w // 2 - 120, 20, w // 2 + 120, 20 + 18 * len(lines),
                fill=self.COLOR_ALERT, outline='', stipple='gray50',
                tags='monitor_hud',
            )
            for i, line in enumerate(lines):
                canvas.create_text(
                    w // 2, 26 + 16 * i, anchor='center', text=line,
                    fill='#fff', font=('Consolas', 11, 'bold'),
                    tags='monitor_hud',
                )

    def clear_canvas(self, canvas: Any) -> None:
        """Удалить все элементы монитора с canvas."""
        canvas.delete('monitor_hud')


# ═══════════════════════════════════════════════════════════════════════════
# Константы (импортируются из main.py)
# ═══════════════════════════════════════════════════════════════════════════

# FRAME_MS для расчёта процента budget
try:
    from cube_app import FRAME_MS as _FRAME_MS
except ImportError:
    _FRAME_MS = 42
FRAME_MS: int = _FRAME_MS
