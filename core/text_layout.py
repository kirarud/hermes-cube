#!/usr/bin/env python3
"""core/text_layout.py — Раскладка текста в позиции частиц.

Преобразует строку текста в массив позиций и индексов символов
для рендера через font atlas.

Все позиции — в 3D пространстве [-1, 1].
Каждый символ = одна частица.
Моноширинная раскладка: строки × колонки.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Параметры раскладки ─────────────────────────────────────────────

# Максимальная ширина текста в единицах [-1,1] (0.0 = auto-fit)
MAX_WIDTH: float = 0.0  # 0 = растянуть до 0.95 в обе стороны

# Отступы между строками (в единицах [-1,1])
LINE_HEIGHT: float = 0.14

# Расстояние между символами (динамическое — подгоняется под ширину)
CHAR_SPACING: float = 0.0  # 0 = авто-подгонка

# Максимальная ширина строки в долях [-1,1]
FILL_WIDTH: float = 0.85

# Предельное количество строк (текст обрезается)
MAX_LINES: int = 6

# Z-позиция для неиспользуемых частиц (прячутся глубоко сзади)
HIDDEN_Z: float = 5.0


def layout_text(text: str, n_particles: int) -> Tuple[NDArray[np.float64],
                                                       NDArray[np.int32],
                                                       int]:
    """Сгенерировать раскладку текста.

    Args:
        text: строка для отображения (очищается от JSON-мусора)
        n_particles: общее количество частиц в системе

    Returns:
        positions: (n_particles, 3) float64 — позиции каждой частицы
        char_indices: (n_particles,) int32 — индекс символа в font atlas
        n_used: количество частиц, занятых текстом
    """
    # Чистим текст
    clean = _clean_text(text)

    # Разбиваем на символы
    chars = list(clean)
    n_chars = len(chars)

    if n_chars == 0:
        # Пустой текст — все частицы на месте, без изменений
        positions = np.zeros((n_particles, 3), dtype=np.float64)
        char_indices = np.zeros(n_particles, dtype=np.int32)
        return positions, char_indices, 0

    # Auto-fit: считаем колонки и строки
    n_cols, n_rows, lines = _fit_lines(chars, n_chars)

    # Динамический CHAR_SPACING: растягиваем текст на FILL_WIDTH
    if CHAR_SPACING > 0:
        spacing = CHAR_SPACING
    elif n_cols > 1:
        spacing = FILL_WIDTH * 2.0 / (n_cols - 1)
    else:
        spacing = FILL_WIDTH * 2.0

    # Центрируем блок текста
    block_w = n_cols * spacing
    block_h = n_rows * LINE_HEIGHT
    offset_x = -block_w / 2.0 + spacing / 2.0
    offset_y = block_h / 2.0 - LINE_HEIGHT / 2.0

    # Генерируем позиции
    positions = np.zeros((n_particles, 3), dtype=np.float64)
    char_indices = np.zeros(n_particles, dtype=np.int32)

    # Маппинг символа → индекс в font atlas
    char_map = _build_char_to_idx()

    idx = 0
    for row_idx, row_chars in enumerate(lines):
        if idx >= n_particles:
            break
        for col_idx, ch in enumerate(row_chars):
            if idx >= n_particles:
                break
            x = offset_x + col_idx * spacing
            y = offset_y - row_idx * LINE_HEIGHT
            z = 0.0

            positions[idx] = [x, y, z]
            # Пробелы получают индекс 0 (◆) — шейдер их отрисует,
            # но с низким cell_size они почти невидимы
            ci = char_map.get(ch, 0)
            char_indices[idx] = ci
            idx += 1

    n_used = idx

    # Оставшиеся частицы — убираем глубоко за текст
    if n_used < n_particles:
        positions[n_used:, :] = 0.0
        positions[n_used:, 2] = HIDDEN_Z
        char_indices[n_used:] = 0

    # Добавляем лёгкий Z-разброс для визуальной глубины
    rng = np.random.default_rng(42)
    z_jitter = rng.uniform(-0.02, 0.02, n_used)
    positions[:n_used, 2] += z_jitter

    return positions, char_indices, n_used


def _clean_text(text: str) -> str:
    """Очистить текст от управляющих символов, лишних пробелов."""
    # Убираем всё что не буква/цифра/пробел/пунктуация
    import re
    # Оставляем: буквы Unicode, цифры, пробелы, основные знаки
    cleaned = re.sub(r'[^\w\s.,!?—…:;\'"()\[\]{}@#$%^&*+=<>/~`|\\-]', '', text)
    # Схлопываем множественные пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _fit_lines(chars: List[str], n_chars: int) -> Tuple[int, int, List[List[str]]]:
    """Разбить символы на строки с авто-переносом.

    Returns:
        (n_cols, n_rows, lines) где lines = список строк (каждая список символов)
    """
    if n_chars == 0:
        return 0, 0, []

    # Вычисляем оптимальную ширину
    # Пытаемся найти n_cols такое, чтобы текст поместился в ~5-6 строк
    # или в одну строку если короткий
    if n_chars <= 20:
        # Короткий текст — в одну строку
        return n_chars, 1, [chars]

    # Длинный текст — вычисляем количество колонок
    # Цель: отношение n_cols / n_rows ≈ 2..4 (широкий, не квадрат)
    # n_cols * n_rows ≈ n_chars
    # n_cols ≈ 4 * n_rows → 4 * n_rows² ≈ n_chars → n_rows ≈ sqrt(n_chars / 4)
    ideal_rows = max(1, int(np.sqrt(n_chars / 4.0)))
    n_cols = max(10, int(np.ceil(n_chars / ideal_rows)))
    n_cols = min(n_cols, n_chars)

    # Перенос слов
    words = _split_words(chars)
    lines: List[List[str]] = []
    current_line: List[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if current_len + word_len + (1 if current_len > 0 else 0) > n_cols:
            if current_line:
                lines.append(current_line)
            current_line = list(word)
            current_len = word_len
        else:
            if current_len > 0:
                current_line.append(' ')
                current_len += 1
            current_line.extend(word)
            current_len += word_len

    if current_line:
        lines.append(current_line)

    # Обрезаем по MAX_LINES
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        # Добавляем '…' в конец последней строки
        if lines[-1]:
            lines[-1].append('…')

    # Финальные размеры
    max_cols = max(len(line) for line in lines) if lines else 0
    return max_cols, len(lines), lines


def _split_words(chars: List[str]) -> List[List[str]]:
    """Разбить список символов на слова (по пробелам)."""
    words: List[List[str]] = []
    current: List[str] = []
    for ch in chars:
        if ch == ' ' or ch == '\t':
            if current:
                words.append(current)
                current = []
        else:
            current.append(ch)
    if current:
        words.append(current)
    return words


def _build_char_to_idx() -> Dict[str, int]:
    """Построить маппинг символ → индекс в font atlas.

    Использует тот же алгоритм что и font_atlas.build_atlas(),
    чтобы индексы были согласованы.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from char_cube import SYMBOL_SETS

    char_map: Dict[str, int] = {}
    GRID = 16
    max_slots = GRID * GRID  # 256

    for name, syms in SYMBOL_SETS.items():
        for ch in syms:
            if ch not in char_map and len(char_map) < max_slots:
                char_map[ch] = len(char_map)

    # Буквы/цифры/знаки которых нет в SYMBOL_SETS — добавляем
    extra = (
        list('abcdefghijklmnopqrstuvwxyz') +
        list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
        list('0123456789') +
        list(".,!?\u2014\u2026:;'\"()[]{}@#$%^&*+=<>/~`|\\-") +
        # Строчные русские (в SYMBOL_SETS['rus'] только заглавные)
        list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    )
    for ch in extra:
        if ch not in char_map and len(char_map) < max_slots:
            char_map[ch] = len(char_map)

    return char_map


def get_text_scale_override() -> float:
    """Коэффициент масштаба для текстового режима (вместо cube_scale).

    Возвращает временный cube_scale для текстового режима.
    Текст должен заполнять ~70-80% экрана.
    """
    return 0.65
