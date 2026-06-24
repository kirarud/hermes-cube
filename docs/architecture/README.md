# 🏗 Architecture — Hermes Cube

## Трёхслойная архитектура (на текущий момент)

```
┌─────────────────────────────────────────────┐
│  1. CubeWindow (overlay)                    │
│     - Tk() with transparentcolor            │
│     - Canvas для частиц куба               │
│     - Размер: 400×400 (настраивается)       │
│     - overrideredirect, topmost             │
│     - drag, tray icon, контекстное меню     │
├─────────────────────────────────────────────┤
│  2. TextOverlay (полноэкранный)             │
│     - Toplevel с transparentcolor           │
│     - Полный экран (1920×1080)              │
│     - Буквы, агенты, спрайты                │
│     - Скрыт (withdraw) когда нет контента   │
├─────────────────────────────────────────────┤
│  3. InputWindow (отдельное окно)            │
│     - Toplevel с тёмным фоном               │
│     - Внизу экрана (центр)                  │
│     - overrideredirect, topmost             │
│     - Enter → отправка, Escape → закрыть    │
└─────────────────────────────────────────────┘
```

## Поток данных

```
Клавиша C → show_input → InputWindow появляется
  → пользователь печатает → Enter
    → InputWindow закрывается
    → ai_chat(text) → LM Studio API
      → analyze_mood() → куб меняет цвет/пульсацию
      → spawn_text_particles() → буквы на TextOverlay
        → буквы летят из куба в строку
```

## Проблемы которые решили

1. `overrideredirect + transparentcolor` — окно невидимо на Windows. Решение: без overrideredirect нормальная рамка.
2. `-disabled` на Toplevel — ломает event loop. Решение: не использовать.
3. `-topmost` на текстовом overlay — перекрывает куб. Решение: показать overlay только когда есть контент.
4. Текст за границами куба — буквы не видны. Решение: TextOverlay — отдельное полноэкранное окно.
