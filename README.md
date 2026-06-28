# ♢ Hermes Engine — Desktop Particle Simulation Runtime v2

**Hermes Engine** — data-driven particle simulation runtime с AI-ядром.  
Работает как прозрачный overlay поверх всех окон на Windows.  
Общается через локальную LLM (LM Studio).  
Рисует через numpy → PIL или GPU (OpenGL 3.3+).

> **Архитектура:** Мир = данные + системы  
> `World (sim/render/meta) → System Pipeline → Render Graph → Renderer → Экран`

---

## 🚀 Быстрый старт

```bash
git clone https://github.com/kirarud/hermes-cube.git
cd hermes-cube

pip install numpy pillow moderngl

# Запуск новой версии (Engine v2)
python main.py

# Или старая точка входа (совместимость)
python cube_app.py
```

---

## ⚙️ Архитектура

### Мир (core/world.py)

Всё состояние проекта — в одном dataclass-е с тремя зонами:

```
World
├── sim     — симуляция (position, velocity, color, shape_cache)
├── render  — представление (projected_x/y, final_rgb, depth, trails)
└── meta    — управление (config, время, AI, события, ввод)
```

### System Pipeline (core/pipeline.py)

Системы — чистые функции `(World, dt) → None`, выполняемые по порядку:

```
Sim Stage  (fixed dt) — GridGenerator → Morph → Animation → Rotation
FX Stage   (variable) — Trail
View Stage (variable) — Color → Projection
Out Stage  (variable) — RenderSystem (через Render Graph)
```

Ни одна система:
- Не вызывает другие системы
- Не знает про Renderer / Tkinter / UI
- Не создаёт новые частицы напрямую (через active_mask)

### Render Graph (core/render_graph.py)

Data-driven конвейер отрисовки:
- **TrailPass** — шлейф частиц (под кубом)
- **GeometryPass** — сами частицы (точки или символы)
- **Post-processing** — bloom, glow, blur (через core/effects.py)

### GPU (core/gpu.py)

- moderngl (OpenGL 3.3+, NVIDIA/AMD/Intel)
- VBO с позициями частиц (один раз, static draw)
- Vertex shader: rotation via uniform matrix
- PBO read-back → numpy → Tk bridge
- CPU-fallback если GPU недоступен

---

## ✨ Возможности

| Возможность | Описание |
|-------------|----------|
| **3D-куб из частиц** | RGB-градиент, вращение по 3 осям, пульсация |
| **5 форм** | cube, sphere, torus, dna, metaball с плавным морфингом |
| **5 анимаций** | wave, breathe, orbit, geyser, off |
| **AI-общение** | LM Studio (gemma-4-e4b-it), авто-запуск, structured output JSON |
| **Настроение** | mood меняет пульсацию, скорость вращения, цвет куба |
| **Парящие буквы** | ответ AI вылетает буквами с гравитацией |
| **Прозрачный фон** | color-key #000001 + WS_EX_TRANSPARENT (click-through) |
| **Drag** | T/toggle, convex hull drag handle |
| **Настройки** | S/ПКМ — scrollable панель с instant-apply |
| **Трей** | меню: ввод, настройки, трейлы, выход |
| **PixelGrid** | G/toggle — framebuffer для пиксельных агентов |
| **Char mode** | dots, symbols, words, glow — 9 наборов символов |
| **Трейлы** | R/toggle — шлейф с затуханием (12 кадров) |
| **Single-instance** | lock-файл в temp |
| **GPU** | moderngl, OpenGL 3.3, CPU fallback |
| **Эффекты** | Bloom, glow, blur, depth fog |
| **Сохранение** | World snapshot (zip+npz), ReplayBuffer |

---

## 🎮 Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| `s` | Настройки |
| `c` / `C` | AI-ввод |
| `t` / `T` | Режим перемещения |
| `r` / `R` | Трейлы вкл/выкл |
| `g` / `G` | PixelGrid |
| `a` / `A` | Создать агента |
| `Esc` / `q` / `h` | Скрыть |
| ЛКМ + drag | Переместить куб (в режиме drag) |
| ПКМ | Контекстное меню |

---

## 🧱 Структура проекта

```
hermes-cube-repo/
├── main.py                 # Точка входа Engine v2
├── cube_app.py             # Старая точка входа (обратная совместимость)
│
├── core/                   # Ядро Engine
│   ├── world.py            # World — sim/render/meta dataclass
│   ├── pipeline.py         # Pipeline + Stage оркестрация
│   ├── render_graph.py     # Pass-based render pipeline
│   ├── gpu.py              # GPU-рендерер (moderngl)
│   ├── effects.py          # Post-processing (bloom, blur, glow)
│   ├── serialization.py    # Сохранение/загрузка World + Replay
│   ├── ai_constants.py     # AI mood константы
│   └── systems/            # Системы — каждая в своём файле
│       ├── grid_generator.py   # Генерация сетки частиц
│       ├── rotation.py         # 3D-вращение
│       ├── morph.py            # Морфинг форм
│       ├── animation.py        # Анимации частиц
│       ├── color.py            # Depth shading + HSV shift
│       ├── projection.py       # 3D → 2D проекция
│       ├── trail.py            # Шлейф частиц
│       ├── ai.py               # AI-чат (LM Studio)
│       ├── mood.py             # Анализ настроения
│       ├── lm_autostart.py     # Авто-запуск LM Studio
│       ├── text_overlay.py     # Парящие буквы
│       ├── input_window.py     # Окно ввода
│       ├── window.py           # Tkinter-окно
│       ├── input.py            # Клавишный ввод
│       └── drag.py             # Drag-перемещение
│
├── renderer.py             # Бэкенд отрисовки (numpy буфер → Tk canvas)
├── pixel_grid.py           # Пиксельный framebuffer
├── cube_agents.py          # UI-агенты (Button, Slider, TextLabel)
├── particle_agents.py      # Частицы-агенты (cursor, spawner)
├── char_cube.py            # Символьный куб (9 наборов)
├── obsidian_graph.py       # Obsidian граф
├── spatial_depth.py        # Spatial Depth
│
├── tests/                  # Тесты
│   ├── test_systems_vs_cubeengine.py
│   ├── test_pipeline.py
│   └── debug_diff.py
│
├── scripts/                # Скрипты
│   └── progress_watcher.py
│
├── screenshots/            # Скриншоты
└── dist/                   # Сборки .exe
```

---

## 🧠 Как написать свою систему

```python
"""systems/my_system.py — Новая система."""
from core.world import World

def update(world: World, dt: float) -> None:
    """Прочитать world, изменить world. Никаких побочных эффектов."""
    for i in range(world.sim.active_count):
        # Мутируем sim
        world.sim.position[i, 1] += dt * 10.0  # падение вниз
```

Подключить в `build_default_pipeline()` или в `main.py`.

---

## 📊 Производительность

| Конфигурация | ms/frame | FPS |
|---|---|---|
| Pipeline (864 частиц, CPU) | 0.51 | ~2000 |
| GPU (864 частиц, RTX 2070) | <0.1 | ~10000+ |
| Все формы + анимации (CPU) | 0.54 | ~1800 |
| Render Graph (один кадр) | 1.82 | ~550 |

---

## 🔧 Требования

- **ОС:** Windows 10/11 (Linux/macOS не тестировались)
- **Python:** 3.11+
- **GPU (опционально):** OpenGL 3.3+ (RTX 2070 SUPER протестирована)
- **AI (опционально):** LM Studio с gemma-4-e4b-it

## Установка

```bash
pip install numpy pillow moderngl pystray
```

---

## 📄 Лицензия

MIT
