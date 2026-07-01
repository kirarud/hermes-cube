# ♢ Hermes Cube — Desktop Particle Avatar

**GPU-ускоренный 3D-куб из частиц на прозрачном фоне.**  
Работает как overlay поверх всех окон на Windows.  
~68 FPS (600 частиц, Tk PPM) · ~312 FPS (DIB overlay).

---

## 🚀 Быстрый старт

```bash
git clone https://github.com/kirarud/hermes-cube.git
cd hermes-cube

pip install numpy pillow moderngl pystray

python main.py
```

### Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| `s` | Настройки |
| `c` | Ввод текста для AI |
| `t` / `T` | Режим перемещения |
| `r` / `R` | Трейлы вкл/выкл |
| `Esc` / `q` / `h` | Скрыть окно |

---

## ✨ Возможности

| Возможность | Описание |
|-------------|----------|
| **3D-куб из частиц** | RGB-градиент, вращение по 3 осям, пульсация |
| **6 форм** | cube, sphere, torus, dna, metaball, spiral с плавным морфингом |
| **4 анимации** | wave, breathe, orbit, geyser |
| **GPU-рендер** | moderngl batch quads (один draw call, 600+ FPS) |
| **Font atlas** | символы/эмодзи на гранях через GPU текстурный шейдер |
| **9 наборов символов** | default(◆◇●○■□▲△♥♢★☆), hex, binary, blocks, arrows, moods, rus, custom |
| **3 формы точек** | square, circle, dot через GLSL uniform |
| **Трейлы** | затухающий шлейф (12 кадров, GPU) |
| **Прозрачный фон** | color-key #000001 + WS_EX_TRANSPARENT (click-through) |
| **Трей** | меню: показать/скрыть, ввод, трейлы, настройки, выход |
| **AI-общение** | LM Studio (gemma-4-e4b-it), парящие буквы с гравитацией |
| **Single-instance** | lock-файл в temp |
| **Настройки** | S/трей — scrollable панель с instant-apply слайдерами |

---

## ⚙ Архитектура

### World (core/world.py)

```text
World
├── sim     — симуляция (base_position, morphed, animated, world_position, color)
├── render  — представление (projected_x/y, final_rgb, depth, trail_layer)
└── meta    — управление (config, время, AI, события)
```

### Pipeline (core/pipeline.py)

```text
Sim Stage  (fixed)  — GridGenerator → Morph → Animation → Rotation
FX Stage   (var)    — Trail
View Stage (var)    — Color → Projection
Out Stage  (var)    — (зарезервирован)
```

### Рендер-путь

```text
GPU batch quads → FBO → readback → PPM → Tk PhotoImage → Canvas
                          или
GPU batch quads → FBO → readback → DIB → UpdateLayeredWindow
```

### Font Atlas (core/font_atlas.py)

- PIL рендерит символы в текстуру 128×128 (16×16 ячеек по 8px)
- Шейдер выбирает ячейку через `in_char_idx` per-instance
- Один draw call на все частицы, независимо от режима

---

## 🧱 Структура

```
hermes-cube-repo/
├── main.py                     # Точка входа
│
├── core/                       # Ядро
│   ├── world.py                # World — sim/render/meta
│   ├── pipeline.py             # Pipeline + Stage
│   ├── gpu.py                  # GPU-рендерер (moderngl, batch quads)
│   ├── font_atlas.py           # Текстурный атлас символов
│   ├── monitor.py              # FrameMonitor (тайминги + rolling log)
│   ├── render_graph.py         # Pass-based рендер (CPU fallback)
│   └── systems/
│       ├── grid_generator.py   # Генерация сетки куба
│       ├── rotation.py         # 3D-вращение
│       ├── morph.py            # Морфинг (lerp формы)
│       ├── animation.py        # Анимации частиц
│       ├── color.py            # Depth shading + z_layers
│       ├── projection.py       # 3D → 2D + scale + pulse
│       ├── trail.py            # Кольцевой буфер трейлов
│       ├── text_overlay.py     # Парящие буквы
│       ├── input_window.py     # Окно ввода
│       └── gpu_window.py       # Win32 DIB overlay (резерв)
│
├── char_cube.py                # Наборы символов SYMBOL_SETS
├── cube_app.py                 # Старая точка входа (обратная совместимость)
│
├── tests/
├── screenshots/
└── dist/
```

---

## 📊 Производительность

| Конфигурация | ms/frame | FPS |
|---|---|---|
| Pipeline (600 ptcl, CPU) | ~0.3 | ~3000 |
| GPU batch quads (600 ptcl) | <0.1 | ~10000+ |
| Readback + PPM → Tk | ~14 | ~68 |
| Readback + DIB overlay | ~1.5 | ~660 |
| Font atlas chars (600 ptcl) | ~14 | ~68 |

Узкое место: `fbo.read() + PPM → Tk PhotoImage` (~14ms).  
DIB overlay быстрее (~1.5ms), но не работает на всех системах.

---

## 🔧 Требования

- **ОС:** Windows 10/11
- **Python:** 3.11+
- **GPU:** OpenGL 3.3+ (любая дискретная/integrated)
- **AI (опционально):** LM Studio с gemma-4-e4b-it

---

## 📄 Лицензия

MIT
