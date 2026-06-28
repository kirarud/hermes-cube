# ♢ Hermes Engine v2 — Release Notes

**Тэг:** v2.0  
**Дата:** 2026-06-28  
**Engine Blueprint v2:** 100/100 шагов

---

## Архитектура

```
World (sim/render/meta)
  → Pipeline (Sim → FX → View → Out)
    → Render Graph (Trails → Geometry → Effects)
      → Renderer (numpy → Tk canvas)
```

## Stage-буферы

```
base_position  ← GridGenerator (read-only)
  → morphed    ← Morph (lerp к target форме)
    → animated ← Animation (wave/breathe/orbit/geyser)
      → world_position ← Rotation (3 оси, финальная 3D)
```

Каждая система читает из **входного** буфера, пишет в **свой**.  
Ни одной `.copy()` в hot path. Все верифицированы против CubeEngine — **diff 0.0**.

## Что вошло

| Компонент | Статус |
|-----------|--------|
| World (sim/render/meta) | ✅ |
| Pipeline + Stage scheduling | ✅ |
| 7 simulation systems | ✅ |
| 4 AI systems (chat, mood, autostart, overlay) | ✅ |
| Render Graph (TrailPass, GeometryPass) | ✅ |
| GPU (moderngl preview, RTX 2070 SUPER) | 🔜 |
| Effects (bloom, blur, glow) | ✅ |
| World serialization (save/load/replay) | ✅ |
| main.py — новая точка входа | ✅ |
| cube_app.py — обратная совместимость | ✅ |
| Performance: 0.55 ms/frame (864 particles) | ✅ |

## Как установить

```bash
git clone https://github.com/kirarud/hermes-cube.git
cd hermes-cube
pip install numpy pillow moderngl

# Новая версия
python main.py

# Старая версия (совместимость)
python cube_app.py
```

## Создание Release на GitHub

1. Открыть https://github.com/kirarud/hermes-cube/releases/new
2. Выбрать тэг `v2.0`
3. Вставить содержимое этого файла как Release Notes
4. Опубликовать
