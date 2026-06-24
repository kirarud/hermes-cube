# 🎨 PixelGrid — Фреймбуфер агентов

## Концепция

Вместо tkinter-виджетов — один numpy-буфер (H×W×4), где агенты рисуют пиксели напрямую.

## Архитектура (проект)

```python
class PixelGrid:
    buffer = np.zeros((1080, 1920, 4), dtype=np.uint8)  # RGB + alpha
    zbuffer = np.zeros((1080, 1920), dtype=np.float32)
    hit_zones = {}  # (x1,y1,x2,y2) → agent_ref

    def paint(x, y, r, g, b, a=255)     — пиксель
    def paint_rect(x1,y1,x2,y2, color)  — заливка
    def paint_sprite(x,y, matrix, pal)  — спрайт
    def render_to_canvas(canvas)         — blit на Canvas
```

## Что даёт PixelGrid

| Концепт | Традиционный UI | PixelGrid |
|---------|----------------|-----------|
| Кнопка | Button виджет | Прямоугольник пикселей + hit-test |
| Текст | Label виджет | Bitmap-шрифт 5×7 |
| Смайлик | Emoji в label | Матрица пикселей 16×16 |
| 3D-сцена | OpenGL | Проекция в пиксели |
| Анимация | .after() | Покадровая смена буфера |

## Преимущества

- **Один Canvas** вместо десятков виджетов
- **1000+ объектов** без потери FPS
- **Пиксельная свобода** — любой интерфейс
- **3D-проекция** — куб может рисовать свои проекции

## Статус

🚧 В разработке — ещё не реализован
