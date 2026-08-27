# ♢ Hermes Cube — North Star

**Выбрано:** Companion — живой собеседник на рабочем столе.

## Смысл (одно предложение)

> **Один живой аватар на Windows, с которым можно поговорить, который реагирует телом (пульс, цвет, текст) и постепенно узнаёт тебя.**

## Критерий «готово» (минимальный продукт)

1. Запуск: `python main.py`
2. Клавиша **C** → сообщение → ответ (LM Studio или явный offline-fallback)
3. Куб **меняет настроение** (пульс / скорость)
4. Ответ виден: **парящие буквы** и/или **текст формой куба**
5. Смена density / resize окна **не роняет** приложение
6. Ты открываешь куб, чтобы **поговорить**, а не «чтобы пофиксить»

## Делаем сейчас (фокус)

- [x] Один путь: `main.py`
- [x] Chat → mood → letters + avatar text
- [x] Mood params → config (pulse / rotation)
- [x] Offline fallback без LM Studio
- [x] Безопасный resize density
- [x] Мини-память (`%APPDATA%\HermesCube\memory.md`)
- [ ] Стабильный smoke-тест 5+ минут (ручной)

## Не делаем до MM (parked)

| Область | Почему отложено |
|---------|-----------------|
| PixelGrid / CubeAgents | Вторая подсистема UI |
| ParticleAgents (cursor/spawner) | Вторая «игра» |
| Obsidian graph overlay | Knowledge body — следующий слой |
| Spatial Depth fancy | Визуал без Companion-цикла |
| PyOpenGL / DIB / 300 FPS | Оптимизация без замкнутого UX |
| Новые формы / trails-фичи | Шум |

См. также: [LEGACY.md](LEGACY.md)

## Следующий слой (после MM)

Только **один** из:

1. Память глубже (Obsidian read-only)  
2. Hermes Agent → HTTP API как пульт  
3. Obsidian graph v1  

Не все сразу.
