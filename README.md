# ♢ Hermes Cube

Десктопный аватар-куб из частиц на прозрачном фоне. Работает как overlay поверх всех окон на Windows.

![Hermes Cube Preview](screenshots/preview.png)

## ✨ Возможности

- 3D-куб из частиц 64×64 с RGB-градиентом
- Плавная пульсация и вращение по 3 осям
- Прозрачный фон — виден только куб
- Перетаскивание мышкой (ЛКМ)
- Real-time настройки (клавиша `s` или ПКМ)
- Системный трей с меню
- Автозапуск при старте Windows

## 🚀 Установка

**Вариант 1 — установщик (рекомендуется):**

1. Скачай `HermesCubeSetup.exe` из [Releases](https://github.com/kirarud/hermes-cube/releases)
2. Запусти — выбери путь, настройки, готово

**Вариант 2 — portable `.exe`:**

1. Скачай `HermesCube.exe` из [Releases](https://github.com/kirarud/hermes-cube/releases)
2. Запусти — работает из любой папки

## ⚙️ Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| `s` | Открыть настройки |
| `Esc` / `q` | Скрыть окно |
| ЛКМ + тащить | Переместить окно |
| ПКМ | Контекстное меню |

## 🛠 Настройки

| Параметр | По умолчанию | Диапазон | Описание |
|----------|-------------|----------|----------|
| Размер куба | 0.27 | 0.08 – 0.6 | Общий масштаб куба |
| Скорость вращения | 0.28 | 0.05 – 1.0 | Скорость вращения |
| Частота пульсации | 1.8 | 0.3 – 5.0 | Частота пульсации |
| Амплитуда пульсации | 0.12 | 0.0 – 0.35 | Сила пульсации |
| Плотность частиц | 12 | 6 – 20 | Частиц на грань |
| Размер частицы | 6 | 2 – 12 | Размер в px |
| Пресет формы | cube | cube/sphere/torus/dna | Форма куба (реал-тайм) |
| Морфинг | 0% | 0% – 100% | Плавный переход куба в форму |
| Форма частиц | square | square/circle/dot | Форма частицы |
| Поверх всех окон | да | да/нет | Overlay-режим |

## 🏗 Сборка из исходников

```bash
# Требуется: Python 3.11+, PyInstaller

# Установка зависимостей
pip install numpy pillow pystray pyinstaller

# Сборка куба
pyinstaller --onefile --windowed \
  --collect-all pystray --collect-all PIL --collect-all numpy \
  --hidden-import=pystray._win32 --hidden-import=PIL._tkinter_finder \
  --name HermesCube cube_app.py

# Сборка установщика (требуется HermesCube.exe рядом)
pyinstaller --onefile --windowed \
  --collect-all pystray --collect-all PIL \
  --hidden-import=pystray._win32 \
  --add-data "HermesCube.exe;." \
  --name HermesCubeSetup installer.py
```

## 📦 Структура проекта

```
hermes-cube/
├── cube_app.py              # Основное приложение — куб
├── installer.py              # Установщик
├── dist/                     # Сборки
│   ├── HermesCube.exe        # Portable .exe
│   └── HermesCubeSetup.exe   # Установщик
├── src/                      # (будущие модули)
└── screenshots/              # Скриншоты
```

## 📄 Лицензия

MIT
