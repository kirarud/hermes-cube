# Legacy — не развиваем в main

Актуальная точка входа: **`python main.py`** (`HermesEngine`).

Ниже — код, который **существует**, но **не является частью Companion-цикла** в `main.py`.  
Не удаляем (можно учиться / переносить кусками позже). **Не расширяем** без решения сменить North Star.

| Модуль | Назначение | Где жил |
|--------|------------|---------|
| `cube_app.py` | Старый монолит CubeApp + Settings/config helpers | `python cube_app.py` |
| `pixel_grid.py` | Пиксельный framebuffer UI | cube_app (G) |
| `cube_agents.py` | Button/Slider/Text агенты | cube_app / demo |
| `particle_agents.py` | Cursor / Spawner / роли | cube_app |
| `obsidian_graph.py` | 3D граф vault | отдельный overlay |
| `spatial_depth.py` | Спираль / Z-слои | эксперимент |
| `cube_agent_demo.py` | Демо агентов | demo |
| `AUTONOMOUS_PLAN.md` | Старый многоэтапный план | устарел vs NORTH_STAR |

## Что main всё ещё берёт из cube_app

- `load_config` / `save_config` / `DEFAULT_CONFIG`
- `SettingsWindow`
- tray helpers / transparent color constants

Это **shared utils**, не «вторая программа».

## Правило

Новая фича → только если она улучшает сценарий из [NORTH_STAR.md](NORTH_STAR.md).  
Иначе — в parked, не в `main.py`.
