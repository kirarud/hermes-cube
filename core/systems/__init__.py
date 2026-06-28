#!/usr/bin/env python3
"""core/systems/ — System pipeline for Hermes Engine.

Каждая System — чистая функция с единой сигнатурой:

    def update(world: World, dt: float) -> None:
        \"\"\"Прочитать world, изменить world. Никаких побочных эффектов.\"\"\"

Правила:
  1. System НЕ вызывает другие System
  2. System НЕ знает про Renderer / Tkinter / UI
  3. System НЕ создаёт новые entity (active_mask меняется через SpawnerSystem)
  4. System пишет только в свою зону World (sim/render/meta)
  5. System может читать любые зоны World
  6. Порядок System = порядок в Pipeline (Stage)
"""
