"""core/ai_constants.py — Общие константы для AI-систем.

Вынесено из ai_module.py для удаления старого модуля.
"""

# Настроения и их параметры (pulse, speed, color shift)
AI_MOODS: dict = {
    'idle':     {'pulse_rate': 1.8, 'pulse_amp': 0.12, 'speed': 0.28, 'color_shift': 0.0},
    'thinking': {'pulse_rate': 3.5, 'pulse_amp': 0.25, 'speed': 0.50, 'color_shift': 0.15},
    'speaking': {'pulse_rate': 2.5, 'pulse_amp': 0.18, 'speed': 0.40, 'color_shift': 0.08},
    'happy':    {'pulse_rate': 2.8, 'pulse_amp': 0.22, 'speed': 0.50, 'color_shift': 0.12},
    'sad':      {'pulse_rate': 0.8, 'pulse_amp': 0.05, 'speed': 0.10, 'color_shift': 0.55},
}

LM_STUDIO_URL: str = "http://127.0.0.1:1234"
AI_MODEL: str = "gemma-4-e4b-it"
AI_MODEL_ID: str = "lmstudio-community/gemma-4-E4B-it-GGUF"
