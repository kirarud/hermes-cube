"""core/ai_constants.py — Общие константы для AI-систем.

Ключи настроения совпадают с DEFAULT_CONFIG / projection:
  pulse_rate, pulse_amplitude, rotation_speed, color_shift
"""

# Настроения и их параметры (имена = ключи config куба)
AI_MOODS: dict = {
    'idle': {
        'pulse_rate': 1.8,
        'pulse_amplitude': 0.12,
        'rotation_speed': 0.28,
        'color_shift': 0.0,
    },
    'thinking': {
        'pulse_rate': 3.5,
        'pulse_amplitude': 0.25,
        'rotation_speed': 0.50,
        'color_shift': 0.15,
    },
    'speaking': {
        'pulse_rate': 2.5,
        'pulse_amplitude': 0.18,
        'rotation_speed': 0.40,
        'color_shift': 0.08,
    },
    'happy': {
        'pulse_rate': 2.8,
        'pulse_amplitude': 0.22,
        'rotation_speed': 0.50,
        'color_shift': 0.12,
    },
    'sad': {
        'pulse_rate': 0.8,
        'pulse_amplitude': 0.05,
        'rotation_speed': 0.10,
        'color_shift': 0.55,
    },
}

# --- Бэкенд чата -------------------------------------------------------------
# Куб говорит через любой OpenAI-совместимый endpoint.
# Приоритет источников (первый сработавший):
#   1. env CUBE_AI_BASE_URL / CUBE_AI_KEY / CUBE_AI_MODEL
#   2. ~/.hermes/config.yaml  (тот же роутер, что использует Hermes Agent)
#   3. локальный LM Studio на 127.0.0.1:1234 без ключа
#
# Ключ НИКОГДА не хранится в репозитории — только env или config.yaml.

import os
import re
from pathlib import Path
from typing import Dict

LM_STUDIO_URL: str = "http://127.0.0.1:1234"  # legacy-имя, локальный фолбэк
LOCAL_BASE_URL: str = "http://127.0.0.1:1234/v1"
LOCAL_MODEL: str = "gemma-4-e4b-it"
AI_MODEL_ID: str = "lmstudio-community/gemma-4-E4B-it-GGUF"

HERMES_CONFIG: Path = Path.home() / ".hermes" / "config.yaml"

_ENV_REF = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def _load_dotenv(path: Path) -> Dict[str, str]:
    """Прочитать KEY=VALUE файл в словарь. Без внешних зависимостей.

    Hermes хранит ключи в ~/.hermes/.env, но в окружение процесса
    их никто не загружает — поэтому ${VAR} раньше разворачивался в пустоту.
    """
    out: Dict[str, str] = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key:
                    out[key] = val
    except OSError:
        pass
    return out


_DOTENV: Dict[str, str] = _load_dotenv(Path.home() / '.hermes' / '.env')


def _expand_env(value: str) -> str:
    """Разворачивает ${VAR}: сначала окружение, потом ~/.hermes/.env."""
    def _sub(m: 're.Match[str]') -> str:
        name = m.group(1)
        return os.environ.get(name) or _DOTENV.get(name, '')
    return _ENV_REF.sub(_sub, value).strip()


def _parse_model_block(text: str) -> Dict[str, str]:
    """Минимальный разбор блока `model:` без PyYAML.

    PyYAML не входит в зависимости куба и в venv его нет. Раньше это
    приводило к `except Exception: return ('', '', '')` — конфиг Гермеса
    молча не читался, и куб уходил в offline, хотя ключ и URL были верны.
    """
    out: Dict[str, str] = {}
    in_model = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_model = line.strip().rstrip(':') == 'model'
            continue
        if in_model and indent <= 4:
            key, sep, val = line.strip().partition(':')
            if sep:
                out[key.strip()] = val.strip()
    return out


def _from_hermes_config() -> tuple[str, str, str]:
    """(base_url, api_key, model) из config.yaml Hermes. Пустые строки если нет."""
    try:
        with open(HERMES_CONFIG, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return ('', '', '')

    model_cfg: Dict[str, str] = {}
    try:
        import yaml  # локальный импорт: есть — хорошо, нет — обойдёмся
        cfg = yaml.safe_load(text) or {}
        model_cfg = cfg.get('model') or {}
    except Exception:
        model_cfg = _parse_model_block(text)
    if not isinstance(model_cfg, dict):
        model_cfg = _parse_model_block(text)

    return (
        _expand_env(str(model_cfg.get('base_url') or '')),
        _expand_env(str(model_cfg.get('api_key') or '')),
        _expand_env(str(model_cfg.get('default') or '')),
    )


def resolve_ai_backend() -> tuple[str, str, str]:
    """Возвращает (base_url с /v1, api_key, model)."""
    env_url = os.environ.get('CUBE_AI_BASE_URL', '').strip()
    if env_url:
        return (
            env_url.rstrip('/'),
            os.environ.get('CUBE_AI_KEY', '').strip(),
            os.environ.get('CUBE_AI_MODEL', '').strip() or LOCAL_MODEL,
        )

    # Облако — только по явному разрешению. Оно стоит денег, локальная
    # модель — нет. К тому же lm_autostart считает наличие API-ключа
    # признаком «облако активно» и не поднимает локальный сервер.
    if os.environ.get('CUBE_AI_CLOUD') == '1':
        url, key, model = _from_hermes_config()
        if url and key:
            return (url.rstrip('/'), key, model or LOCAL_MODEL)

    return (LOCAL_BASE_URL, '', LOCAL_MODEL)


AI_BASE_URL, AI_API_KEY, AI_MODEL = resolve_ai_backend()

# Как показывать ответ: letters | avatar | both
DEFAULT_RESPONSE_MODE: str = "both"
