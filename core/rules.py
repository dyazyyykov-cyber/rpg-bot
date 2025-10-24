from __future__ import annotations

"""
Правила и справочники для ядра:
- Бан-лист (BANLIST_DEFAULT)
- Разрешённые ключи world_flags (WORLD_FLAGS_WHITELIST / WORLD_FLAG_PREFIXES)
- Допустимые значения mood (NPC_MOODS)
- Границы HP (HP_MIN/HP_MAX_*) и утилиты
- Словари причинности (ACTION_TO_EFFECTS, CAUSALITY_RULES)
- Лёгкий парсинг/валидация пользовательского действия (parse_action_text/validate_action)

NB: Модуль без сторонних зависимостей. Точки использования:
- prompts.py (бан-лист, причинность)
- engine.py (валидация флагов/мудов/эффектов, при желании)
- utils.py (опционально)
"""

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# =============================================================================
# БАН-ЛИСТ (можно расширить через state["forbidden"])
# =============================================================================

BANLIST_DEFAULT: List[str] = [
    # мета/бот/четвёртая стена
    "чатгпт", "chatgpt", "бот", "чат-бот", "чат бот", "нейросеть",
    "четвёртая стена", "четвертая стена", "мета-комментарий", "метакомментарий",
    # современные медиа-обёртки
    "стрим", "стример", "подписчики", "ютуб", "youtube", "twitch", "твитч", "подкаст",
    # вне-мира
    "смартфон", "браузер", "интернет", "сайт", "веб-сайт",
    # мусор/токсичное
    "ддос", "ddos", "дудос",
]

# alias для внешнего кода
BANLIST = BANLIST_DEFAULT

# =============================================================================
# WORLD FLAGS — допустимые ключи и префиксы
# =============================================================================

# Чёткие ключи (добавляйте по мере необходимости для ваших миров)
WORLD_FLAGS_WHITELIST: List[str] = [
    # базовые состояния сцены/уровня угрозы/света и т.д.
    "alarm_level", "power", "door_locked", "lights", "weather",
    "oxygen_level", "radiation_level", "noise", "visibility",
    # пазлы/квесты/таймеры/подсказки
    "puzzle_state", "quest_step", "timer_main", "hint_1", "hint_2",
]

# Разрешённые префиксы (позволяют добавлять конкретизированные ключи)
WORLD_FLAG_PREFIXES: List[str] = [
    "device_", "door_", "light_", "alarm_", "puzzle_", "hint_", "timer_", "scene_", "quest_", "threat_",
]

EFFECT_KEYS_WHITELIST: List[str] = [
    "world_flags", "location", "scene_items_add", "scene_items_remove", "players", "npcs", "introductions",
]


def is_allowed_world_flag(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return False
    if k in WORLD_FLAGS_WHITELIST:
        return True
    return any(k.startswith(p) for p in WORLD_FLAG_PREFIXES)


# =============================================================================
# NPC mood / HP границы
# =============================================================================

NPC_MOODS: List[str] = [
    "нейтральный", "спокоен", "дружелюбный", "насторожен", "взволнован",
    "испуган", "агрессивный", "взбешён", "уставший", "ранен", "холоден",
]

HP_MIN = 0
HP_MAX_PLAYER = 100
HP_MAX_NPC = 200


def clamp_hp(value: Any, *, is_npc: bool = False, max_hp: Optional[int] = None) -> int:
    try:
        v = int(float(value))
    except Exception:
        v = HP_MAX_NPC if is_npc else HP_MAX_PLAYER
    mx = (HP_MAX_NPC if is_npc else HP_MAX_PLAYER) if max_hp is None else int(max_hp)
    if mx < 1:
        mx = 1
    if v < HP_MIN:
        return HP_MIN
    if v > mx:
        return mx
    return v


def normalize_mood(mood: Optional[str]) -> Optional[str]:
    if not mood:
        return None
    s = str(mood).strip().lower()
    # простая нормализация диакритики/вариантов
    repl = {
        "взволнованный": "взволнован",
        "спокойный": "спокоен",
        "агрессивен": "агрессивный",
        "нейтрально": "нейтральный",
    }
    s = repl.get(s, s)
    return s if s in NPC_MOODS else None


# =============================================================================
# Причинность (для промптов Effects и эвристик)
# =============================================================================

# Описание: ключевые слова в действиях → рекомендуемые типы дельт
ACTION_TO_EFFECTS: List[Tuple[Sequence[str], Sequence[str]]] = [
    (("взять", "поднять", "забрать"), ("scene_items_remove", "players.items_add")),
    (("уронить", "бросить", "оставить"), ("scene_items_add", "players.items_remove")),
    (("ударить", "атаковать", "выстрелить", "ранить"), ("npcs.hp_delta", "npcs.mood")),
    (("переместиться", "подойти", "отступить", "зайти"), ("players.position",)),
    (("открыть", "взломать", "выключить", "включить"), ("world_flags", "location")),
    (("осмотреть", "исследовать", "поискать"), ("introductions.items", "introductions.locations", "world_flags")),
    (("поговорить", "спросить", "уговорить"), ("npcs.mood", "world_flags")),
]

# Человекочитаемые правила для вставки в промпт (см. prompts.build_effects_prompt)
CAUSALITY_RULES: List[str] = [
    "взять/поднять предмет → scene_items_remove + players.items_add",
    "уронить/оставить предмет → scene_items_add + players.items_remove",
    "ударить/ранить NPC → npcs.hp_delta (и, возможно, mood)",
    "исследовать устройство → world_flags.device_* или introductions.items/locations",
    "переместиться → players.position или location",
    "диалог с NPC → npcs.mood и/или world_flags.hint_*",
]


# =============================================================================
# Парсинг и базовая валидация действий игрока (легковесно)
# =============================================================================

_MAX_ACTION_CHARS = int(os.getenv("MAX_ACTION_CHARS", "1200"))
_MAX_ACTION_TOKENS = int(os.getenv("MAX_ACTION_TOKENS", "300"))
_FORBID_COMMAND_PREFIX = os.getenv("FORBID_COMMAND_PREFIX", "1") == "1"
_FORBID_EMPTY_AFTER_NORMALIZE = True

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _normalize_ws(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u200b", "")  # zero-width
    t = _CTRL_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def detect_language(text: str) -> str:
    if re.search(r"[А-Яа-яЁё]", text or ""):
        return "ru"
    if re.search(r"[A-Za-z]", text or ""):
        return "en"
    return "unknown"


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def is_command_like(text: str) -> bool:
    return bool((text or "").startswith("/"))


def is_ooc(text: str) -> bool:
    t = (text or "").lower()
    return ("[ooc" in t) or ("(ooc" in t) or bool(re.search(r"\(\(.*?\)\)", t))


def parse_action_text(text: str) -> Dict[str, Any]:
    norm = _normalize_ws(text or "")
    lang = detect_language(norm)
    tokens = approx_token_count(norm)
    return {
        "raw": text or "",
        "text": norm,
        "lang": lang,
        "length": len(norm),
        "tokens_approx": tokens,
        "is_command": is_command_like(norm),
        "has_ooc": is_ooc(norm),
    }


def validate_action(
    session_state: Dict[str, Any],
    player_snapshot: Dict[str, Any],
    action_parsed: Dict[str, Any],
) -> Tuple[bool, str]:
    text = action_parsed.get("text", "")
    if _FORBID_EMPTY_AFTER_NORMALIZE and not text:
        return False, "empty_action"

    if _FORBID_COMMAND_PREFIX and action_parsed.get("is_command", False):
        return False, "starts_with_command"

    if action_parsed.get("length", 0) > _MAX_ACTION_CHARS:
        return False, "too_long"
    if action_parsed.get("tokens_approx", 0) > _MAX_ACTION_TOKENS:
        return False, "too_many_tokens"

    # игрок должен быть в состоянии действовать
    hp = player_snapshot.get("hp")
    try:
        if hp is not None and int(hp) <= 0:
            return False, "player_unconscious"
    except Exception:
        pass

    status = player_snapshot.get("status") or {}
    if isinstance(status, dict):
        if status.get("asleep") or status.get("stunned") or status.get("muted"):
            return False, "player_restricted"

    pos = (player_snapshot.get("position") or "").strip().lower()
    if pos in {"off", "disabled", "removed"}:
        return False, "invalid_position"

    return True, "ok"


__all__ = [
    # банлист
    "BANLIST_DEFAULT", "BANLIST",
    # world flags
    "WORLD_FLAGS_WHITELIST", "WORLD_FLAG_PREFIXES", "EFFECT_KEYS_WHITELIST", "is_allowed_world_flag",
    # mood/hp
    "NPC_MOODS", "HP_MIN", "HP_MAX_PLAYER", "HP_MAX_NPC", "clamp_hp", "normalize_mood",
    # причинность
    "ACTION_TO_EFFECTS", "CAUSALITY_RULES",
    # действия
    "parse_action_text", "validate_action", "detect_language", "approx_token_count", "is_command_like", "is_ooc",
]
