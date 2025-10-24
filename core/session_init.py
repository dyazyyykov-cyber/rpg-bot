# core/session_init.py
from __future__ import annotations

"""
Инициализация новой сессии (стартовые значения) + LLM-генераторы мира/ролей/прологов.

Пайплайн:
1) generate_world_v2()          — стартовый мир (учитывает story_theme при наличии)
2) generate_roles_for_players() — роли 1:1 по игрокам
3) generate_initial_backstory() — персональные прологи (DM) по каждому игроку
4) assemble_initial_state()     — финальный state для сохранения в БД/Redis
"""

import logging
from typing import Any, Dict, List, Optional

from .llm import request_llm_chat_json
from .utils import get_env, ensure_inventory_invariants, normalize_text
from .prompts import (
    build_world_prompt_with_players,
    build_roles_for_players_prompt,
    build_initial_backstory_prompt,
)

try:
    # Если есть строгая pydantic-схема State — проверим итоговое состояние
    from .schemas import State as StateSchema  # type: ignore
except Exception:  # pragma: no cover
    StateSchema = None  # type: ignore

logger = logging.getLogger(__name__)
logger.name = "core.session_init"
logger.propagate = True


# -------------------- ENV --------------------

SESSION_INIT_TEMPERATURE = float(get_env("SESSION_INIT_TEMPERATURE", "0.6"))
SESSION_INIT_TOP_P       = float(get_env("SESSION_INIT_TOP_P", "0.92"))
SESSION_INIT_N_PREDICT   = int(get_env("SESSION_INIT_N_PREDICT", "1000"))
SESSION_INIT_TIMEOUT     = int(get_env("SESSION_INIT_TIMEOUT", "300"))


# -------------------- Grammar-friendly JSON Schemas --------------------

def _world_v2_schema() -> Dict[str, Any]:
    """
    Плоская schema InitialWorldV2 без внешних $ref (дружелюбна к JSON-grammar моделей):
    """
    return {
        "type": "object",
        "properties": {
            "turn": {"type": "integer"},
            "title": {"type": "string"},
            "setting": {"type": "string", "minLength": 20, "maxLength": 2000},
            "location": {"type": "string", "minLength": 2, "maxLength": 400},
            "world_flags": {
                "type": "object",
                "additionalProperties": {
                    "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]
                },
            },
            "npcs": {
                "type": "array",
                "minItems": 0,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "id":   {"type": "string", "minLength": 1, "maxLength": 64},
                        "name": {"type": "string", "minLength": 1, "maxLength": 80},
                        "mood": {"type": "string"},
                        "hp":   {"type": "integer"},
                        "items": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "name":   {"type": "string", "minLength": 1, "maxLength": 80},
                                            "source": {"type": "string", "minLength": 1, "maxLength": 80},
                                        },
                                        "required": ["name"],
                                        "additionalProperties": True,
                                    },
                                ]
                            },
                        },
                    },
                    "required": ["id", "name"],
                    "additionalProperties": True,
                },
            },
            "available_items": {
                "type": "array",
                "minItems": 0,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "name":   {"type": "string", "minLength": 1, "maxLength": 80},
                        "source": {"type": "string", "minLength": 1, "maxLength": 80},
                    },
                    "required": ["name"],
                    "additionalProperties": True,
                },
            },
            "opening_hook": {"type": "string", "minLength": 10, "maxLength": 500},
            "visibility_defaults": {"type": "object", "additionalProperties": True},
            "style": {"type": "object", "additionalProperties": True},
            "story_theme": {"type": "string"},
        },
        "required": [
            "turn",
            "setting",
            "location",
            "world_flags",
            "npcs",
            "available_items",
            "opening_hook",
        ],
        "additionalProperties": True,
    }


def _roles_for_players_schema(expected_count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "roles_for_players": {
                "type": "array",
                "minItems": int(expected_count),
                "maxItems": int(expected_count),
                "items": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "role":      {"type": "string", "minLength": 1, "maxLength": 80},
                        "summary":   {"type": "string", "minLength": 3, "maxLength": 280},
                    },
                    "required": ["player_id", "role", "summary"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["roles_for_players"],
        "additionalProperties": False,
    }


def _player_story_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "text":           {"type": "string", "minLength": 1, "maxLength": 2000},
            "echo_of_action": {"type": "string", "minLength": 1, "maxLength": 400},
            "highlights":     {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
        },
        "required": ["text", "echo_of_action"],
        "additionalProperties": False,
    }


# -------------------- Helpers --------------------

def _players_min(players: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"player_id": str(p.get("player_id") or ""), "name": str(p.get("name") or "")}
        for p in (players or [])
    ]


def _sanitize_world_flags(flags: Any) -> Dict[str, Any]:
    if not isinstance(flags, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in flags.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out


def _normalize_available_items(items: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(items, list):
        return out
    seen = set()
    for it in items:
        name = it.get("name") if isinstance(it, dict) else str(it)
        name_n = normalize_text(str(name or ""))
        if not name_n:
            continue
        key = name_n.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name_n[:80], "source": "scene"})
    return out


def _build_default_open_threads(world: Dict[str, Any]) -> List[str]:
    threads: List[str] = []
    flags = (world.get("world_flags") or {}) if isinstance(world, dict) else {}
    if isinstance(flags, dict):
        if flags.get("alarm_level") not in (None, 0, False, "0", "off"):
            threads.append("Сбросить уровень тревоги")
        if flags.get("door_locked") in (True, "true", "locked"):
            threads.append("Открыть запертую дверь")
    while len(threads) < 2:
        threads.append(["Разведать локацию", "Найти полезные предметы", "Связаться с NPC поблизости"][len(threads)])
    return threads[:4]


def _build_default_clocks(world: Dict[str, Any]) -> Dict[str, int]:
    flags = (world.get("world_flags") or {}) if isinstance(world, dict) else {}
    clocks: Dict[str, int] = {"satellite_warmup": 0}
    clocks["alarm_pressure"] = 1 if isinstance(flags, dict) and flags.get("alarm_level") not in (None, 0, False, "0", "off") else 0
    return clocks


def _looks_bad_text(s: str) -> bool:
    s_l = (s or "").lower().strip()
    if len(s_l) < 40:
        return True
    # Частые маркеры-заглушки из «плохих» ответов
    bad_markers = ("должен", "пуст", "заполн", "placeholder", "todo")
    return any(x in s_l for x in bad_markers)


# -------------------- Public LLM API --------------------

async def generate_world_v2(
    *,
    group_chat_id: int,
    title: str,
    players: List[Dict[str, Any]],
    story_theme: str = "",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Генерация стартового мира (turn=0).
    Тема story_theme опциональна: если передана — обязана органично лечь в сеттинг/локацию/хук.
    """
    players_min = _players_min(players)
    if not players_min:
        raise RuntimeError("INIT: нет игроков — сначала присоединиться, потом генерировать мир")

    messages = [
        {"role": "system", "content": "—"},
        {"role": "user", "content": build_world_prompt_with_players(title=title, players=players_min, story_theme=story_theme)},
    ]
    schema = _world_v2_schema()

    logger.info("INIT.WORLD: request (players=%d) title=%s", len(players_min), title)
    data: Dict[str, Any] = await request_llm_chat_json(
        messages=messages,
        schema=schema,
        temperature=SESSION_INIT_TEMPERATURE,
        top_p=SESSION_INIT_TOP_P,
        max_tokens=SESSION_INIT_N_PREDICT,
        timeout_s=(timeout or SESSION_INIT_TIMEOUT),
        log_prefix="INIT_WORLD",
        log_dir="./logs",
    )

    # Quality-gate: ретрай «жёстко», если пришла ерунда/заглушки
    if _looks_bad_text(data.get("setting", "")) or _looks_bad_text(data.get("location", "")) or _looks_bad_text(data.get("opening_hook", "")):
        data = await request_llm_chat_json(
            messages=messages,
            schema=schema,
            temperature=0.0,
            top_p=0.0,
            max_tokens=SESSION_INIT_N_PREDICT,
            timeout_s=(timeout or SESSION_INIT_TIMEOUT),
            log_prefix="INIT_WORLD_RETRY",
            log_dir="./logs",
        )

    # Жёсткие правки формы (без подмены смысла)
    data["title"] = title
    data["turn"] = 0
    if story_theme and not data.get("story_theme"):
        data["story_theme"] = story_theme

    data["world_flags"]     = _sanitize_world_flags(data.get("world_flags"))
    data["available_items"] = _normalize_available_items(data.get("available_items"))

    # Мягко «подсветим» тему в opening_hook, если модель её не упомянула
    if story_theme:
        hook = (data.get("opening_hook") or "").strip()
        if story_theme.lower() not in hook.lower():
            data["opening_hook"] = f"Тема: {story_theme}. {hook}".strip()

    # Мини-валидация обязательных полей
    for key in ("setting", "location", "opening_hook"):
        if not isinstance(data.get(key), str) or not str(data[key]).strip():
            raise RuntimeError(f"INIT.WORLD: обязательное поле '{key}' пустое")

    return data


async def generate_roles_for_players(
    *,
    state: Dict[str, Any],
    players: List[Dict[str, Any]],
    story_theme: str = "",
    timeout: Optional[int] = None,
) -> List[dict]:
    """
    Ровно по числу игроков: [{player_id, role, summary}, ...]
    Порядок в результате соответствует исходному players.
    """
    players_min = _players_min(players)
    if not players_min:
        raise RuntimeError("INIT.ROLES: пустой список игроков")

    messages = [
        {"role": "system", "content": "—"},
        {"role": "user", "content": build_roles_for_players_prompt(state=state, players=players_min, story_theme=story_theme)},
    ]
    schema = _roles_for_players_schema(expected_count=len(players_min))

    logger.info("INIT.ROLES: request (players=%d)", len(players_min))
    resp: Dict[str, Any] = await request_llm_chat_json(
        messages=messages,
        schema=schema,
        temperature=SESSION_INIT_TEMPERATURE,
        top_p=SESSION_INIT_TOP_P,
        max_tokens=SESSION_INIT_N_PREDICT,
        timeout_s=(timeout or SESSION_INIT_TIMEOUT),
        log_prefix="INIT_ROLES",
        log_dir="./logs",
    )

    roles_list = resp.get("roles_for_players") or []
    if not isinstance(roles_list, list) or len(roles_list) != len(players_min):
        raise RuntimeError("INIT.ROLES: размер roles_for_players не совпадает с числом игроков")

    # Верифицируем и нормализуем + карта по player_id
    mapping: Dict[str, Dict[str, str]] = {}
    for it in roles_list:
        pid = str(it.get("player_id") or "")
        role = normalize_text(it.get("role") or "")
        summary = normalize_text(it.get("summary") or "")
        if not pid or not role or not summary:
            raise RuntimeError("INIT.ROLES: элемент роли имеет пустые поля")
        if pid in mapping:
            raise RuntimeError(f"INIT.ROLES: дубликат player_id={pid}")
        mapping[pid] = {"player_id": pid, "role": role[:80], "summary": summary[:280]}

    # Сохраним порядок игроков
    ordered: List[Dict[str, str]] = []
    for p in players_min:
        pid = p["player_id"]
        if pid not in mapping:
            raise RuntimeError(f"INIT.ROLES: не найдена роль для player_id={pid}")
        ordered.append(mapping[pid])

    return ordered


async def generate_initial_backstory(
    *,
    state: Dict[str, Any],
    player: Dict[str, Any],
    story_theme: str = "",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Персональный пролог (PlayerStory): {text, echo_of_action, highlights?}
    """
    pid = player.get("player_id")
    if not pid:
        raise RuntimeError("INIT.PRELUDE: у игрока отсутствует player_id")

    messages = [
        {"role": "system", "content": "—"},
        {"role": "user", "content": build_initial_backstory_prompt(state=state, player=player, story_theme=story_theme)},
    ]
    schema = _player_story_schema()

    logger.info("INIT.PRELUDE: request (pid=%s)", pid)
    story: Dict[str, Any] = await request_llm_chat_json(
        messages=messages,
        schema=schema,
        temperature=SESSION_INIT_TEMPERATURE,
        top_p=SESSION_INIT_TOP_P,
        max_tokens=SESSION_INIT_N_PREDICT,
        timeout_s=(timeout or SESSION_INIT_TIMEOUT),
        log_prefix=f"INIT_PRIVATE:{pid}",
        log_dir="./logs",
    )

    # Мини-проверка + ретрай «жёстко» при пустых полях
    for key in ("text", "echo_of_action"):
        if not isinstance(story.get(key), str) or not story[key].strip():
            story = await request_llm_chat_json(
                messages=messages,
                schema=schema,
                temperature=0.0,
                top_p=0.0,
                max_tokens=SESSION_INIT_N_PREDICT,
                timeout_s=(timeout or SESSION_INIT_TIMEOUT),
                log_prefix=f"INIT_PRIVATE_RETRY:{pid}",
                log_dir="./logs",
            )
            break

    for key in ("text", "echo_of_action"):
        if not isinstance(story.get(key), str) or not story[key].strip():
            raise RuntimeError(f"INIT.PRELUDE: обязательное поле '{key}' пустое (pid={pid})")

    story["text"] = normalize_text(story.get("text", ""))[:2000]
    story["echo_of_action"] = normalize_text(story.get("echo_of_action", ""))[:400]
    story["highlights"] = list(story.get("highlights") or [])[:8]
    return story


# -------------------- State assembly --------------------

def _attach_roles_to_players(players_snapshot: List[Dict[str, Any]], roles: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_id = {r["player_id"]: r for r in roles or []}
    out: List[Dict[str, Any]] = []
    for p in (players_snapshot or []):
        pid = str(p.get("player_id") or "")
        q = dict(p)
        if pid and pid in by_id:
            role = by_id[pid].get("role")
            if role:
                q["role"] = role
        out.append(q)
    return out


def assemble_initial_state(
    *,
    world: Dict[str, Any],
    players_snapshot: List[Dict[str, Any]],
    roles_for_players: Optional[List[Dict[str, str]]] = None,
    open_threads: Optional[List[str]] = None,
    clocks: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Собирает корректное начальное состояние (совместимо с engine/turn_runner).
    Инварианты:
      - scene items — только уникальные, нормализованные
      - у игроков валидные hp/max_hp/position/items
      - присутствуют open_threads и clocks
    """
    st: Dict[str, Any] = {
        "title":      normalize_text(world.get("title", ""))[:200] if isinstance(world, dict) else "",
        "turn":       int(world.get("turn", 0)) if isinstance(world, dict) else 0,
        "setting":    normalize_text(world.get("setting", ""))[:2000],
        "location":   normalize_text(world.get("location", ""))[:160],
        "world_flags": world.get("world_flags") if isinstance(world, dict) else {},
        "npcs":        world.get("npcs") if isinstance(world, dict) else [],
        "available_items": _normalize_available_items((world.get("available_items") or []) if isinstance(world, dict) else []),
        "opening_hook": normalize_text(world.get("opening_hook", ""))[:500],
        "players": [
            {
                "player_id": p.get("player_id"),
                "name":      normalize_text(p.get("name", ""))[:80],
                "hp":        int(p.get("hp", 100) or 100),
                "max_hp":    int(p.get("max_hp", 100) or 100),
                "position":  normalize_text(p.get("position", "scene") or "scene")[:64] or "scene",
                "status":    p.get("status") or {},
                "items":     list(p.get("items") or []),
                "role":      normalize_text(p.get("role", ""))[:80] if "role" in p else "",
            }
            for p in (players_snapshot or [])
        ],
        "effects_log":     [],
        "raw_history":     [],
        "private_history": [],
        "general_history": [],
        "forbidden":       list(world.get("forbidden", []) or []),
    }

    # Протащим тему, если она есть в мире
    if isinstance(world, dict) and world.get("story_theme"):
        st["story_theme"] = normalize_text(world.get("story_theme", ""))[:200]

    # Применим роли к снимку игроков
    if roles_for_players:
        st["players"] = _attach_roles_to_players(st["players"], roles_for_players)

    # Open threads / clocks
    st["open_threads"] = [normalize_text(x)[:120] for x in (open_threads if open_threads is not None else _build_default_open_threads(world)) if normalize_text(x)]
    st["clocks"] = {str(k): int(v) for k, v in ((clocks if clocks is not None else _build_default_clocks(world)).items())}

    # Инвентари/сцена — инварианты
    ensure_inventory_invariants(st)

    # Строгая проверка (если схемы доступны)
    if StateSchema is not None:
        StateSchema.model_validate(st)  # type: ignore[attr-defined]

    return st


# -------------------- Legacy API (удалено намеренно) --------------------

async def generate_initial_state_via_llm(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
    raise RuntimeError(
        "generate_initial_state_via_llm() удалён. Используйте последовательность: "
        "generate_world_v2 → generate_roles_for_players → generate_initial_backstory → assemble_initial_state"
    )
