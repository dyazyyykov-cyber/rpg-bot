from __future__ import annotations

import json
import re
import inspect
from textwrap import dedent
from typing import Any, Dict, List, Optional

# =============================================================
# PROMPTS: PLAN → EFFECTS → PRIVATE → GENERAL → WORLD/ROLES/PRELUDES
#  - Строгий JSON (никакого текста вне JSON)
#  - Анти-«фиолетовая проза»
#  - Антимета (бренды/мемы НЕ запрещаем)
#  - Поддержка темы (опционально)
# =============================================================

# -----------------------------
# Общая шапка (строгий JSON)
# -----------------------------
_HEADER = (
    "Ты — генератор СТРОГОГО JSON. "
    "Верни РОВНО ОДИН JSON-объект; НИКАКОГО текста вне JSON, без бэктиков и комментариев. "
    "НЕ добавляй лишние поля и не меняй имена ключей. "
    "Не сериализуй массивы/объекты в строки. "
    "Обязательные поля ДОЛЖНЫ присутствовать; корневой объект не может быть пустым. "
    "Язык: русский."
)

# -----------------------------
# Антимета (бренды/мемы разрешены)
# -----------------------------
BANLIST_DEFAULT: List[str] = [
    # запрещаем только мета-выходы
    "четвёртая стена", "четвертая стена",
    "мета-комментарий", "метакомментарий",
    "бот", "чат-бот", "чат бот",
    "чатгпт", "chatgpt",
]

def _banlist_block(extra_forbidden: Optional[List[str]] = None) -> str:
    words = list(dict.fromkeys([*(extra_forbidden or []), *BANLIST_DEFAULT]))
    if not words:
        return ""
    return (
        "Запрещено: метакомментарии и выход из художественной роли вроде "
        + ", ".join(words[:24])
        + ". Нарушение приведёт к отбраковке и перегенерации.\n"
    )

# -----------------------------
# Лёгкие утилиты сжатия контекста
# -----------------------------

def _norm_item(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return {"name": str(x.get("name", ""))[:80], "source": str(x.get("source", "scene"))[:32]}
    return {"name": str(x)[:80], "source": "scene"}

def _norm_npc(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return {
            "id": x.get("id"),
            "name": str(x.get("name", ""))[:80],
            "mood": str(x.get("mood", ""))[:24],
            "hp": int(x["hp"]) if ("hp" in x and x["hp"] is not None) else 100,
            "items": [str(i.get("name") if isinstance(i, dict) else i)[:80] for i in (x.get("items") or [])][:8],
        }
    return {"id": None, "name": str(x)[:80], "mood": "", "hp": 100, "items": []}

def _norm_player(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return {
            "player_id": x.get("player_id"),
            "name": str(x.get("name", ""))[:80],
            "role": (str(x.get("role", ""))[:80] if x.get("role") else ""),
            "hp": int(x.get("hp") or 100),
            "max_hp": int(x.get("max_hp") or x.get("hp") or 100),
            "position": str(x.get("position", "scene"))[:32],
            "items": [str(i.get("name") if isinstance(i, dict) else i)[:80] for i in (x.get("items") or [])][:12],
        }
    return {"player_id": None, "name": str(x)[:80], "role": "", "hp": 100, "max_hp": 100, "position": "scene", "items": []}

def _short_world_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    npcs_raw = state.get("npcs") or []
    players_raw = state.get("players") or []
    items_raw = state.get("available_items") or []
    flags = dict(state.get("world_flags") or {})

    return {
        "title": state.get("title") or state.get("session_title") or "",
        "setting": state.get("setting") or "",
        "location": state.get("location") or "",
        "world_flags": flags,
        "turn": int(state.get("turn", 0) or 0),
        "npcs": [_norm_npc(n) for n in npcs_raw][:8],
        "players": [_norm_player(p) for p in players_raw][:12],
        "available_items": [_norm_item(i) for i in items_raw][:12],
        "allowed_world_flag_keys": sorted(list(flags.keys())),
    }

def _split_sents(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", s)
    return [p.strip() for p in parts if p.strip()]

def _recap_from_tail(tail_text: str, k: int = 2, max_chars: int = 320) -> str:
    sents = _split_sents(tail_text or "")
    if not sents:
        return ""
    recap = " ".join(sents[-k:])
    if len(recap) > max_chars:
        recap = recap[-max_chars:].lstrip()
    return recap

def _jsonable(obj: Any) -> Any:
    """Аккуратно превращает pydantic/объекты в json-способные словари."""
    try:
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return obj.model_dump()
        if hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()
        if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
            return obj
        # последний шанс
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return str(obj)

# =============================================================
# 0) PLAN / BeatPlan — сюжетный скелет сцены
# =============================================================

def build_beat_plan_prompt(
    *,
    state: Dict[str, Any],
    actions: List[Dict[str, Any]],
    open_threads: Optional[List[str]] = None,
    clocks: Optional[Dict[str, int]] = None,
    forbidden: Optional[List[str]] = None,
    must_mention: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> str:
    """
    Возврат: BeatPlan (goal, stakes, obstacle, twist_or_clue?,
    scene_outcomes[], must_mention[], forbidden[], open_threads_next[])
    """
    snap = _short_world_snapshot(state)
    acts = [{"player_id": a.get("player_id"), "text": (a.get("text") or "").strip()} for a in (actions or [])]
    must = [str(x).strip() for x in (must_mention or []) if str(x).strip()]

    parts: List[str] = []
    parts.append(_HEADER + "\n")
    parts.append("Сформируй план сцены (BeatPlan). Верни JSON по схеме BeatPlan.\n")
    parts.append("Обязательно: goal, stakes, obstacle, 1–3 scene_outcomes; допустим один короткий twist_or_clue.\n")
    parts.append(
        "must_mention: перечисли ТОЛЬКО те объекты/имена, которые уже есть в контексте (players/npcs/items/flags). "
        "Новые сущности в must_mention НЕ добавляй.\n"
    )
    if theme:
        parts.append(f"Тема сцены: {theme}\n")
    parts.append(_banlist_block(forbidden))

    if must:
        parts.append("Предметы/имена, которые нужно упомянуть: " + ", ".join(must[:8]) + "\n")

    parts.append("Контекст мира (сжатый):\n")
    parts.append(json.dumps(snap, ensure_ascii=False) + "\n")
    parts.append("Публичные действия игроков (нормализованные):\n")
    parts.append(json.dumps(acts, ensure_ascii=False) + "\n")
    if open_threads:
        parts.append("Открытые нити (можно продвинуть одну):\n" + json.dumps(open_threads[:6], ensure_ascii=False) + "\n")
    if clocks:
        parts.append("Таймеры сцены (clocks):\n" + json.dumps({k: clocks[k] for k in list(clocks)[:4]}, ensure_ascii=False) + "\n")

    parts.append(
        "Требования: лаконичность; никаких новых сущностей без необходимости; "
        "forbidden — учитывать; open_threads_next — не более 2 пунктов, если появляются."
    )
    return "".join(parts)

# =============================================================
# 1) EFFECTS + RAW — строгое применение причинности
# =============================================================

def build_effects_prompt(
    *,
    state: Dict[str, Any],
    actions: List[Dict[str, Any]],
    last_raw: Optional[List[Dict[str, Any]]] = None,
    forbidden: Optional[List[str]] = None,
    beat_plan: Optional[Dict[str, Any]] = None,
    theme: Optional[str] = None,
) -> str:
    snap = _short_world_snapshot(state)
    act = [{"player_id": a.get("player_id"), "text": (a.get("text") or "").strip()} for a in (actions or [])]
    last_raw_tail = (last_raw or [])[-1:] if last_raw else []

    allowed_effects_root = [
        "world_flags", "location",
        "scene_items_add", "scene_items_remove",
        "players", "npcs", "introductions",
    ]

    causality = [
        "взять/поднять предмет → scene_items_remove + players.items_add",
        "уронить/оставить предмет → scene_items_add + players.items_remove",
        "ударить/ранить NPC → npcs.hp_delta (и, возможно, mood)",
        "исследовать устройство → world_flags.device_* или introductions.items/locations",
        "переместиться → players.position или location",
        "диалог с NPC → npcs.mood и/или world_flags.hint_*",
    ]

    parts: List[str] = []
    parts.append(_HEADER + "\n")
    parts.append("Вычисли эффекты хода и сырую сцену. Верни JSON по схеме EffectsAndRaw.\n")
    if theme:
        parts.append(f"Тема сцены: {theme}\n")
    parts.append("Допустимые ключи effects: " + ", ".join(allowed_effects_root) + ". Другие запрещены.\n")
    parts.append("ИНВАРИАНТЫ:\n")
    parts.append("- Действия игроков — ФАКТ. Смысл действий НЕ меняй; если действие странное, опиши реакцию мира/NPC.\n")
    parts.append("- world_flags: изменяй ТОЛЬКО существующие ключи из списка ниже и ТОЛЬКО по явной причине (никаких новых ключей).\n")
    parts.append("- Инвентарь: available_items — только сцена; перенос предмета между сценой/игроком/НПС обязан быть консистентным.\n")
    parts.append("- ПРИ НАЛИЧИИ ДЕЙСТВИЙ effects НЕ МОЖЕТ БЫТЬ ПУСТЫМ: внеси минимум одну дельту.\n")
    parts.append("ФОРМА:\n")
    parts.append("- location: строка или null; без изменений — опусти ключ.\n")
    parts.append("- scene_items_add: [{\"name\",\"source\"}]; scene_items_remove: [имена].\n")
    parts.append("- players/npcs: только существующие id; hp_delta — целое; items_add/items_remove — списки имён предметов.\n")
    parts.append("- introductions: новые NPC/предметы/локации; иначе пустые списки.\n")
    parts.append("- RawStory.text: 1–3 предложения ТОЛЬКО про наблюдаемый итог ТЕКУЩИХ действий. Без мета.\n\n")
    if beat_plan:
        parts.append("Опорный план (BeatPlan):\n" + json.dumps(_jsonable(beat_plan), ensure_ascii=False) + "\n")
    parts.append(_banlist_block(forbidden))

    parts.append("Контекст мира (сжатый):\n")
    parts.append(json.dumps(snap, ensure_ascii=False) + "\n")
    parts.append("Список допустимых world_flag-ключей:\n")
    parts.append(json.dumps(snap.get("allowed_world_flag_keys", []), ensure_ascii=False) + "\n")
    parts.append("Действия игроков (нормализованные):\n")
    parts.append(json.dumps(act, ensure_ascii=False) + "\n")
    parts.append("Хвост предыдущего сырого лога (если есть):\n")
    parts.append(json.dumps(last_raw_tail, ensure_ascii=False) + "\n")

    parts.append("Подсказки причинности (примеры):\n• " + "\n• ".join(causality) + "\n")
    parts.append("Требования: консистентность, минимум одна дельта при непустых действиях, без новых world_flags-ключей.")
    return "".join(parts)

# =============================================================
# 2) PRIVATE STORY — внутренняя перспектива
# =============================================================

def build_private_story_prompt(
    *,
    state: Dict[str, Any],
    player: Dict[str, Any],
    raw_story: Dict[str, Any],
    last_private: Optional[List[Dict[str, Any]]] = None,
    public_names: Optional[List[str]] = None,
    forbidden: Optional[List[str]] = None,
    effects: Optional[Dict[str, Any]] = None,
) -> str:
    snap = _short_world_snapshot(state)
    pid = player.get("player_id")
    tail = (last_private or [])[-1:] if last_private else []

    if public_names is None:
        maybe_names = [p.get("name") for p in snap.get("players", []) if p.get("name")]
        public_names = [str(n).strip() for n in maybe_names if str(n).strip()]

    names_line = ""
    if public_names:
        safe = [str(n).strip() for n in public_names if str(n).strip()]
        if safe:
            names_line = "Допустимо аккуратно упоминать игровые имена/роли (ровно в данном написании): " + ", ".join(safe[:8]) + ". "

    effects_hint = ""
    if effects:
        effects_hint = "Если эффекты этого хода затрагивают героя (предметы/позицию/ранения/реакции NPC) — вплети 1–2 детали. "

    parts: List[str] = []
    parts.append(_HEADER + "\n")
    parts.append("Задача: личная история (PlayerStory) для ОДНОГО игрока. Верни JSON по схеме PlayerStory.\n")
    parts.append("Требования к полям:\n")
    parts.append("- text: 3–5 коротких предложений, НЕ пусто. Внутренняя перспектива; учитывай текущий наблюдаемый итог и контекст. ")
    parts.append("Избегай витиеватых метафор; не повторяй дословно прошлые тексты — переформулируй и продвигай состояние героя.\n")
    parts.append("- echo_of_action: краткая переформулировка действия (<=80 символов), без кавычек и имён игроков; СТРОКА.\n")
    parts.append("- highlights — опционально; если нечего выделить — опусти.\n")
    parts.append(_banlist_block(forbidden))
    if names_line:
        parts.append(names_line + "\n")
    if effects_hint:
        parts.append(effects_hint + "\n")

    parts.append("Контекст мира (сжатый):\n")
    parts.append(json.dumps(snap, ensure_ascii=False) + "\n")
    parts.append("Сырой лог текущего хода (raw):\n")
    parts.append(json.dumps(raw_story, ensure_ascii=False) + "\n")
    parts.append("Хвост предыдущей личной истории этого игрока:\n")
    parts.append(json.dumps(tail, ensure_ascii=False) + "\n")
    parts.append("Игрок: " + json.dumps({"player_id": pid, "name": player.get("name"), "role": player.get("role")}, ensure_ascii=False))
    return "".join(parts)

# =============================================================
# 3) GENERAL STORY — цельный абзац, сцепленный с эффектами
# =============================================================

def _summarize_effects(snapshot: Dict[str, Any], effects: Optional[Dict[str, Any]]) -> str:
    if not effects:
        return "—"
    parts: List[str] = []
    if effects.get("location") is not None:
        parts.append("локация: изменена")
    wf = effects.get("world_flags") or {}
    if isinstance(wf, dict) and wf:
        parts.append("флаги: " + ", ".join(list(wf.keys())[:4]))
    add = effects.get("scene_items_add") or []
    rem = effects.get("scene_items_remove") or []
    if add:
        names = [str(i.get("name") if isinstance(i, dict) else i) for i in add]
        parts.append("предметы добавлены: " + ", ".join(names[:4]))
    if rem:
        parts.append("предметы убраны: " + ", ".join([str(x) for x in rem[:4]]))
    if effects.get("npcs"):
        parts.append("NPC: реакции/изменения")
    if effects.get("players"):
        parts.append("игроки: изменения инвентаря/позиции/ранений")
    return "; ".join(parts) or "—"

def build_general_story_prompt(
    *,
    snapshot: Dict[str, Any],
    raw_story: str,
    prev_general_tail: Optional[List[str]] = None,
    actions_public: Optional[List[Dict[str, Any]]] = None,   # [{player:"Name", text:"..."}]
    private_echos: Optional[List[str]] = None,
    effects: Optional[Dict[str, Any]] = None,
    open_threads: Optional[List[str]] = None,
    beat_plan: Optional[Dict[str, Any]] = None,
    facts: Optional[List[str]] = None,
    theme: Optional[str] = None,
    forbidden: Optional[List[str]] = None,
) -> str:
    """
    GENERAL: один абзац (6–8 коротких предложений), явно отражающий публичные действия и минимум 2 факта из effects.
    Имена игроков использовать РОВНО в данном написании. Без витиеватых метафор и «фиолетовой прозы».
    """
    setting = str(snapshot.get("setting", "")).strip()
    location = str(snapshot.get("location", "")).strip()
    wf = snapshot.get("world_flags") or {}
    wf_kv = ", ".join([f"{k}={v}" for k, v in list(wf.items())[:6]]) or "—"

    acts = [a for a in (actions_public or []) if isinstance(a, dict) and a.get("text")]
    echos = [e for e in (private_echos or []) if str(e).strip()]
    prev_text = " ".join((prev_general_tail or [])[-1:]).strip()
    recap = _recap_from_tail(prev_text, k=2, max_chars=320)

    actions_block = (
        "Публичные действия ЭТОГО хода (имена используй РОВНО в данном написании; переписывать нельзя):\n"
        + "\n".join(["• " + str(a.get("player", "")) + " — " + str(a.get("text", "")) for a in acts])
        if acts else "Публичные действия ЭТОГО хода:\n• —"
    )
    raw_block = "Сырый наблюдаемый итог (raw):\n• " + (raw_story or "—")
    echos_block = (
        "Эхо приватных событий (неприватные, для атмосферы):\n" + "\n".join(["• " + str(e) for e in echos])
        if echos else "Эхо приватных событий:\n• —"
    )
    recap_block = "Короткий recap прошлого абзаца (перефразируй, не цитируй дословно):\n• " + recap if recap else ""

    eff_digest = _summarize_effects(snapshot, effects)

    style = (
        "Пиши просто и предметно. Короткие ясные предложения. "
        "Опиши последовательность событий и последствия, а не настроения. "
        "Избегай туманных сравнений, витиеватых метафор и поэтизмов."
    )

    parts: List[str] = []
    parts.append(_HEADER + "\n")
    parts.append("Верни JSON по схеме GeneralStory.\n")
    if theme:
        parts.append(f"Тема сцены: {theme}\n")
    parts.append("Срез сцены:\n")
    parts.append("— Мир: " + setting + "\n")
    parts.append("— Локация: " + location + "\n")
    parts.append("— Флаги: " + wf_kv + "\n\n")
    parts.append(actions_block + "\n")
    parts.append(raw_block + "\n")
    parts.append("Факты из effects (вплети минимум ДВА явно):\n• " + eff_digest + "\n")
    if facts:
        must_lines = "\n".join(f"- {str(f)}" for f in facts if str(f).strip())
        if must_lines:
            parts.append("ФАКТЫ, КОТОРЫЕ ОБЯЗАТЕЛЬНО ДОЛЖНЫ ПРОЗВУЧАТЬ (в явном виде):\n" + must_lines + "\n")
    parts.append(echos_block + "\n")
    if recap_block:
        parts.append(recap_block + "\n")
    if beat_plan:
        parts.append("Опорный план (BeatPlan):\n" + json.dumps(_jsonable(beat_plan), ensure_ascii=False) + "\n")
    if open_threads:
        parts.append("Открытые нити (заверши одну или продвинь одну вперёд):\n" + json.dumps(open_threads[:4], ensure_ascii=False) + "\n")

    parts.append(
        _banlist_block(forbidden)
        + "Задание:\n"
        f"{style} "
        "Напиши ОДИН цельный абзац (6–8 предложений) общей истории текущего хода. "
        "Вплети КАЖДОЕ публичное действие через последствия (реакции мира/NPC, риски, положение). "
        "Вставь минимум ДВА конкретных факта из effects (предметы/флаги/локация/NPC/раны/позиции) "
        "и ВСЕ факты из списка выше (если он есть). "
        "Имена игроков используй ровно в данном написании. "
        "Не пересказывай прошлый абзац и не цитируй его дословно; история должна сдвинуться на 1–2 шага вперёд. "
        "Запрещено: списки в тексте, мета/бот/чат/четвёртая стена. /no_think"
    )
    return "".join(parts)

# =============================================================
# 4) Генераторы мира/ролей/прологов — с поддержкой story_theme
# =============================================================

def build_world_prompt_with_players(
    *,
    title: str,
    players: List[Dict[str, Any]],
    story_theme: Optional[str] = None,
) -> str:
    players_line = ", ".join([p.get("name") or "Игрок" for p in players]) or "Игрок"
    theme_block = (
        f"Тема игры: «{story_theme.strip()}». Это главный мотив. "
        "Интегрируй тему в сеттинг, стартовую локацию и opening_hook; не делай отдельной сноски."
        if story_theme and story_theme.strip()
        else "Тема игры не задана. Придумай цельный, реалистичный сеттинг без отсылок к теме."
    )

    tmpl = {
        "turn": 0,
        "setting": "содержательный сеттинг мира (2–3 абзаца, без клише и мета-элементов)",
        "location": "конкретная стартовая сцена (1–2 насыщенных предложения)",
        "world_flags": {"time_of_day": "утро"},
        "npcs": [
            {"id": "npc_1", "name": "Имя", "mood": "нейтральный", "hp": 100, "items": []},
            {"id": "npc_2", "name": "Имя", "mood": "нейтральный", "hp": 100, "items": []},
        ],
        "available_items": [{"name": "предмет-1", "source": "scene"}],
        "opening_hook": "вступительный мотив к первому действию (1 предложение)",
    }

    return dedent(f"""
    {_HEADER}
    Ты генерируешь стартовый мир строго по схеме InitialWorldV2.
    Никаких ролей — они будут назначены отдельно.

    Комната/сессия: {title}
    Игроки: {players_line}
    {theme_block}

    Требования:
      • world_flags — только действительно нужные простые ключи (строки/числа/логика), без выдуманных API/интернета.
      • 2–3 осмысленных NPC с именами и предметами (без placeholder-имен).
      • 2–6 предметов сцены в available_items ({{"name","source":"scene"}}).
      • opening_hook должен естественно вызывать первое действие.
      • Тон: ясный, конкретный, без туманных метафор.

    Верни JSON строго соответствующий InitialWorldV2.
    Шаблон формы (значения придумай заново):
    {json.dumps(tmpl, ensure_ascii=False)}
    """).strip()

def build_roles_for_players_prompt(
    *,
    state: Dict[str, Any],
    players: List[Dict[str, Any]],
    story_theme: Optional[str] = None,
) -> str:
    snap = _short_world_snapshot(state)
    players_min = [{"player_id": str(p.get("player_id") or ""), "name": str(p.get("name") or "")} for p in players]
    theme_line = f"Тема: «{story_theme.strip()}». " if story_theme and story_theme.strip() else ""

    tmpl = {"roles_for_players": [{"player_id": "pid", "role": "роль", "summary": "кратко (1–2 предложения)"}]}
    parts: List[str] = []
    parts.append(_HEADER + "\n")
    parts.append("Сформируй роли 1:1 для игроков. Верни JSON ровно вида {\"roles_for_players\": [{...}]}.\n")
    parts.append(theme_line + "Роли должны вытекать из setting/location (и темы, если задана), помогать старту игры.\n\n")
    parts.append("Контекст мира (сжатый):\n")
    parts.append(json.dumps(snap, ensure_ascii=False) + "\n")
    parts.append("Игроки:\n")
    parts.append(json.dumps(players_min, ensure_ascii=False) + "\n\n")
    parts.append("СТРОГИЙ ШАБЛОН:\n")
    parts.append(json.dumps(tmpl, ensure_ascii=False))
    return "".join(parts)

def build_initial_backstory_prompt(
    *,
    state: Dict[str, Any],
    player: Dict[str, Any],
    story_theme: Optional[str] = None,
) -> str:
    snap = _short_world_snapshot(state)
    pid = player.get("player_id")
    role = ""
    try:
        for pl in (state.get("players") or []):
            if str(pl.get("player_id")) == str(pid):
                role = pl.get("role") or ""
                break
    except Exception:
        pass

    theme_line = f"Тема истории: «{story_theme.strip()}». " if story_theme and story_theme.strip() else ""
    role_line = f"Ваша роль: {role}. " if role else ""

    return dedent(f"""
    /no_think{_HEADER}
    Персональный ПРОЛОГ (PlayerStory) для одного игрока до начала игры. Верни JSON по схеме PlayerStory.
    {theme_line}{role_line}Пролог должен мягко подвести к стартовой сцене и мотивировать на первый ход.

    Требования к полям:
      • text — 6–10 предложений (2–3 абзаца), конкретный и читабельный, без «висячих» метафор и воды.
      • echo_of_action — краткая формулировка устремления героя (<=120 симв.), без кавычек и имён игроков.
      • highlights — 1–3 маркера важных деталей из пролога (если есть).

    Контекст мира (сжатый):
    {json.dumps(snap, ensure_ascii=False)}
    Игрок: {json.dumps({"player_id": pid, "name": player.get("name"), "role": player.get("role")}, ensure_ascii=False)}
    """).strip()
