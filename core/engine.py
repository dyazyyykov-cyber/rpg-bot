from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ValidationError

from .llm import generate_json  # единая точка входа в LLM (strict JSON по схеме)
from .prompts import (
    build_beat_plan_prompt,
    build_effects_prompt,
    build_general_story_prompt,
    build_private_story_prompt,
    BANLIST_DEFAULT,
)
from .schemas import (
    BeatPlan,
    EffectsAndRawSchema,
    EffectsDelta,
    GeneralStory,
    GeneralStorySchema,
    PlayerStory,
    PlayerStorySchema,
    RawStory,
    TurnConfig,
    TurnOutputs,
)
from .utils import (
    ensure_dir,
    get_env,
    ensure_inventory_invariants,
    links_to_effects,
)

logger = logging.getLogger(__name__)
logger.propagate = True

# ---------------------------- Генераторные параметры ----------------------------
GENERAL_TEMPERATURE = get_env("GENERAL_TEMPERATURE", 0.7, float)
GENERAL_TOP_P = get_env("GENERAL_TOP_P", 0.9, float)
PRIVATE_TOP_P = get_env("PRIVATE_TOP_P", 0.9, float)
STRUCT_TOP_P = get_env("STRUCT_TOP_P", 0.0, float)

_LLMS = os.getenv("LLM_SEED_STRUCT")
try:
    LLM_SEED_STRUCT: Optional[int] = int(_LLMS) if _LLMS and _LLMS.strip() else None
except Exception:
    LLM_SEED_STRUCT = None

# ---------------------------- УТИЛИТЫ ----------------------------

def _g(obj: Any, key: str, default: Any = None) -> Any:
    """Безопасный доступ к полям dict/Pydantic/объектов."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "model_fields"):  # pydantic v2
        return getattr(obj, key, default)
    return getattr(obj, key, default)


def _write_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            if hasattr(obj, "model_dump"):
                json.dump(obj.model_dump(exclude_none=True), f, ensure_ascii=False, indent=2)  # type: ignore
            else:
                json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _truncate(s: Optional[str], n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n]


def _as_dict(x: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(x, BaseModel) or hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    return dict(x or {})


def _find_player(state: Dict[str, Any], player_id: str) -> Optional[Dict[str, Any]]:
    for p in state.get("players", []) or []:
        if str(p.get("player_id")) == str(player_id):
            return p
    return None


def _find_npc(state: Dict[str, Any], npc_id: str) -> Optional[Dict[str, Any]]:
    for n in state.get("npcs", []) or []:
        if str(n.get("id")) == str(npc_id):
            return n
    return None


def _players_maps_from_state(state: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    by_id: Dict[str, str] = {}
    by_name_lower: Dict[str, str] = {}
    for p in state.get("players", []) or []:
        pid = str(p.get("player_id") or "").strip()
        nm = str(p.get("name") or "").strip()
        if pid:
            by_id[pid] = nm
        if nm:
            by_name_lower[nm.lower()] = pid
    return by_id, by_name_lower


def _npcs_maps_from_state(state: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    by_id: Dict[str, str] = {}
    by_name_lower: Dict[str, str] = {}
    for n in state.get("npcs", []) or []:
        nid = str(n.get("id") or "").strip()
        nm = str(n.get("name") or "").strip()
        if nid:
            by_id[nid] = nm
        if nm:
            by_name_lower[nm.lower()] = nid
    return by_id, by_name_lower


def _relink_ids_in_effects(state: Dict[str, Any], eff: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(eff, ensure_ascii=False))  # deep copy
    p_by_id, p_by_name_lower = _players_maps_from_state(state)
    n_by_id, n_by_name_lower = _npcs_maps_from_state(state)

    for pd in out.get("players", []) or []:
        pid = str(pd.get("player_id") or "").strip()
        if not pid:
            continue
        if pid in p_by_id:
            continue
        mapped = p_by_name_lower.get(pid.lower())
        if mapped:
            pd["player_id"] = mapped

    for nd in out.get("npcs") or []:
        nid = str(nd.get("npc_id") or "").strip()
        if not nid:
            continue
        if nid in n_by_id:
            continue
        mapped = n_by_name_lower.get(nid.lower())
        if mapped:
            nd["npc_id"] = mapped

    return out


def _world_flags_has_jsonish_strings(world_flags: Dict[str, Any]) -> bool:
    try:
        for v in (world_flags or {}).values():
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") or s.startswith("["):
                    return True
    except Exception:
        return False
    return False

# ---------- Нормализация ответов LLM ----------

def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        try:
            return int(v)
        except Exception:
            return None
    if isinstance(v, str):
        try:
            return int(v.strip().replace("+", "").replace(",", ""))
        except Exception:
            return None
    if isinstance(v, dict):
        for key in ("current", "value", "now", "hp"):
            if key in v:
                return _to_int_or_none(v.get(key))
        return None
    return None


def _flatten_raw(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        return {"text": raw}
    d: Dict[str, Any] = dict(raw or {})
    t = d.get("text")
    if t is not None:
        return {"text": str(t)}
    log = d.get("log")
    if isinstance(log, list) and log:
        parts: List[str] = []
        for entry in log:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                text = entry.get("text")
                if text:
                    parts.append(str(text))
        return {"text": "\n".join([p for p in parts if p.strip()])}
    return {"text": ""}


def _normalize_effects_and_raw(obj: Any) -> Dict[str, Any]:
    d = dict(obj or {})
    eff = dict(d.get("effects") or {})
    d["effects"] = eff

    eff.setdefault("world_flags", {})
    eff.setdefault("location", None)
    eff.setdefault("scene_items_add", [])
    eff.setdefault("scene_items_remove", [])
    eff.setdefault("players", [])
    eff.setdefault("npcs", [])
    eff.setdefault("introductions", {"npcs": [], "items": [], "locations": []})

    # players
    norm_players: List[Dict[str, Any]] = []
    for pd in eff.get("players") or []:
        if not isinstance(pd, dict):
            continue
        pd = dict(pd)
        if not pd.get("player_id") and pd.get("id"):
            pd["player_id"] = str(pd.pop("id"))
        if "hp" in pd:
            hp_int = _to_int_or_none(pd.get("hp"))
            if hp_int is None:
                pd.pop("hp", None)
            else:
                pd["hp"] = hp_int
        if "hp_delta" in pd:
            dlt = _to_int_or_none(pd.get("hp_delta"))
            if dlt is None:
                pd.pop("hp_delta", None)
            else:
                pd["hp_delta"] = dlt
        for k in ("items_add", "items_remove"):
            if k in pd and not isinstance(pd[k], list):
                pd[k] = [pd[k]]
        norm_players.append(pd)
    eff["players"] = norm_players

    # npcs
    norm_npcs: List[Dict[str, Any]] = []
    for nd in eff.get("npcs") or []:
        if not isinstance(nd, dict):
            continue
        nd = dict(nd)
        if not nd.get("npc_id") and nd.get("id"):
            nd["npc_id"] = str(nd.pop("id"))
        if "hp" in nd:
            hp_int = _to_int_or_none(nd.get("hp"))
            if hp_int is None:
                nd.pop("hp", None)
            else:
                nd["hp"] = hp_int
        if "hp_delta" in nd:
            dlt = _to_int_or_none(nd.get("hp_delta"))
            if dlt is None:
                nd.pop("hp_delta", None)
            else:
                nd["hp_delta"] = dlt
        for k in ("items_add", "items_remove"):
            if k in nd and not isinstance(nd[k], list):
                nd[k] = [nd[k]]
        norm_npcs.append(nd)
    eff["npcs"] = norm_npcs

    # dedupe: если предмет и в add, и в remove — оставить только remove
    try:
        add_names = {str(x.get("name", "")).strip().lower() for x in (eff.get("scene_items_add") or []) if isinstance(x, dict)}
        rm_names = {str(x).strip().lower() for x in (eff.get("scene_items_remove") or [])}
        clash = add_names & rm_names
        if clash:
            eff["scene_items_add"] = [x for x in (eff.get("scene_items_add") or [])
                                      if str(x.get("name", "")).strip().lower() not in clash]
    except Exception:
        pass

    d["raw"] = _flatten_raw(d.get("raw") or d.get("raw_story"))
    return d


def _normalize_player_story(obj: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = obj if isinstance(obj, dict) else {}
    t = d.get("text")
    d["text"] = "" if t is None else str(t)
    hl = d.get("highlights")
    if hl is None:
        pass
    elif isinstance(hl, list):
        d["highlights"] = [str(x) for x in hl if str(x).strip()]
    elif isinstance(hl, dict):
        d["highlights"] = [str(k) for k, v in hl.items() if v or v is None]
    else:
        d["highlights"] = [str(hl)]
    e = d.get("echo_of_action")
    if e is None:
        eo = ""
    elif isinstance(e, list):
        eo = str(e[0]) if e else ""
    elif isinstance(e, dict):
        eo = str(e.get("text") or "")
    else:
        eo = str(e)
    d["echo_of_action"] = _truncate(eo, 240)
    return d


def _normalize_general_story(obj: Any) -> Dict[str, Any]:
    d = dict(obj or {})
    t = d.get("text")
    d["text"] = "" if t is None else str(t)
    hl = d.get("highlights")
    if hl is None:
        return d
    if isinstance(hl, list):
        d["highlights"] = [str(x) for x in hl if str(x).strip()]
    elif isinstance(hl, dict):
        d["highlights"] = [str(k) for k, v in hl.items() if v or v is None]
    else:
        d["highlights"] = [str(hl)]
    return d

# ---------------------------- Изменение состояния ----------------------------

def apply_effects(state: Dict[str, Any], effects: Union[EffectsDelta, Dict[str, Any]]) -> Dict[str, Any]:
    """Применяет дельты к state и соблюдает инварианты инвентаря."""
    if "effects_log" not in state:
        state["effects_log"] = []

    e = _as_dict(effects)

    # Мир / локация
    wf = e.get("world_flags") or {}
    if wf:
        if "world_flags" not in state or not isinstance(state["world_flags"], dict):
            state["world_flags"] = {}
        state["world_flags"].update({str(k): str(v) for k, v in wf.items()})

    if e.get("location"):
        state["location"] = _truncate(str(e["location"]), 160)

    # Предметы сцены
    if e.get("scene_items_add"):
        avail = state.setdefault("available_items", [])
        for it in e["scene_items_add"]:
            it = dict(it or {})
            name_raw = it.get("name")
            name = _truncate(str(name_raw or ""), 64)
            if not name:
                continue
            avail.append({"name": name, "source": "scene"})

    if e.get("scene_items_remove") and state.get("available_items"):
        to_rm = {str(x).strip().lower() for x in e["scene_items_remove"]}
        state["available_items"] = [
            it for it in state["available_items"] if str(it.get("name", "")).strip().lower() not in to_rm
        ]

    # Игроки
    for pd in (e.get("players") or []):
        pid = pd.get("player_id")
        if not pid:
            continue
        pl = _find_player(state, pid)
        if not pl:
            continue

        if "hp" in pd and pd["hp"] is not None:
            try:
                pl["hp"] = max(0, int(pd["hp"]))
            except Exception:
                pass
        elif "hp_delta" in pd and pd["hp_delta"] is not None:
            try:
                pl["hp"] = max(0, int(pl.get("hp", 0)) + int(pd["hp_delta"]))
            except Exception:
                pass

        if "max_hp" in pl:
            try:
                pl["hp"] = min(int(pl.get("hp", 0)), int(pl["max_hp"]))
            except Exception:
                pass

        if pd.get("status_apply"):
            st = pl.setdefault("status", {})
            for k, v in (pd["status_apply"] or {}).items():
                st[str(k)] = v

        if pd.get("items_add"):
            inv = pl.setdefault("items", [])
            for x in pd["items_add"]:
                nm = _truncate(str(x), 64)
                if nm:
                    inv.append(nm)
        if pd.get("items_remove"):
            inv = pl.setdefault("items", [])
            rm = {str(x).strip().lower() for x in pd["items_remove"]}
            pl["items"] = [x for x in inv if str(x).strip().lower() not in rm]

        if pd.get("position"):
            pl["position"] = _truncate(str(pd["position"]), 64)

    # NPC
    for nd in (e.get("npcs") or []):
        nid = nd.get("npc_id")
        if not nid:
            continue
        npc = _find_npc(state, nid)
        if not npc:
            continue

        if "hp" in nd and nd["hp"] is not None:
            try:
                npc["hp"] = max(0, int(nd["hp"]))
            except Exception:
                pass
        elif "hp_delta" in nd and nd["hp_delta"] is not None:
            try:
                npc["hp"] = max(0, int(npc.get("hp", 0)) + int(nd["hp_delta"]))
            except Exception:
                pass

        if nd.get("mood"):
            npc["mood"] = _truncate(str(nd.get("mood")), 24)

        if nd.get("items_add"):
            inv = npc.setdefault("items", [])
            for x in nd["items_add"]:
                nm = _truncate(str(x), 64)
                if nm:
                    inv.append(nm)
        if nd.get("items_remove"):
            inv = npc.setdefault("items", [])
            rm = {str(x).strip().lower() for x in nd["items_remove"]}
            npc["items"] = [x for x in inv if str(x).strip().lower() not in rm]

    ensure_inventory_invariants(state)
    state.setdefault("effects_log", []).append(e)
    return state


def push_raw_story(state: Dict[str, Any], raw: Union[RawStory, Dict[str, Any]]) -> None:
    state.setdefault("raw_history", []).append(_as_dict(raw))


def _last_private_entries_for_pid(state: dict, pid: str, k: int = 1):
    ph = state.get("private_history") or []
    if isinstance(ph, dict):
        lst = ph.get(pid) or []
        return lst[-k:] if isinstance(lst, list) else ([] if lst is None else [lst])[-k:]
    if isinstance(ph, list):
        items = [x for x in ph if isinstance(x, dict) and str(x.get("player_id")) == str(pid)]
        return items[-k:]
    return []


def push_private_story(state: dict, player_id: str, story, *, keep_last: int = 50) -> None:
    entry = _as_dict(story)
    entry["player_id"] = str(player_id)

    ph = state.get("private_history")
    if isinstance(ph, list):
        ph.append(entry)
        if keep_last and len(ph) > keep_last:
            ph = ph[-keep_last:]
        state["private_history"] = ph
        return

    if isinstance(ph, dict):
        lst = ph.get(player_id)
        if not isinstance(lst, list):
            lst = []
        lst.append(entry)
        if keep_last and len(lst) > keep_last:
            lst = lst[-keep_last:]
        ph[player_id] = lst
        state["private_history"] = ph
        return

    state["private_history"] = [entry]


def push_general_story(state: Dict[str, Any], story: Union[GeneralStory, Dict[str, Any]]) -> None:
    state.setdefault("general_history", []).append(_as_dict(story))

# ---------------------------- Факты для General ----------------------------

def _extract_facts_for_general(
    state: Dict[str, Any],
    eff_for_apply: Dict[str, Any],
    privates: Dict[str, PlayerStory],
    plan_obj: Any,
    actions_public: List[Dict[str, str]],
) -> List[str]:
    facts: List[str] = []
    p_by_id, _ = _players_maps_from_state(state)
    n_by_id, _ = _npcs_maps_from_state(state)

    loc = eff_for_apply.get("location")
    if loc:
        facts.append(f"Сцена происходит: {str(loc)}")

    for pd in (eff_for_apply.get("players") or []):
        pid = str(pd.get("player_id") or "")
        name = p_by_id.get(pid) or "игрок"
        for it in (pd.get("items_add") or []):
            facts.append(f"{name} получил предмет: {str(it)}")
        for it in (pd.get("items_remove") or []):
            facts.append(f"{name} больше не имеет предмет: {str(it)}")

    for nd in (eff_for_apply.get("npcs") or []):
        nid = str(nd.get("npc_id") or "")
        npc = n_by_id.get(nid) or nid or "NPC"
        mood = nd.get("mood")
        if mood:
            facts.append(f"Настроение {npc}: {str(mood)}")

    wf = (eff_for_apply.get("world_flags") or {})
    if wf.get("time_of_day"):
        facts.append(f"Время суток: {wf['time_of_day']}")
    if wf.get("weather"):
        facts.append(f"Погода: {wf['weather']}")

    for st in (privates or {}).values():
        for h in (getattr(st, "highlights", None) or []):
            s = str(h).strip()
            if s:
                facts.append(s)

    for m in (_g(plan_obj, "must_mention", []) or []):
        s = str(m).strip()
        if s:
            facts.append(s)

    for a in actions_public:
        if a.get("player") and a.get("text"):
            facts.append(f"{a['player']} сделал: {a['text']}")

    facts = list(dict.fromkeys([f for f in facts if f]))[:12]
    return facts

# ---------------------------- ОРКЕСТРАЦИЯ ХОДА ----------------------------

async def run_turn_with_llm(
    state: Dict[str, Any],
    actions: List[Dict[str, Any]],
    cfg: TurnConfig,
) -> TurnOutputs:
    # Санитизация входных actions
    norm_actions: List[Dict[str, Any]] = []
    for a in actions or []:
        if isinstance(a, dict):
            norm_actions.append({
                "player_id": str(a.get("player_id") or ""),
                "player_name": str(a.get("player_name") or ""),
                "text": str(a.get("text") or ""),
            })
        else:
            norm_actions.append({
                "player_id": str(getattr(a, "player_id", "") or ""),
                "player_name": str(getattr(a, "player_name", "") or ""),
                "text": str(getattr(a, "text") or ""),
            })

    public_names: List[str] = []
    for p in state.get("players", []) or []:
        nm = str(p.get("name") or p.get("role") or "").strip()
        if nm:
            public_names.append(nm)

    sid = str(getattr(cfg, "group_chat_id", "unknown"))
    turn_no = int(state.get("turn", 0))
    turn_dir = os.path.join(".", "logs", f"session_{sid}", f"turn_{turn_no:04d}")
    ensure_dir(turn_dir)
    llm_dir = os.path.join(turn_dir, "llm")
    ensure_dir(llm_dir)

    telemetry: Dict[str, Any] = {}

    # Бан-лист: дефолт + из state
    extra_forbidden: List[str] = []
    try:
        extra_forbidden = list(state.get("forbidden", []) or [])
    except Exception:
        extra_forbidden = []
    banlist = list(dict.fromkeys([*BANLIST_DEFAULT, *extra_forbidden]))

    # Тема сцены (из лобби)
    theme = (state.get("story_theme") or "").strip() or None

    # ---------------- PLAN ----------------
    plan_prompt = build_beat_plan_prompt(
        state=state,
        actions=norm_actions,
        open_threads=state.get("open_threads"),
        clocks=state.get("clocks"),
        forbidden=banlist,
        theme=theme,
    )
    try:
        with open(os.path.join(turn_dir, "plan_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(plan_prompt)
    except Exception:
        pass

    logger.info("TURN[%s #%d]: request BeatPlan", sid, turn_no)

    plan_obj = await generate_json(
        model=None,
        schema=BeatPlan,
        prompt=plan_prompt,
        temperature=float(getattr(cfg, "temperature_effects", 0.35)),
        top_p=float(getattr(cfg, "top_p_effects", STRUCT_TOP_P or 0.0)),
        seed=LLM_SEED_STRUCT,
        timeout=getattr(cfg, "llm_timeout", None),
        log_prefix="PLAN",
        log_dir=llm_dir,
        forbidden=banlist,
        max_retries=1,
    )
    _write_json(os.path.join(turn_dir, "plan_content.json"), plan_obj)

    # ---------------- EFFECTS+RAW ----------------
    last_raw_tail = (state.get("raw_history") or [])[-1:]

    effects_prompt = build_effects_prompt(
        state=state,
        actions=norm_actions,
        last_raw=last_raw_tail,
        forbidden=list(dict.fromkeys([*banlist, *(_g(plan_obj, "forbidden", []) or [])])),
        beat_plan=_as_dict(plan_obj),
        theme=theme,
    )
    try:
        with open(os.path.join(turn_dir, "effects_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(effects_prompt)
    except Exception:
        pass

    logger.info("TURN[%s #%d]: request Effects+Raw", sid, turn_no)

    eff_raw_dict = await generate_json(
        model=None,
        schema=EffectsAndRawSchema,
        prompt=effects_prompt,
        temperature=float(getattr(cfg, "temperature_effects", 0.35)),
        top_p=float(getattr(cfg, "top_p_effects", STRUCT_TOP_P or 0.0)),
        seed=LLM_SEED_STRUCT,
        timeout=getattr(cfg, "llm_timeout", None),
        log_prefix="EFFECTS",
        log_dir=llm_dir,
        forbidden=banlist,
        require_nonempty_effects=True,
        actions_for_effects=norm_actions,
        max_retries=1,
    )
    _write_json(os.path.join(turn_dir, "effects_response_raw.json"), eff_raw_dict)

    eff_raw_norm = _normalize_effects_and_raw(eff_raw_dict)
    _write_json(os.path.join(turn_dir, "effects_response_norm.json"), eff_raw_norm)

    try:
        eff_raw_model = EffectsAndRawSchema.model_validate(eff_raw_norm)
    except ValidationError as ve:
        logger.error("EffectsAndRaw validation failed: %s", ve)
        raise

    if _world_flags_has_jsonish_strings(eff_raw_model.effects.world_flags):
        logger.error("EffectsAndRaw: world_flags contains JSON-like strings")
        raise ValidationError.from_exception_data("EffectsDelta", [])

    eff_dict = eff_raw_model.effects.model_dump(exclude_none=True)
    eff_relinked = _relink_ids_in_effects(state, eff_dict)
    _write_json(os.path.join(turn_dir, "effects_relinked.json"), eff_relinked)

    try:
        effects_model = EffectsDelta.model_validate(eff_relinked)
    except ValidationError as ve:
        logger.error("EffectsDelta validation failed after relink: %s", ve)
        logger.error("Offending relinked: %s", json.dumps(eff_relinked, ensure_ascii=False)[:2000])
        raise

    raw_story_model: RawStory = eff_raw_model.raw
    _write_json(os.path.join(turn_dir, "raw_story.json"), _as_dict(raw_story_model))

    # применяем эффекты к state + пушим raw
    # защитим дельты игроков/NPC от «съедания» валидатором
    eff_for_apply: Dict[str, Any] = effects_model.model_dump(exclude_none=True)
    if (not eff_for_apply.get("players")) and (eff_relinked.get("players")):
        eff_for_apply["players"] = eff_relinked.get("players") or []
    if (not eff_for_apply.get("npcs")) and (eff_relinked.get("npcs")):
        eff_for_apply["npcs"] = eff_relinked.get("npcs") or []

    apply_effects(state, eff_for_apply)
    push_raw_story(state, raw_story_model)
    _write_json(os.path.join(turn_dir, "state_after_effects.json"), state)

    # ---------------- PRIVATE ----------------
    private_outputs: Dict[str, PlayerStory] = {}
    for p in state.get("players", []) or []:
        pid = p.get("player_id")
        if not pid:
            continue
        player_tail = _last_private_entries_for_pid(state, pid, k=1)

        priv_prompt = build_private_story_prompt(
            state=state,
            player=p,
            raw_story=_as_dict(raw_story_model),
            last_private=player_tail,
            public_names=public_names or None,
            forbidden=banlist,
            effects=eff_relinked,
        )
        try:
            with open(os.path.join(turn_dir, f"private_prompt_{pid}.txt"), "w", encoding="utf-8") as f:
                f.write(priv_prompt)
        except Exception:
            pass

        logger.info("TURN[%s #%d]: request PrivateStory for player=%s", sid, turn_no, pid)

        try:
            priv_dict = await generate_json(
                model=None,
                schema=PlayerStorySchema,
                prompt=priv_prompt,
                temperature=float(getattr(cfg, "temperature_private", 0.55)),
                top_p=float(getattr(cfg, "top_p_private", PRIVATE_TOP_P or 0.9)),
                timeout=getattr(cfg, "llm_timeout", None),
                log_prefix=f"PRIVATE:{pid}",
                log_dir=llm_dir,
                forbidden=banlist,
                max_retries=1,
            )
        except Exception as e:
            logger.warning("PlayerStory LLM error (pid=%s): %s", pid, e)
            continue

        _write_json(os.path.join(turn_dir, f"private_response_{pid}.json"), priv_dict)
        priv_norm = _normalize_player_story(priv_dict)

        # жёсткий фоллбэк: пустые строки заполняем действием игрока
        if not (priv_norm.get("text", "").strip()) or not (priv_norm.get("echo_of_action", "").strip()):
            last_act = next((a.get("text") for a in norm_actions if str(a.get("player_id")) == str(pid)), "")
            if not priv_norm.get("text", "").strip():
                priv_norm["text"] = (f"Ты {last_act}." if last_act else "Ты осматриваешься и оцениваешь обстановку.")
            if not priv_norm.get("echo_of_action", "").strip():
                priv_norm["echo_of_action"] = (last_act or "осматриваюсь")

        try:
            candidate = PlayerStory.model_validate(priv_norm)
        except ValidationError as ve:
            # крайний случай: собрать минимально валидный ответ
            logger.warning("PlayerStory validation failed (pid=%s): %s — using hard fallback", pid, ve)
            last_act = next((a.get("text") for a in norm_actions if str(a.get("player_id")) == str(pid)), "действуешь осторожно")
            candidate = PlayerStory(text=f"Ты {last_act}.", echo_of_action=last_act)

        if not (candidate.text or "").strip():
            logger.warning("PlayerStory empty text (pid=%s)", pid)
            continue

        push_private_story(state, str(pid), candidate)
        private_outputs[str(pid)] = candidate
        _write_json(os.path.join(turn_dir, f"private_story_{pid}.json"), _as_dict(candidate))

    # ---------------- GENERAL ----------------
    general_tail = (state.get("general_history") or [])[-1:]

    p_by_id, _ = _players_maps_from_state(state)
    actions_public: List[Dict[str, str]] = []
    for a in norm_actions:
        pid = str(a.get("player_id") or "")
        name = p_by_id.get(pid) or str(a.get("player_name") or "").strip()
        actions_public.append({"player": name, "text": str(a.get("text") or "").strip()})

    private_echos_list: List[str] = []
    for st in private_outputs.values():
        eo = (st.echo_of_action or "").strip()
        if eo:
            private_echos_list.append(eo)

    world_snapshot = {
        "setting": state.get("setting"),
        "location": state.get("location"),
        "world_flags": state.get("world_flags"),
        "npcs": state.get("npcs"),
        "available_items": state.get("available_items"),
        "players": state.get("players"),
    }

    # факты для обязательного включения
    facts = _extract_facts_for_general(state, eff_for_apply, private_outputs, plan_obj, actions_public)

    gen_prompt = build_general_story_prompt(
        snapshot=world_snapshot,
        raw_story=str(getattr(raw_story_model, "text", "") or ""),
        prev_general_tail=[(g.get("text", "") if isinstance(g, dict) else str(getattr(g, "text", ""))) for g in general_tail],
        actions_public=actions_public,
        private_echos=private_echos_list,
        effects=eff_relinked,
        open_threads=state.get("open_threads") or [],
        beat_plan=_as_dict(plan_obj),
        facts=facts,
        theme=theme,
        forbidden=banlist,
    )
    try:
        with open(os.path.join(turn_dir, "general_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(gen_prompt)
    except Exception:
        pass

    logger.info("TURN[%s #%d]: request GeneralStory", sid, turn_no)

    gen_dict = await generate_json(
        model=None,
        schema=GeneralStorySchema,
        prompt=gen_prompt,
        temperature=float(getattr(cfg, "temperature_general", GENERAL_TEMPERATURE or 0.7)),
        top_p=float(getattr(cfg, "top_p_general", GENERAL_TOP_P or 0.9)),
        timeout=getattr(cfg, "llm_timeout", None),
        log_prefix="GENERAL",
        log_dir=llm_dir,
        forbidden=banlist,
        max_retries=1,
    )

    _write_json(os.path.join(turn_dir, "general_response_raw.json"), gen_dict)
    gen_norm = _normalize_general_story(gen_dict)

    try:
        general_model: GeneralStory = GeneralStory.model_validate(gen_norm)
    except ValidationError as ve:
        logger.error("GeneralStory validation failed: %s", ve)
        raise

    if not (general_model.text or "").strip():
        logger.error("GeneralStory empty text")
        raise ValidationError.from_exception_data("GeneralStory", [])

    # Проверка покрытия эффектов и фактов
    def _missing_facts(text: str, facts_list: List[str]) -> List[str]:
        low = (text or "").lower()
        miss = []
        for f in facts_list:
            s = str(f).strip()
            if s and s.lower() not in low:
                miss.append(s)
        return miss

    weak_links = not links_to_effects(general_model.text, eff_relinked, snapshot=world_snapshot, min_refs=2)
    missing = _missing_facts(general_model.text, facts)

    if weak_links or missing:
        logger.warning("GeneralStory lacks links/facts — retrying with stronger hint")
        todo = ""
        if missing:
            bullets = "\n".join(f"- {m}" for m in missing)
            todo = f"\n\nОБЯЗАТЕЛЬНО включи каждую из следующих деталей (словесно и явно):\n{bullets}\n"
        hint = (
            "\n\nВНИМАНИЕ: привяжи текст к эффектам (называй вещи своими именами, без метафор). "
            "Пиши короткими ясными предложениями." + todo
        )

        gen_dict_retry = await generate_json(
            model=None,
            schema=GeneralStorySchema,
            prompt=(gen_prompt + hint),
            temperature=float(getattr(cfg, "temperature_general", GENERAL_TEMPERATURE or 0.7)),
            top_p=float(getattr(cfg, "top_p_general", GENERAL_TOP_P or 0.9)),
            timeout=getattr(cfg, "llm_timeout", None),
            log_prefix="GENERAL_RETRY",
            log_dir=llm_dir,
            forbidden=banlist,
            max_retries=0,
        )
        _write_json(os.path.join(turn_dir, "general_response_retry.json"), gen_dict_retry)
        gen_norm = _normalize_general_story(gen_dict_retry)
        general_model = GeneralStory.model_validate(gen_norm)

    # фиксируем общий абзац
    push_general_story(state, general_model)
    _write_json(os.path.join(turn_dir, "general_story.json"), _as_dict(general_model))

    # ---------------- NEXT TURN + обновление нитей ----------------
    plan_threads = [str(x).strip() for x in (_g(plan_obj, "open_threads_next", []) or []) if str(x).strip()]
    if plan_threads:
        seen = set([str(x).strip() for x in (state.get("open_threads") or []) if str(x).strip()])
        for t in plan_threads:
            if t not in seen:
                (state.setdefault("open_threads", [])).append(t)
                seen.add(t)

    state["turn"] = int(state.get("turn", 0)) + 1
    _write_json(os.path.join(turn_dir, "final_state.json"), state)

    # ---------------- OUT ----------------
    out = TurnOutputs(
        effects=EffectsDelta.model_validate(eff_relinked),
        raw=raw_story_model,
        players_private={k: v for k, v in private_outputs.items() },
        general=general_model,
        turn=int(state["turn"]),
        telemetry={},
    )
    return out
