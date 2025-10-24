# core/assimilator.py
from __future__ import annotations

"""
Ассимилятор эффектов LLM в состояние игры.

Основные функции:
- apply_effects(state, effects) -> dict             — применяет JSON-эффекты к состоянию
- append_raw(state, text)                           — добавляет черновой рассказ в raw_history
- append_general(state, text)                       — добавляет общий рассказ в general_history
- append_private(state, player_id, payload_dict)    — добавляет персональный рассказ в private_history

Формат ожидаемых effects (допускаются частично заполненные поля):
{
  "world_flags": {...},                   # dict — мержится
  "location": "Новая локация",            # str — если есть, подменяет state["location"]
  "scene_items_add": [ "фонарь", {"name":"ключ","source":"scene"} ],
  "scene_items_remove": [ "фонарь" ],
  "players": [
      {"player_id":"...", "hp_delta": -3, "items_add":["..."], "items_remove":["..."], "status_merge":{"bleeding":true}}
  ],
  "npcs": [
      {"npc_id":"npc_1", "hp_delta": -2, "items_add":["..."], "items_remove":["..."], "status_merge":{"angry":true}}
  ],
  "introductions": {
      "npcs": [ {"id":"npc_x","name":"...","mood":"...","hp":100,"items":["..."]}, ... ],
      "items": [ {"name":"...","source":"scene"}, "верёвка" ],
      "locations": [ "Подземный тоннель" ]
  }
}

Все неизвестные поля игнорируются, «кривые» значения пытаемся нормализовать.
"""

import logging
from typing import Any, Dict, List, Tuple

from .utils import ensure_inventory_invariants, normalize_text

logger = logging.getLogger(__name__)
logger.name = "core.assimilator"
logger.propagate = True


# ------------ Public helpers for history ------------

def append_raw(state: Dict[str, Any], text: str) -> None:
    """Добавить запись в raw_history (без форматирования)."""
    if not isinstance(state.get("raw_history"), list):
        state["raw_history"] = []
    txt = normalize_text(text)[:4000]
    if txt:
        state["raw_history"].append({"text": txt})


def append_general(state: Dict[str, Any], text: str) -> None:
    """Добавить запись в general_history."""
    if not isinstance(state.get("general_history"), list):
        state["general_history"] = []
    txt = normalize_text(text)[:4000]
    if txt:
        state["general_history"].append({"text": txt})


def append_private(state: Dict[str, Any], player_id: str, payload: Dict[str, Any]) -> None:
    """
    Добавить персональную запись в private_history.
    payload ожидается вида: {"text": str, "echo_of_action": str, "highlights": [str, ...]?}
    """
    if not isinstance(state.get("private_history"), list):
        state["private_history"] = []
    if not player_id:
        return
    text = normalize_text(payload.get("text", ""))[:4000]
    echo = normalize_text(payload.get("echo_of_action", ""))[:400]
    highlights = payload.get("highlights") or []
    entry = {"player_id": player_id}
    if text:
        entry["text"] = text
    if echo:
        entry["echo_of_action"] = echo
    if isinstance(highlights, list) and highlights:
        entry["highlights"] = [normalize_text(x)[:200] for x in highlights if normalize_text(x)]
    state["private_history"].append(entry)


# ------------ Core assimilation ------------

def apply_effects(state: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Применить эффекты к state. Возвращает краткую сводку по применённым изменениям.

    Ставит инварианты в конце (ensure_inventory_invariants).
    Безопасна к частично заполненным или «шумным» структурам effects.
    """
    if not isinstance(effects, dict):
        logger.warning("ASSIM: effects is not a dict — ignored")
        return {"applied": False, "reason": "effects_not_dict"}

    summary: Dict[str, Any] = {
        "applied": True,
        "world_flags_updated": 0,
        "location_changed": False,
        "scene_items_added": 0,
        "scene_items_removed": 0,
        "players_touched": 0,
        "npcs_touched": 0,
        "introduced_npcs": 0,
        "introduced_items": 0,
        "introduced_locations": 0,
    }

    # world_flags
    wf = effects.get("world_flags")
    if isinstance(wf, dict):
        updated = _merge_world_flags(state, wf)
        summary["world_flags_updated"] = updated

    # location
    loc = effects.get("location")
    if isinstance(loc, str):
        loc_n = normalize_text(loc)[:160]
        if loc_n and loc_n != state.get("location"):
            state["location"] = loc_n
            summary["location_changed"] = True

    # scene items
    added_cnt = _apply_scene_items_add(state, effects.get("scene_items_add"))
    removed_cnt = _apply_scene_items_remove(state, effects.get("scene_items_remove"))
    summary["scene_items_added"] = added_cnt
    summary["scene_items_removed"] = removed_cnt

    # players
    touched_players = _apply_players_effects(state, effects.get("players"))
    summary["players_touched"] = touched_players

    # npcs
    touched_npcs = _apply_npcs_effects(state, effects.get("npcs"))
    summary["npcs_touched"] = touched_npcs

    # introductions
    intro = effects.get("introductions")
    if isinstance(intro, dict):
        n_npcs, n_items, n_locs = _apply_introductions(state, intro)
        summary["introduced_npcs"] = n_npcs
        summary["introduced_items"] = n_items
        summary["introduced_locations"] = n_locs

    # инварианты
    ensure_inventory_invariants(state)

    # лог эффектов (сырые), если нужен трекинг
    if isinstance(state.get("effects_log"), list):
        # сохраняем компактно
        state["effects_log"].append(_compact_effects_for_log(effects))

    return summary


# ------------ Internals ------------

def _merge_world_flags(state: Dict[str, Any], new_flags: Dict[str, Any]) -> int:
    if not isinstance(state.get("world_flags"), dict):
        state["world_flags"] = {}
    out = 0
    for k, v in new_flags.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (str, int, float, bool)):
            prev = state["world_flags"].get(k)
            if prev != v:
                state["world_flags"][k] = v
                out += 1
    return out


def _scene_items(state: Dict[str, Any]) -> List[Dict[str, str]]:
    items = state.get("available_items")
    if not isinstance(items, list):
        state["available_items"] = []
    return state["available_items"]


def _normalize_item_entry(x: Any) -> Tuple[str, str]:
    """
    Нормализует элемент предмета к (name, source).
    Допускает строку или словарь {"name":..., "source":...}
    """
    if isinstance(x, dict):
        name = normalize_text(x.get("name", ""))[:80]
        source = normalize_text(x.get("source", "scene"))[:40] or "scene"
    else:
        name = normalize_text(str(x))[:80]
        source = "scene"
    return name, source


def _apply_scene_items_add(state: Dict[str, Any], payload: Any) -> int:
    if not isinstance(payload, list) or not payload:
        return 0
    scene = _scene_items(state)
    seen = { (normalize_text(it.get("name", "")).lower(), normalize_text(it.get("source","scene")).lower())
             for it in scene if isinstance(it, dict) and it.get("name") }
    added = 0
    for it in payload:
        name, source = _normalize_item_entry(it)
        if not name:
            continue
        key = (name.lower(), source.lower())
        if key in seen:
            continue
        scene.append({"name": name, "source": source})
        seen.add(key)
        added += 1
    return added


def _apply_scene_items_remove(state: Dict[str, Any], payload: Any) -> int:
    if not isinstance(payload, list) or not payload:
        return 0
    scene = _scene_items(state)
    if not scene:
        return 0
    removed = 0
    # допускаем строки и объекты
    to_remove_keys = set()
    for it in payload:
        name, source = _normalize_item_entry(it)
        if name:
            to_remove_keys.add((name.lower(), source.lower() if source else "scene"))

    # если указали только имя без источника — удаляем по имени (любой source)
    names_only = {name for (name, src) in to_remove_keys if src}
    if not names_only:
        # fallback на имена из payload (если приходила строка)
        names_only = {normalize_text(str(x)).lower() for x in payload if isinstance(x, str)}

    keep: List[Dict[str, str]] = []
    for it in scene:
        nm = normalize_text(it.get("name", "")).lower()
        src = normalize_text(it.get("source", "scene")).lower()
        key = (nm, src)
        if key in to_remove_keys or nm in names_only:
            removed += 1
            continue
        keep.append(it)
    state["available_items"] = keep
    return removed


def _players_index(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for p in state.get("players") or []:
        pid = str(p.get("player_id") or "")
        if pid and pid not in idx:
            idx[pid] = p
    return idx


def _apply_players_effects(state: Dict[str, Any], payload: Any) -> int:
    if not isinstance(payload, list) or not payload:
        return 0
    idx = _players_index(state)
    touched = 0
    for eff in payload:
        if not isinstance(eff, dict):
            continue
        pid = str(eff.get("player_id") or "")
        if not pid or pid not in idx:
            continue
        p = idx[pid]
        changed = False

        # hp_delta
        if isinstance(eff.get("hp_delta"), int):
            new_hp = int(p.get("hp", 100)) + int(eff["hp_delta"])
            max_hp = int(p.get("max_hp", 100) or 100)
            p["hp"] = max(0, min(max_hp, new_hp))
            changed = True

        # items_add
        for item in eff.get("items_add") or []:
            name, source = _normalize_item_entry(item)
            if not name:
                continue
            inv = p.get("items")
            if not isinstance(inv, list):
                p["items"] = inv = []
            # уникальность по имени
            if name.lower() not in {normalize_text(x).lower() if isinstance(x, str) else normalize_text(x.get("name","")).lower() for x in inv}:
                inv.append(name)
                changed = True

        # items_remove
        if eff.get("items_remove"):
            inv = p.get("items")
            if isinstance(inv, list) and inv:
                to_remove = {normalize_text(str(x)).lower() for x in eff["items_remove"]}
                keep = []
                removed_any = False
                for it in inv:
                    nm = normalize_text(it if isinstance(it, str) else str(it)).lower()
                    if nm in to_remove:
                        removed_any = True
                        continue
                    keep.append(it)
                if removed_any:
                    p["items"] = keep
                    changed = True

        # status_merge
        if isinstance(eff.get("status_merge"), dict):
            if not isinstance(p.get("status"), dict):
                p["status"] = {}
            for k, v in eff["status_merge"].items():
                if isinstance(k, str):
                    p["status"][k] = v
                    changed = True

        if changed:
            touched += 1
    return touched


def _npcs_index(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for n in state.get("npcs") or []:
        nid = str(n.get("id") or "")
        if nid and nid not in idx:
            idx[nid] = n
    return idx


def _apply_npcs_effects(state: Dict[str, Any], payload: Any) -> int:
    if not isinstance(payload, list) or not payload:
        return 0
    idx = _npcs_index(state)
    touched = 0
    for eff in payload:
        if not isinstance(eff, dict):
            continue
        nid = str(eff.get("npc_id") or eff.get("id") or "")
        if not nid or nid not in idx:
            continue
        npc = idx[nid]
        changed = False

        # hp_delta
        if isinstance(eff.get("hp_delta"), int):
            new_hp = int(npc.get("hp", 100)) + int(eff["hp_delta"])
            npc["hp"] = max(0, new_hp)
            changed = True

        # items_add
        for item in eff.get("items_add") or []:
            name, source = _normalize_item_entry(item)
            if not name:
                continue
            inv = npc.get("items")
            if not isinstance(inv, list):
                npc["items"] = inv = []
            if name.lower() not in {normalize_text(x).lower() if isinstance(x, str) else normalize_text(x.get("name","")).lower() for x in inv}:
                inv.append(name)
                changed = True

        # items_remove
        if eff.get("items_remove"):
            inv = npc.get("items")
            if isinstance(inv, list) and inv:
                to_remove = {normalize_text(str(x)).lower() for x in eff["items_remove"]}
                keep = []
                removed_any = False
                for it in inv:
                    nm = normalize_text(it if isinstance(it, str) else str(it)).lower()
                    if nm in to_remove:
                        removed_any = True
                        continue
                    keep.append(it)
                if removed_any:
                    npc["items"] = keep
                    changed = True

        # status_merge
        if isinstance(eff.get("status_merge"), dict):
            if not isinstance(npc.get("status"), dict):
                npc["status"] = {}
            for k, v in eff["status_merge"].items():
                if isinstance(k, str):
                    npc["status"][k] = v
                    changed = True

        if changed:
            touched += 1
    return touched


def _apply_introductions(state: Dict[str, Any], intro: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Возвращает (npcs_added, items_added, locations_added)
    """
    n_npcs = _intro_npcs(state, intro.get("npcs"))
    n_items = _apply_scene_items_add(state, intro.get("items"))
    n_locs = _intro_locations(state, intro.get("locations"))
    return n_npcs, n_items, n_locs


def _intro_npcs(state: Dict[str, Any], payload: Any) -> int:
    if not isinstance(payload, list) or not payload:
        return 0
    if not isinstance(state.get("npcs"), list):
        state["npcs"] = []
    existing_ids = {str(n.get("id")).lower() for n in state["npcs"] if isinstance(n.get("id"), (str, int))}
    added = 0
    for npc in payload:
        if not isinstance(npc, dict):
            continue
        nid = str(npc.get("id") or "").strip()
        name = normalize_text(npc.get("name", ""))[:80]
        if not nid or not name:
            continue
        if nid.lower() in existing_ids:
            continue
        # базовая нормализация
        entry = {
            "id": nid,
            "name": name,
            "mood": normalize_text(npc.get("mood", ""))[:40] if npc.get("mood") else "нейтральный",
            "hp": int(npc.get("hp", 100) or 100),
            "items": list(npc.get("items") or []),
        }
        state["npcs"].append(entry)
        existing_ids.add(nid.lower())
        added += 1
    return added


def _intro_locations(state: Dict[str, Any], payload: Any) -> int:
    """
    Список известных локаций храним в state["known_locations"] (создаём при необходимости).
    Если эффекты хотят «ввести» новую — просто добавляем в справочник. Текущую локацию не меняем.
    """
    if not isinstance(payload, list) or not payload:
        return 0
    kl = state.get("known_locations")
    if not isinstance(kl, list):
        state["known_locations"] = kl = []
    known = {normalize_text(x).lower() for x in kl if isinstance(x, str)}
    added = 0
    for loc in payload:
        if not isinstance(loc, str):
            continue
        nm = normalize_text(loc)
        if not nm:
            continue
        key = nm.lower()
        if key in known:
            continue
        kl.append(nm[:160])
        known.add(key)
        added += 1
    return added


def _compact_effects_for_log(effects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сжать effects для логов/телеметрии, обрезая потенциально длинные поля.
    """
    out: Dict[str, Any] = {}
    if isinstance(effects.get("world_flags"), dict):
        out["world_flags"] = effects["world_flags"]
    if isinstance(effects.get("location"), str):
        out["location"] = normalize_text(effects["location"])[:160]
    if isinstance(effects.get("scene_items_add"), list):
        out["scene_items_add"] = [_safe_name_or_dict(x) for x in effects["scene_items_add"]][:20]
    if isinstance(effects.get("scene_items_remove"), list):
        out["scene_items_remove"] = [_safe_name_or_dict(x) for x in effects["scene_items_remove"]][:20]
    if isinstance(effects.get("players"), list):
        out["players"] = [
            {k: v for k, v in it.items() if k in ("player_id", "hp_delta", "items_add", "items_remove")}
            for it in effects["players"]
            if isinstance(it, dict)
        ]
    if isinstance(effects.get("npcs"), list):
        out["npcs"] = [
            {k: v for k, v in it.items() if k in ("npc_id", "hp_delta", "items_add", "items_remove")}
            for it in effects["npcs"]
            if isinstance(it, dict)
        ]
    if isinstance(effects.get("introductions"), dict):
        intro = {}
        if isinstance(effects["introductions"].get("npcs"), list):
            intro["npcs"] = [
                {"id": normalize_text(n.get("id", ""))[:64], "name": normalize_text(n.get("name", ""))[:80]}
                for n in effects["introductions"]["npcs"]
                if isinstance(n, dict)
            ][:10]
        if isinstance(effects["introductions"].get("items"), list):
            intro["items"] = [_safe_name_or_dict(x) for x in effects["introductions"]["items"]][:20]
        if isinstance(effects["introductions"].get("locations"), list):
            intro["locations"] = [normalize_text(str(x))[:160] for x in effects["introductions"]["locations"] if str(x).strip()][:10]
        out["introductions"] = intro
    return out


def _safe_name_or_dict(x: Any) -> Any:
    if isinstance(x, dict):
        return {"name": normalize_text(x.get("name", ""))[:80]}
    return normalize_text(str(x))[:80]
