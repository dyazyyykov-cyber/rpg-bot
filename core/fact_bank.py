# fact_bank.py
# -*- coding: utf-8 -*-
"""
FactBank — лёгкий слой консистентности для нарратива.

Задачи:
- Хранить проверяемые "факты" (subject, predicate, object, scope, confidence, turn).
- Устранять противоречия: для ряда предикатов "единственность" (напр. location, status, time_of_day).
- Выдавать сводки фактов для LLM-промптов (общие и приватные).
- Автоматически извлекать и обновлять факты из effects/state (world_flags, предметы, статусы и т.п.).
- Позволять мягкую дедупликацию и "суперсиды" (новые факты замещают устаревшие).

Подключение:
    from fact_bank import FactBank, Fact

    fb = FactBank.from_state(state)          # загрузить из state["fact_bank"]
    fb.ingest_state_snapshot(state, turn)    # первичная инициализация (без дубликатов)
    fb.apply_effects(effects, turn)          # обновить фактами из effects

    # Сводки для промптов:
    bullets = fb.render_prompt_bullets(state, audience="general", max_facts=12)
    bullets_private = fb.render_prompt_bullets(state, audience="private", player_id=pid)

    # Сохранить обратно:
    state["fact_bank"] = fb.export()

Соглашения:
- subject: "world", "scene", "player:<pid>", "npc:<id>", "item:<slug>"
- predicates: свободные строки, но некоторые считаются "уникальными":
  UNIQUE_PREDICATES = {"location","status","time_of_day","mood","holding","world_flag"}
- scope: "world" | "scene" | "private"
- object: примитив/словарь; для ссылок на сущности рекомендуем object={"ref":"npc:npc_1","name":"..."}
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Iterable, Tuple
import re
import uuid
import json
import copy


# --------------------------
# Вспомогательные утилиты
# --------------------------

def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _slug(s: str) -> str:
    s = _lc(s)
    s = re.sub(r"[^a-z0-9а-яё_ -]+", "", s)
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s[:64]

def _is_truthy(x: Any) -> bool:
    return not (x is None or x is False or x == "")

def _deepcopy(x: Any) -> Any:
    try:
        return copy.deepcopy(x)
    except Exception:
        return json.loads(json.dumps(x, ensure_ascii=False))  # грубый фоллбэк


# --------------------------------
# Модель факта и банк фактов
# --------------------------------

UNIQUE_PREDICATES = {
    "location",
    "status",
    "time_of_day",
    "mood",
    "holding",      # кто держит (универсальный "владение/в руках")
    "world_flag",   # world_flag:<key> сведём сюда
}

APPEND_ONLY_PREDICATES = {
    "event",        # любые событийные отметки (не заменяем)
    "memory",       # "память/заметки"
}

@dataclass
class Fact:
    id: str
    subject: str              # "player:<pid>" | "npc:<id>" | "world" | ...
    predicate: str            # "location" | "status" | ...
    object: Any               # значение (строка/число/словарь), допускается {"ref":"npc:npc_1","name":"..."}
    scope: str                # "world" | "scene" | "private"
    confidence: float = 0.9   # 0..1
    turn: int = 0
    source: str = "system"    # "llm" | "storylet" | "system"
    private_to: Optional[str] = None   # player_id для приватных
    supersedes: Optional[str] = None   # id факта, который замещён этим
    superseded_by: Optional[str] = None
    ttl_turns: Optional[int] = None    # если задано — истекает через N ходов от turn
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now_turn: int) -> bool:
        if self.ttl_turns is None:
            return False
        return now_turn >= (self.turn + self.ttl_turns)

    def key_for_uniqueness(self) -> Tuple[str, str, str, Optional[str]]:
        """
        Ключ, по которому определяем "конфликт" для UNIQUE_PREDICATES.
        """
        return (self.subject, self.predicate, self.scope, self.private_to)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Fact":
        return Fact(**d)


class FactBank:
    def __init__(self, facts: Optional[List[Fact]] = None):
        self._facts: List[Fact] = list(facts or [])

    # ------------- CRUD -------------

    def add(self, fact: Fact, *, replace_policy: str = "unique") -> Fact:
        """
        Добавить факт. Если predicate уникален — предыдущие конфликтующие факты помечаются superseded_by.
        replace_policy:
            "unique"  — применять правило уникальности к UNIQUE_PREDICATES
            "append"  — всегда просто добавлять (событийная лента)
        """
        if replace_policy == "unique" and fact.predicate in UNIQUE_PREDICATES:
            self._supersede_conflicts(fact)
        self._facts.append(fact)
        return fact

    def add_many(self, facts: Iterable[Fact], **kw) -> None:
        for f in facts:
            self.add(f, **kw)

    def _supersede_conflicts(self, new_fact: Fact):
        key = new_fact.key_for_uniqueness()
        for f in self._facts:
            if f.superseded_by is None and f.key_for_uniqueness() == key:
                # если объект совпадает — ничего не делаем
                if json.dumps(f.object, ensure_ascii=False, sort_keys=True) == json.dumps(new_fact.object, ensure_ascii=False, sort_keys=True):
                    continue
                f.superseded_by = new_fact.id
                new_fact.supersedes = f.id

    def retract_where(self, *, subject: Optional[str] = None, predicate: Optional[str] = None) -> int:
        """
        Мягкое снятие фактов: помечаем superseded_by="retract:<uuid>" все подходящие.
        Возвращает количество ретрактов.
        """
        tag = f"retract:{uuid.uuid4().hex[:8]}"
        cnt = 0
        for f in self._facts:
            if f.superseded_by is not None:
                continue
            if subject and f.subject != subject:
                continue
            if predicate and f.predicate != predicate:
                continue
            f.superseded_by = tag
            cnt += 1
        return cnt

    def all(self, *, include_superseded: bool = False) -> List[Fact]:
        if include_superseded:
            return list(self._facts)
        return [f for f in self._facts if f.superseded_by is None]

    # --------- Импорт/экспорт ---------

    def export(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self._facts]

    @staticmethod
    def from_export(data: List[Dict[str, Any]]) -> "FactBank":
        return FactBank([Fact.from_dict(x) for x in data or []])

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "FactBank":
        return FactBank.from_export(state.get("fact_bank") or [])

    def save_into_state(self, state: Dict[str, Any]) -> None:
        state["fact_bank"] = self.export()

    # ------------- Запросы -------------

    def query(self,
              *,
              subjects: Optional[Iterable[str]] = None,
              predicates: Optional[Iterable[str]] = None,
              scope: Optional[str] = None,
              private_for: Optional[str] = None,
              include_superseded: bool = False) -> List[Fact]:
        subs = set(subjects) if subjects else None
        preds = set(predicates) if predicates else None
        items = self._facts if include_superseded else self.all()
        out: List[Fact] = []
        for f in items:
            if subs and f.subject not in subs:
                continue
            if preds and f.predicate not in preds:
                continue
            if scope and f.scope != scope:
                continue
            if private_for is not None and f.private_to != private_for:
                continue
            out.append(f)
        return out

    def sweep_expired(self, now_turn: int) -> int:
        """
        Мягко "гасим" истёкшие факты (superseded_by=expire:<turn>)
        """
        cnt = 0
        for f in self._facts:
            if f.superseded_by is None and f.is_expired(now_turn):
                f.superseded_by = f"expire:{now_turn}"
                cnt += 1
        return cnt

    # --------- Пополнение из state/effects ---------

    def ingest_state_snapshot(self, state: Dict[str, Any], turn: int) -> None:
        """
        Однократная инициализация фактами из текущего state (без агрессивной замены).
        """
        # world_flags -> world_flag
        for k, v in (state.get("world_flags") or {}).items():
            self._ensure_unique(
                subject="world",
                predicate="world_flag",
                obj={"key": str(k), "value": v},
                scope="world",
                turn=turn,
                source="system",
                confidence=0.95,
            )

        # location
        loc = state.get("location")
        if _is_truthy(loc):
            self._ensure_unique(
                subject="scene",
                predicate="location",
                obj=str(loc),
                scope="scene",
                turn=turn,
                source="system",
            )

        # scene items
        for it in (state.get("available_items") or []):
            name = it.get("name")
            if not _is_truthy(name):
                continue
            self.add(Fact(
                id=uuid.uuid4().hex,
                subject="scene",
                predicate="has_item",
                object={"ref": f"item:{_slug(name)}", "name": name},
                scope="scene",
                confidence=0.7,
                turn=turn,
                source="system",
            ), replace_policy="append")

        # npcs
        for npc in (state.get("npcs") or []):
            npc_id = npc.get("id")
            if not _is_truthy(npc_id):
                continue
            name = npc.get("name") or npc_id
            mood = npc.get("mood")
            self._ensure_unique(
                subject=f"npc:{npc_id}",
                predicate="name",
                obj=str(name),
                scope="scene",
                turn=turn,
                source="system",
                confidence=0.99,
            )
            if _is_truthy(mood):
                self._ensure_unique(
                    subject=f"npc:{npc_id}",
                    predicate="mood",
                    obj=str(mood),
                    scope="scene",
                    turn=turn,
                    source="system",
                    confidence=0.8,
                )

        # players
        for p in (state.get("players") or []):
            pid = p.get("player_id")
            if not _is_truthy(pid):
                continue
            name = p.get("name") or pid
            self._ensure_unique(
                subject=f"player:{pid}",
                predicate="name",
                obj=str(name),
                scope="scene",
                turn=turn,
                source="system",
                confidence=0.99,
            )

    def apply_effects(self, effects: Dict[str, Any], turn: int, source: str = "effects") -> None:
        """
        Преобразуем effects в факты (корректируем location, world_flags,
        владение предметами, статусы/урон NPC/players).
        """
        # world_flags
        for k, v in (effects.get("world_flags") or {}).items():
            self._ensure_unique(
                subject="world",
                predicate="world_flag",
                obj={"key": str(k), "value": v},
                scope="world",
                turn=turn,
                source=source,
                confidence=0.95,
            )

        # location
        loc = effects.get("location")
        if _is_truthy(loc):
            self._ensure_unique(
                subject="scene",
                predicate="location",
                obj=str(loc),
                scope="scene",
                turn=turn,
                source=source,
                confidence=0.95,
            )

        # scene items add/remove
        for it in (effects.get("scene_items_add") or []):
            name = it.get("name") if isinstance(it, dict) else it
            if not _is_truthy(name):
                continue
            self.add(Fact(
                id=uuid.uuid4().hex,
                subject="scene",
                predicate="has_item",
                object={"ref": f"item:{_slug(str(name))}", "name": str(name)},
                scope="scene",
                confidence=0.8,
                turn=turn,
                source=source,
            ), replace_policy="append")

        for name in (effects.get("scene_items_remove") or []):
            if not _is_truthy(name):
                continue
            # Ретракт предмета сцены
            self.retract_where(subject="scene", predicate="has_item")

        # players patches
        for pat in (effects.get("players") or []):
            pid = pat.get("player_id")
            if not _is_truthy(pid):
                continue
            subj = f"player:{pid}"

            # hp_delta -> event
            if "hp_delta" in pat and pat.get("hp_delta"):
                self.add(Fact(
                    id=uuid.uuid4().hex,
                    subject=subj,
                    predicate="event",
                    object={"hp_delta": pat["hp_delta"]},
                    scope="scene",
                    confidence=0.7,
                    turn=turn,
                    source=source,
                ), replace_policy="append")

            # flags/status
            flags = pat.get("flags") or {}
            if "status" in flags:
                self._ensure_unique(
                    subject=subj, predicate="status", obj=str(flags["status"]),
                    scope="scene", turn=turn, source=source, confidence=0.9
                )

            # items_add/remove -> holding
            for it in (pat.get("items_add") or []):
                self._ensure_unique(
                    subject=subj, predicate="holding",
                    obj={"ref": f"item:{_slug(str(it))}", "name": str(it)},
                    scope="scene", turn=turn, source=source, confidence=0.85
                )
            for it in (pat.get("items_remove") or []):
                # если держал — ретракт holding
                self.retract_where(subject=subj, predicate="holding")

        # npcs patches
        for pat in (effects.get("npcs") or []):
            nid = pat.get("npc_id")
            if not _is_truthy(nid):
                continue
            subj = f"npc:{nid}"

            if "hp_delta" in pat and pat.get("hp_delta"):
                self.add(Fact(
                    id=uuid.uuid4().hex,
                    subject=subj,
                    predicate="event",
                    object={"hp_delta": pat["hp_delta"]},
                    scope="scene",
                    confidence=0.7,
                    turn=turn,
                    source=source,
                ), replace_policy="append")

            flags = pat.get("flags") or {}
            if "status" in flags:
                self._ensure_unique(
                    subject=subj, predicate="status", obj=str(flags["status"]),
                    scope="scene", turn=turn, source=source, confidence=0.9
                )

            for it in (pat.get("items_add") or []):
                self._ensure_unique(
                    subject=subj, predicate="holding",
                    obj={"ref": f"item:{_slug(str(it))}", "name": str(it)},
                    scope="scene", turn=turn, source=source, confidence=0.85
                )
            for it in (pat.get("items_remove") or []):
                self.retract_where(subject=subj, predicate="holding")

        # introductions -> создаём «event» или устойчивые факты
        intro = effects.get("introductions") or {}
        for item in (intro.get("items") or []):
            name = item.get("name") if isinstance(item, dict) else str(item)
            self.add(Fact(
                id=uuid.uuid4().hex,
                subject="scene",
                predicate="event",
                object={"introduced_item": name},
                scope="scene",
                confidence=0.6,
                turn=turn,
                source=source,
            ), replace_policy="append")

        for loc in (intro.get("locations") or []):
            name = loc.get("name") if isinstance(loc, dict) else str(loc)
            self.add(Fact(
                id=uuid.uuid4().hex,
                subject="world",
                predicate="event",
                object={"introduced_location": name},
                scope="world",
                confidence=0.6,
                turn=turn,
                source=source,
            ), replace_policy="append")

        for npc in (intro.get("npcs") or []):
            nid = npc.get("id") or npc.get("npc_id") or str(uuid.uuid4())
            self._ensure_unique(
                subject=f"npc:{nid}",
                predicate="name",
                obj=str(npc.get("name") or nid),
                scope="scene",
                turn=turn,
                source=source,
                confidence=0.9,
            )

    # --------- Рендер сводок для промпта ---------

    def render_prompt_bullets(self,
                              state: Dict[str, Any],
                              *,
                              audience: str = "general",
                              player_id: Optional[str] = None,
                              max_facts: int = 12) -> List[str]:
        """
        Готовые буллеты для промпта:
            - audience="general": публичные факты
            - audience="private": приватные факты (требует player_id)
        Лёгкий скоринг: приоритет уникальных и недавних фактов.
        """
        visible = []
        for f in self.all():
            if audience == "private":
                if f.private_to and f.private_to != player_id:
                    continue
            else:  # general
                if f.private_to:
                    continue
            visible.append(f)

        # фильтр: берём несуперседед, неистёкшие (state.turn если есть)
        now_turn = int(state.get("turn") or 0)
        visible = [f for f in visible if not f.is_expired(now_turn)]

        # скоринг: уникальные > события; свежие > старые; уверенность
        def score(f: Fact) -> float:
            base = 1.0
            if f.predicate in UNIQUE_PREDICATES:
                base += 1.0
            if f.predicate in APPEND_ONLY_PREDICATES:
                base += 0.2
            recency = max(0.0, 1.0 - 0.03 * max(0, now_turn - f.turn))
            return base * (0.5 + 0.5 * recency) * (0.5 + 0.5 * max(0.0, min(1.0, f.confidence)))

            # (в итоге ~0..2.5)
        visible.sort(key=score, reverse=True)

        bullets: List[str] = []
        for f in visible[:max_facts]:
            bullets.append(self._fact_to_bullet(f))
        return bullets

    def _fact_to_bullet(self, f: Fact) -> str:
        subj = f.subject
        # компактный человекочитаемый
        if isinstance(f.object, dict):
            obj = ", ".join(f"{k}={self._short(v)}" for k, v in f.object.items())
        else:
            obj = self._short(f.object)
        return f"• [{f.scope}] {subj} — {f.predicate}: {obj}"

    @staticmethod
    def _short(x: Any) -> str:
        s = str(x)
        return s if len(s) < 160 else s[:157] + "…"

    # --------- Вспомогательная фабрика ---------

    def _ensure_unique(self, *, subject: str, predicate: str, obj: Any, scope: str, turn: int,
                       source: str = "system", confidence: float = 0.9, private_to: Optional[str] = None,
                       ttl_turns: Optional[int] = None) -> Fact:
        f = Fact(
            id=uuid.uuid4().hex,
            subject=subject,
            predicate=predicate,
            object=_deepcopy(obj),
            scope=scope,
            confidence=float(confidence),
            turn=int(turn),
            source=source,
            private_to=private_to,
            ttl_turns=ttl_turns,
        )
        return self.add(f, replace_policy="unique" if predicate in UNIQUE_PREDICATES else "append")


# --------------------------
# Утилиты верхнего уровня
# --------------------------

def make_ref(kind: str, identifier: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Удобная фабрика ссылочного объекта: {"ref":"npc:npc_1","name":"Инженер Лин"}
    """
    ref = f"{kind}:{identifier}"
    d = {"ref": ref}
    if name:
        d["name"] = name
    return d


def add_world_flag(fb: FactBank, key: str, value: Any, turn: int, source: str = "effects") -> Fact:
    return fb._ensure_unique(
        subject="world",
        predicate="world_flag",
        obj={"key": str(key), "value": value},
        scope="world",
        turn=turn,
        source=source,
        confidence=0.95,
    )


# ---------------
# Примеры (doctest-подобно, не исполняются при импорте)
# ---------------
if __name__ == "__main__":
    # Мини-демо
    state = {
        "turn": 3,
        "location": "Лаборатория №13",
        "world_flags": {"time_of_day": "утро"},
        "players": [{"player_id": "p1", "name": "Дмитрий"}],
        "npcs": [{"id": "npc_1", "name": "Инженер Лин", "mood": "взволнованный"}],
        "available_items": [{"name": "кристаллический ключ", "source": "scene"}],
    }
    fb = FactBank.from_state({"fact_bank": []})
    fb.ingest_state_snapshot(state, turn=state["turn"])

    # Эффекты хода:
    effects = {
        "world_flags": {"time_of_day": "день"},
        "players": [{"player_id": "p1", "flags": {"status": "intoxicated"}}],
        "npcs": [{"npc_id": "npc_1", "hp_delta": -3, "flags": {"status": "stunned"}}],
        "scene_items_add": [{"name": "пустой шприц", "source": "scene"}],
    }
    fb.apply_effects(effects, turn=4)

    print("— GENERAL BULLETS —")
    for b in fb.render_prompt_bullets(state, audience="general", max_facts=10):
        print(b)

    print("\n— PRIVATE BULLETS (p1) —")
    for b in fb.render_prompt_bullets(state, audience="private", player_id="p1", max_facts=10):
        print(b)
