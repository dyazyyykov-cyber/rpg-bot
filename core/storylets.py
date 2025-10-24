# storylets.py
# -*- coding: utf-8 -*-
"""
Storylets — мини-сцены/правила, позволяющие:
- принять «любой бред» от игроков (мемы, бренды, неожиданные действия),
- сохранить логичность и причинно-следственную связь,
- выдавать детерминированные эффекты в формате, совместимом с текущим ядром.

Как использовать из движка:
    from storylets import propose_storylet_resolution, classify_action

    result = propose_storylet_resolution(state, {
        "player_id": pid,
        "player_name": name,
        "text": user_text,
        "lang": "ru",
    })
    if result:
        # result: {effects, raw:{text}, private:{text}, general:{text}, confidence, storylet_id}
        apply_effects(result["effects"])
        post_raw(result["raw"]["text"])
        post_private(pid, result["private"]["text"])
        post_general(result["general"]["text"])
    else:
        # Фоллбэк на LLM
        ...

Структура ожидаемого state минимальна и бережно читается через .get():
{
  "title": str,
  "story_theme": Optional[str],
  "location": str,
  "world_flags": dict,
  "npcs": [{"id":str,"name":str,"hp":int,"mood":str,"items":[...]}],
  "players": [{"player_id":str,"name":str,"items":[...]}],
  "available_items": [{"name":str,"source":"scene"|...}],
}

Никаких внешних импортов, всё чистый Python 3.10+.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import re


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _safe_get(lst: Optional[List[dict]], idx: int, default=None):
    try:
        return (lst or [])[idx]
    except Exception:
        return default


def _find_entity_by_name(text_lc: str, entities: Sequence[Dict[str, Any]], name_key: str = "name") -> Optional[Dict[str, Any]]:
    """
    Ищем в тексте упоминание сущности (NPC/игрока/предмета) по подстроке имени.
    Возвращаем первый матч.
    """
    for e in entities or []:
        name = _lc(e.get(name_key))
        if name and name in text_lc:
            return e
    return None


def _choose_default_target(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Фоллбэк-цель: первый NPC в сцене (или None).
    """
    npcs = state.get("npcs") or []
    return _safe_get(npcs, 0)


def _fmt(template: str, **kw) -> str:
    """
    Щадящий форматер: не упадёт, если какой-то ключ не передан.
    """
    class _Dict(dict):
        def __missing__(self, k):  # type: ignore
            return "{" + k + "}"
    return template.format_map(_Dict(**kw))


# ======================================
# КЛАССИФИКАЦИЯ ДЕЙСТВИЙ И ИНТЕНТОВ
# ======================================

_INTENT_PATTERNS_RU: List[Tuple[str, List[str]]] = [
    ("violence", [r"\b(бью|удар|ху(к|ком)|пинаю|режу|стреляю|ломаю|вырубаю)\b"]),
    ("dose",     [r"\b(вкалываю|шприц|инъекц|ввожу|колю|наркот|героин|доза)\b"]),
    ("talk",     [r"\b(говорю|спрашиваю|кричу|обращаюсь|спросить|сказать)\b"]),
    ("examine",  [r"\b(смотрю|осматриваю|изучаю|проверяю|что такое|кто такой)\b"]),
    ("take",     [r"\b(беру|забираю|хватаю|поднимаю|подобираю)\b"]),
    ("use",      [r"\b(использую|применяю|активирую|задействую)\b"]),
    ("move",     [r"\b(иду|двигаюсь|перехожу|перемещаюсь|бегу|выхожу|вхожу)\b"]),
    ("build",    [r"\b(строю|создаю|собираю|конструирую|крафчу)\b"]),
    ("cast",     [r"\b(колдую|заклинан|маг(ия|ический)|призываю)\b"]),
    ("calm",     [r"\b(успокаиваю|усмиряю|уговариваю|умиротворяю)\b"]),
]

_INTENT_PATTERNS_EN: List[Tuple[str, List[str]]] = [
    ("violence", [r"\b(hit|punch|kick|stab|shoot|smash|strike)\b"]),
    ("dose",     [r"\b(inject|syringe|dose|heroin|drug)\b"]),
    ("talk",     [r"\b(say|ask|yell|speak|tell|talk)\b"]),
    ("examine",  [r"\b(look|examine|inspect|check|what is|who is)\b"]),
    ("take",     [r"\b(take|grab|pick up|snatch)\b"]),
    ("use",      [r"\b(use|apply|activate|engage)\b"]),
    ("move",     [r"\b(go|walk|run|move|enter|leave)\b"]),
    ("build",    [r"\b(build|craft|assemble|construct)\b"]),
    ("cast",     [r"\b(cast|spell|magic|summon)\b"]),
    ("calm",     [r"\b(calm|pacify|soothe)\b"]),
]

_BRAND_OR_MEME_CLUES = [
    "марио", "супермарио", "super mario", "пикачу", "гандам", "ивангай", "четкий паца", "кринж",
    "mario", "zelda", "sonic", "doom", "tetris", "warhammer", "skr", "sigma", "gyat"
]


def _detect_lang(text: str) -> str:
    # Очень простая эвристика: наличие кириллицы => ru.
    return "ru" if re.search(r"[а-яА-Я]", text) else "en"


def classify_action(text: str, lang: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Возвращает (primary_intent, tags)
    tags — набор ключевых подсказок: "brand", "question", "target:npc_x" и т.д. (минимум для стоимлета).
    """
    t = _lc(text)
    lang = lang or _detect_lang(t)

    patterns = _INTENT_PATTERNS_RU if lang == "ru" else _INTENT_PATTERNS_EN

    scores: Dict[str, int] = {}
    for intent, regs in patterns:
        for r in regs:
            if re.search(r, t):
                scores[intent] = scores.get(intent, 0) + 1

    # Простейший приоритет
    order = ["dose", "violence", "calm", "use", "take", "talk", "examine", "move", "build", "cast"]
    best = max(scores.items(), key=lambda kv: (kv[1], -order.index(kv[0]))) if scores else ("talk" if "?" in t else "examine" if "что" in t or "what" in t else "talk")

    tags: List[str] = []
    # Бренды/мемы:
    if any(k in t for k in _BRAND_OR_MEME_CLUES):
        tags.append("brand")
    if "?" in t or "что такое" in t or "what is" in t:
        tags.append("question")

    return best[0], tags


# ==========================
# СТОРИЛЕТЫ (ДАТАКЛАССЫ)
# ==========================

@dataclass
class Trigger:
    intents: Sequence[str] = field(default_factory=list)
    keywords_any: Sequence[str] = field(default_factory=list)
    must_have_tag: Optional[str] = None
    score: int = 1  # вклад в итоговый скор при совпадении


@dataclass
class Storylet:
    storylet_id: str
    name: str
    description: str
    strategy: str  # 'embrace' | 'yes_and' | 'redirect' | 'oppose'
    triggers: Sequence[Trigger]
    priority: int = 0  # чем выше, тем предпочтительнее при равном score
    guard: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None
    effect_builder: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] = lambda s, a: {}
    narrative_builder: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Dict[str, str]]] = lambda s, a, e: {"raw": {"text": ""}, "private": {"text": ""}, "general": {"text": ""}}


# ==========================
# РЕЕСТР СТОРИЛЕТОВ
# ==========================

class StoryletRegistry:
    def __init__(self):
        self._items: List[Storylet] = []

    def register(self, storylet: Storylet):
        self._items.append(storylet)

    def match(self, state: Dict[str, Any], action: Dict[str, Any]) -> List[Tuple[Storylet, int]]:
        """
        Подсчёт простого score по триггерам + учёт guard.
        """
        text_lc = _lc(action.get("text"))
        intent, tags = classify_action(text_lc, action.get("lang"))

        candidates: List[Tuple[Storylet, int]] = []
        for st in self._items:
            if st.guard and not st.guard(state, action):
                continue
            local = 0
            for tr in st.triggers:
                # intent
                if tr.intents and intent not in tr.intents:
                    continue
                # tag
                if tr.must_have_tag and tr.must_have_tag not in tags:
                    continue
                # keywords
                if tr.keywords_any and not any(k in text_lc for k in tr.keywords_any):
                    continue
                local += tr.score
            if local > 0:
                candidates.append((st, local))

        # сортировка по score, затем по приоритету
        candidates.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        return candidates


# ==================================
# ЭФФЕКТЫ И НАРРАТИВ (БИЛДЕРЫ)
# ==================================

def _ensure_effect_skeleton() -> Dict[str, Any]:
    return {
        "world_flags": {},
        "location": None,
        "scene_items_add": [],
        "scene_items_remove": [],
        "players": [],
        "npcs": [],
        "introductions": {"npcs": [], "items": [], "locations": []},
    }


def _player_patch(player_id: str,
                  hp_delta: int = 0,
                  items_add: Optional[List[str]] = None,
                  items_remove: Optional[List[str]] = None,
                  flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "player_id": player_id,
        "hp_delta": hp_delta,
        "items_add": items_add or [],
        "items_remove": items_remove or [],
        "flags": flags or {}
    }


def _npc_patch(npc_id: str,
               hp_delta: int = 0,
               items_add: Optional[List[str]] = None,
               items_remove: Optional[List[str]] = None,
               flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "npc_id": npc_id,
        "hp_delta": hp_delta,
        "items_add": items_add or [],
        "items_remove": items_remove or [],
        "flags": flags or {}
    }


# ---------- Storylet 1: Инъекция/дозинг (embrace/yes_and)

def _guard_dose(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    text_lc = _lc(action.get("text"))
    # минимальная защита: в тексте явно шприц/инъекция/героин/ввожу/колю
    return any(k in text_lc for k in ["шприц", "инъек", "ввожу", "колю", "героин", "inject", "syringe", "dose", "drug"])


def _effect_dose(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    text_lc = _lc(action.get("text"))
    player_id = action.get("player_id") or action.get("player") or ""
    player_name = action.get("player_name") or "Игрок"

    npcs = state.get("npcs") or []
    target = _find_entity_by_name(text_lc, npcs) or _choose_default_target(state)

    eff = _ensure_effect_skeleton()

    # Добавим статус "интоксикация" цели; если цели нет — игроку.
    if target and target.get("id"):
        eff["npcs"].append(_npc_patch(target["id"], flags={"status": "intoxicated"}))
        target_name = target.get("name") or "цель"
    else:
        eff["players"].append(_player_patch(player_id, flags={"status": "intoxicated"}))
        target_name = player_name

    # Небольшой «след в мире»
    eff["world_flags"]["last_procedure"] = "injection"
    eff["scene_items_add"].append({"name": "пустой шприц", "source": "scene"})

    # Если в тексте фигурирует бренд/мем, введём это как «артефакт-референс»
    if any(k in text_lc for k in _BRAND_OR_MEME_CLUES):
        eff["introductions"]["items"].append({"name": "медиа-артефакт", "meta": "brand-reference"})

    # Вернём вместе с полезным контекстом
    eff["_storylet_context"] = {"target_name": target_name}
    return eff


def _narrative_dose(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    theme = state.get("story_theme") or state.get("title") or "история"
    player_name = action.get("player_name") or "Игрок"
    target_name = (effects.get("_storylet_context") or {}).get("target_name", "цель")

    raw = _fmt("{player} делает инъекцию. {target} реагирует непредсказуемо.",
               player=player_name, target=target_name)

    private = _fmt(
        "Ты аккуратно вводишь раствор. {target} на мгновение замирает — пульс учащается, "
        "дыхание меняется. В твоей голове выстраивается план: использовать эффект, чтобы продвинуться в теме «{theme}».",
        target=target_name, theme=theme
    )

    general = _fmt(
        "{player} вводит препарат {target}. На секунду пространство будто «провисает»: детали сцены становятся резче, "
        "и {target} проявляет необычную реакцию — не хаос ради хаоса, а закономерность, которую ещё предстоит понять.",
        player=player_name, target=target_name
    )

    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}


# ---------- Storylet 2: Удар/усмирение (yes_and/redirect)

def _guard_calm_or_hit(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    text_lc = _lc(action.get("text"))
    return any(k in text_lc for k in ["бью", "удар", "хуком", "пинаю", "усмиряю", "успокаиваю", "hit", "punch", "calm"])


def _effect_calm_or_hit(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    text_lc = _lc(action.get("text"))
    player_id = action.get("player_id") or ""
    npcs = state.get("npcs") or []
    target = _find_entity_by_name(text_lc, npcs) or _choose_default_target(state)

    eff = _ensure_effect_skeleton()

    if target and target.get("id"):
        # Немного урона + оглушение
        eff["npcs"].append(_npc_patch(target["id"], hp_delta=-3, flags={"status": "stunned"}))
        target_name = target.get("name") or "цель"
    else:
        # Никого не нашли — считаем, что игрок травмировал кисть о твёрдую поверхность
        eff["players"].append(_player_patch(player_id, hp_delta=-1, flags={"note": "боль в руке"}))
        target_name = "твёрдая поверхность"

    eff["world_flags"]["last_conflict"] = "nonlethal_hit"
    eff["_storylet_context"] = {"target_name": target_name}
    return eff


def _narrative_calm_or_hit(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    player_name = action.get("player_name") or "Игрок"
    target_name = (effects.get("_storylet_context") or {}).get("target_name", "цель")
    place = state.get("location") or "здесь"

    raw = _fmt("{player} резко бьёт по {target}, пытаясь перехватить контроль над ситуацией.", player=player_name, target=target_name)

    private = _fmt(
        "Удар даётся тяжело, но ты замечаешь — {target} дезориентирован(а). Окно возможностей открыто на короткое время.",
        target=target_name
    )

    general = _fmt(
        "В {place} раздаётся сухой удар — {player} перехватывает инициативу. {target} дезориентирован(а), хаос "
        "замедляется и складывается в рисунок: теперь можно действовать осмысленно.",
        place=place, player=player_name, target=target_name
    )

    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}


# ---------- Storylet 3: Вопрос/«что такое X?» (redirect/embrace)
# Объясняет непонятный мем/бренд через внутри-сеттинговое «переосмысление».

def _guard_explain(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    t = _lc(action.get("text"))
    return ("что такое" in t) or ("what is" in t) or ("кто такой" in t) or ("who is" in t) or ("супермарио" in t) or ("super mario" in t)


def _extract_unknown_term(text_lc: str) -> str:
    # Попробуем выцепить слово/фразу в кавычках или после "что такое"
    m = re.search(r"что такое\s+([«\"“”]?)([^\"”»]+)\1", text_lc)
    if m:
        return m.group(2).strip()
    m = re.search(r"what is\s+([\"“”]?)([^\"”]+)\1", text_lc)
    if m:
        return m.group(2).strip()
    # Вариант: известные мемы
    for k in _BRAND_OR_MEME_CLUES:
        if k in text_lc:
            return k
    # fallback:
    return "неизвестный термин"


def _effect_explain(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    t = _lc(action.get("text"))
    term = _extract_unknown_term(t)
    eff = _ensure_effect_skeleton()
    eff["world_flags"]["glossary/last_term"] = term
    # Вводим «артефакт-справку», чтобы «бренд/мем» жил в мире как объект культуры.
    eff["introductions"]["items"].append({"name": f"справка о «{term}»", "meta": "diegetic-glossary"})
    eff["_storylet_context"] = {"term": term}
    return eff


def _narrative_explain(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    term = (effects.get("_storylet_context") or {}).get("term", "термин")
    theme = state.get("story_theme") or state.get("title") or "сюжет"
    place = state.get("location") or "сцене"
    player_name = action.get("player_name") or "Игрок"

    raw = _fmt("{player} поднимает вопрос: «Что такое {term}?».", player=player_name, term=term)

    private = _fmt(
        "Ты пытаешься связать «{term}» с текущей темой «{theme}». Сбор фактов даёт рабочую гипотезу: "
        "это культурный маркер/модель, которую можно встроить в контекст, не ломая правдоподобие.",
        term=term, theme=theme
    )

    general = _fmt(
        "В {place} проясняется понятие «{term}»: внутри мира это воспринимается как известная модель/артефакт. "
        "Не «магия из ниоткуда», а ссылка на культурный слой, уже присутствующий в сеттинге.",
        place=place, term=term
    )

    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}


# ---------- Storylet 4: «Да-и» для мемов/брендов (yes_and)
# Универсальная подхватка: если есть tag=brand, мягко вводим это в мир.

def _guard_brand_yes_and(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    t = _lc(action.get("text"))
    return any(k in t for k in _BRAND_OR_MEME_CLUES)


def _effect_brand_yes_and(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    t = _lc(action.get("text"))
    # Попробуем выделить «брендовое» слово (самое длинное из известных в тексте)
    terms = [k for k in _BRAND_OR_MEME_CLUES if k in t]
    key = sorted(terms, key=len, reverse=True)[0] if terms else "мем"
    eff = _ensure_effect_skeleton()
    eff["introductions"]["items"].append({"name": f"культурный артефакт «{key}»", "meta": "brand"})
    eff["_storylet_context"] = {"key": key}
    return eff


def _narrative_brand_yes_and(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    key = (effects.get("_storylet_context") or {}).get("key", "мем")
    player_name = action.get("player_name") or "Игрок"

    raw = _fmt("{player} вводит в разговор культурную отсылку: {key}.", player=player_name, key=key)
    private = _fmt("Ты чувствуешь, как пространство истории подстраивается под отсылку «{key}», не ломая логики.", key=key)
    general = _fmt("Отсылка «{key}» закрепляется в мире как культурный слой, к которому можно обращаться.", key=key)

    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}


# ==========================
# ПОСТРОЕНИЕ РЕЕСТРА
# ==========================

def build_default_registry() -> StoryletRegistry:
    reg = StoryletRegistry()

    reg.register(Storylet(
        storylet_id="dose.inject",
        name="Инъекция/введение препарата",
        description="Игрок вводит что-то в цель — принимаем и привязываем к причинности.",
        strategy="embrace",
        triggers=[
            Trigger(intents=["dose"], keywords_any=["шприц", "инъек", "ввожу", "колю", "героин", "syringe", "inject"], score=2),
        ],
        priority=50,
        guard=_guard_dose,
        effect_builder=_effect_dose,
        narrative_builder=_narrative_dose
    ))

    reg.register(Storylet(
        storylet_id="combat.calm_or_hit",
        name="Нелетальный удар/усмирение",
        description="Игрок пытается силой усмирить цель — переводим хаос в контролируемое состояние.",
        strategy="yes_and",
        triggers=[
            Trigger(intents=["violence", "calm"], keywords_any=["удар", "бью", "хуком", "усмиряю", "успокаиваю", "hit", "punch", "calm"], score=2),
        ],
        priority=40,
        guard=_guard_calm_or_hit,
        effect_builder=_effect_calm_or_hit,
        narrative_builder=_narrative_calm_or_hit
    ))

    reg.register(Storylet(
        storylet_id="explain.term",
        name="Прояснить термин/отсылку",
        description="Игрок спрашивает «что такое X» — делаем диэгетическое объяснение.",
        strategy="redirect",
        triggers=[
            Trigger(intents=["examine", "talk"], keywords_any=["что такое", "what is", "кто такой", "who is", "супермарио", "super mario"], score=2),
        ],
        priority=35,
        guard=_guard_explain,
        effect_builder=_effect_explain,
        narrative_builder=_narrative_explain
    ))

    reg.register(Storylet(
        storylet_id="brand.yes_and",
        name="Да-и: бренд/мем",
        description="Любая брендо- или мемо-отсылка становится культурным артефактом мира.",
        strategy="yes_and",
        triggers=[
            Trigger(intents=[], keywords_any=[], must_have_tag="brand", score=1),
        ],
        priority=20,
        guard=_guard_brand_yes_and,
        effect_builder=_effect_brand_yes_and,
        narrative_builder=_narrative_brand_yes_and
    ))

    return reg


# ==========================
# ПУБЛИЧНЫЙ API МОДУЛЯ
# ==========================

_DEFAULT_REGISTRY = build_default_registry()

def propose_storylet_resolution(state: Dict[str, Any], action: Dict[str, Any], *, min_score: int = 1) -> Optional[Dict[str, Any]]:
    """
    Пытается применить сторилет к действию игрока.
    Возвращает:
        {
          "effects": {...},
          "raw": {"text": "..."},
          "private": {"text": "..."},
          "general": {"text": "..."},
          "confidence": float,
          "storylet_id": str
        }
    или None, если сторилет не найден.
    """
    # Обогащаем action языком/интентом для триггера
    text_lc = _lc(action.get("text"))
    intent, tags = classify_action(text_lc, action.get("lang"))
    action = dict(action)
    action.setdefault("lang", _detect_lang(text_lc))
    action["intent"] = intent
    action["tags"] = tags

    matches = _DEFAULT_REGISTRY.match(state, action)
    if not matches:
        return None

    storylet, score = matches[0]
    if score < min_score:
        return None

    # Синтез эффектов и нарратива
    effects = storylet.effect_builder(state, action) or _ensure_effect_skeleton()
    narrative = storylet.narrative_builder(state, action, effects) or {"raw": {"text": ""}, "private": {"text": ""}, "general": {"text": ""}}

    # Уберём служебный контекст из эффектов
    service = effects.pop("_storylet_context", None)

    return {
        "effects": effects,
        "raw": {"text": str((narrative.get("raw") or {}).get("text", "")).strip()},
        "private": {"text": str((narrative.get("private") or {}).get("text", "")).strip()},
        "general": {"text": str((narrative.get("general") or {}).get("text", "")).strip()},
        "confidence": min(1.0, 0.5 + 0.1 * score + (0.05 if service else 0.0) + (0.05 if "brand" in tags else 0.0)),
        "storylet_id": storylet.storylet_id,
        "strategy": storylet.strategy,
    }


__all__ = [
    "classify_action",
    "propose_storylet_resolution",
    "Storylet",
    "StoryletRegistry",
    "build_default_registry",
]
