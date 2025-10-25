from __future__ import annotations

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
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    tags — набор ключевых подсказок: "brand", "question", "target:npc_x" и т.д. (минимум для сторилета).
    """
    t = _lc(text)
    lang = lang or _detect_lang(t)
    intent_patterns = _INTENT_PATTERNS_RU if lang == "ru" else _INTENT_PATTERNS_EN
    intent = "other"
    for name, patterns in intent_patterns:
        for pat in patterns:
            if re.search(pat, t):
                intent = name
                break
        if intent != "other":
            break

    tags: List[str] = []
    # Указание на конкретную цель: target:<npc_id>
    if "target:" in t:
        tags.append("targeted")

    # Особые типы запросов:
    if any(k in t for k in _BRAND_OR_MEME_CLUES):
        tags.append("brand")
    if "?" in t or "что такое" in t or "what is" in t:
        tags.append("question")

    # Если цель не найдена и действие явно насильственное — предполагаем цель по умолчанию
    # (в guard-функциях используется _choose_default_target)
    return intent, tags

# ======================================
# ФУНКЦИИ СТОРИЛЕТОВ (эффекты и нарратив)
# ======================================

# Storylet 1: Инъекция/дозинг (embrace/yes_and)
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
        eff["npcs"].append({"npc_id": target["id"], "hp_delta": 0, "items_add": [], "items_remove": [], "flags": {"status": "intoxicated"}})
        target_name = target.get("name") or "цель"
    else:
        eff["players"].append({"player_id": player_id, "hp_delta": 0, "items_add": [], "items_remove": [], "flags": {"status": "intoxicated"}})
        target_name = player_name

    # Небольшой «след в мире»
    eff["world_flags"]["last_procedure"] = "injection"
    eff["scene_items_add"].append({"name": "пустой шприц", "source": "scene"})

    # Если в тексте фигурирует бренд/мем, введём это как «артефакт-референс»
    if any(k in text_lc for k in _BRAND_OR_MEME_CLUES):
        eff["introductions"]["items"].append({"name": "медиа-артефакт", "meta": "brand-reference"})

    eff["_storylet_context"] = {"target_name": target_name}
    return eff

def _narrative_dose(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    theme = state.get("story_theme") or state.get("title") or "история"
    player_name = action.get("player_name") or "Игрок"
    target_name = (effects.get("_storylet_context") or {}).get("target_name", "цель")

    raw = _fmt("{player} делает инъекцию. {target} реагирует непредсказуемо.", player=player_name, target=target_name)
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

# Storylet 2: Удар/усмирение (yes_and/redirect)
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
        eff["npcs"].append({"npc_id": target["id"], "hp_delta": -3, "items_add": [], "items_remove": [], "flags": {"status": "stunned"}})
        target_name = target.get("name") or "цель"
    else:
        # Никого не нашли — считаем, что игрок травмировал кисть о твёрдую поверхность
        eff["players"].append({"player_id": player_id, "hp_delta": -1, "items_add": [], "items_remove": [], "flags": {"note": "боль в руке"}})
        target_name = "твёрдая поверхность"

    eff["world_flags"]["last_conflict"] = "nonlethal_hit"
    eff["_storylet_context"] = {"target_name": target_name}
    return eff

def _narrative_calm_or_hit(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    player_name = action.get("player_name") or "Игрок"
    target_name = (effects.get("_storylet_context") or {}).get("target_name", "цель")
    place = state.get("location") or "здесь"

    raw = _fmt("{player} резко бьёт по {target}, пытаясь перехватить контроль над ситуацией.", player=player_name, target=target_name)
    private = _fmt("Ты наносишь удар по {target}. {target} пошатнулся, но ситуация под контролем.", target=target_name)
    general = _fmt("{player} наносит удар по {target}, стараясь успокоить ситуацию. {target} приходит в себя, хотя выглядит потрёпанным.", player=player_name, target=target_name)
    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}

# Storylet 3: Объяснение термина/отсылки (redirect/storylet)
def _guard_explain(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    t = _lc(action.get("text"))
    return "что такое" in t or "what is" in t or "кто такой" in t or "who is" in t

def _effect_explain(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    text_lc = _lc(action.get("text") or "")
    # Выделим термин после "что такое"
    term_match = re.search(r"что такое\s+(.+)", text_lc) or re.search(r"what is\s+(.+)", text_lc)
    term = term_match.group(1) if term_match else (action.get("text") or "").strip()
    eff = _ensure_effect_skeleton()
    eff["introductions"]["items"].append({"name": f"справка о «{term}»", "meta": "diegetic-glossary"})
    eff["_storylet_context"] = {"term": term}
    return eff

def _narrative_explain(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    term = (effects.get("_storylet_context") or {}).get("term", "")
    raw = _fmt("На столе находится папка с надписью \"{term}\".", term=term)
    private = _fmt("Ты находишь записи, проливающие свет на термин «{term}». Возможно, стоит изучить эту папку подробнее.", term=term)
    general = _fmt("В лаборатории обнаруживается документ с заголовком «{term}», содержащий важные сведения.", term=term)
    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}

# Storylet 4: Бренд/мем как артефакт (yes_and)
def _guard_brand_yes_and(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    t = _lc(action.get("text"))
    return any(k in t for k in _BRAND_OR_MEME_CLUES)

def _effect_brand_yes_and(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    t = _lc(action.get("text"))
    eff = _ensure_effect_skeleton()
    terms = [k for k in _BRAND_OR_MEME_CLUES if k in t]
    key = sorted(terms, key=len, reverse=True)[0] if terms else "мем"
    eff["introductions"]["items"].append({"name": f"культурный артефакт «{key}»", "meta": "brand"})
    eff["_storylet_context"] = {"key": key}
    return eff

def _narrative_brand_yes_and(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    key = (effects.get("_storylet_context") or {}).get("key", "мем")
    player_name = action.get("player_name") or "Игрок"
    raw = _fmt("{player} наталкивается на странный объект, явно связанный с культурным феноменом \"{key}\".", player=player_name, key=key)
    private = _fmt("Ты обнаружил артефакт, напоминающий о явлении \"{key}\". Похоже, в этом мире он стал реальным предметом.", key=key)
    general = _fmt("Внимание всех привлекает предмет, явно являющийся воплощением мема \"{key}\" в этом мире.", key=key)
    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}

# Storylet 5: Камео персонажа "Слава КПСС" (analog)
def _guard_cameo_slava(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    t = _lc(action.get("text"))
    return "слава кпсс" in t or "slava kpss" in t

def _effect_cameo_slava(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    # Представляем упомянутую личность как персонажа мира
    eff = _ensure_effect_skeleton()
    # Создаем нового NPC с уникальным id
    npc_id = str(uuid.uuid4())[:8]
    eff["introductions"]["npcs"].append({"id": npc_id, "name": "Славик КПСС", "mood": "neutral", "hp": 100, "items": []})
    return eff

def _narrative_cameo_slava(state: Dict[str, Any], action: Dict[str, Any], effects: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    player_name = action.get("player_name") or "Игрок"
    npc_name = "Славик КПСС"
    raw = _fmt("{player} упоминает имя 'Слава КПСС'. Неожиданно выясняется, что в лаборатории есть сотрудник по прозвищу {npc}.", player=player_name, npc=npc_name)
    private = _fmt("Ты припоминаешь слухи о специалисте с прозвищем {npc}, и обнаруживаешь, что он действительно присутствует здесь.", npc=npc_name)
    general = _fmt("Дверь скрипит, и в комнату входит новый человек — {npc}, известный по прозвищу 'Слава КПСС'.", npc=npc_name)
    return {"raw": {"text": raw}, "private": {"text": private}, "general": {"text": general}}

# =========================
# ВНУТРЕННИЕ УТИЛИТЫ И РЕГИСТР
# =========================

@dataclass
class Trigger:
    intents: List[str] = field(default_factory=list)
    keywords_any: List[str] = field(default_factory=list)
    must_have_tag: Optional[str] = None
    score: int = 0

@dataclass
class Storylet:
    storylet_id: str
    name: str
    description: str
    strategy: str
    triggers: List[Trigger]
    priority: int
    guard: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    effect_builder: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    narrative_builder: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Dict[str, str]]]

class StoryletRegistry:
    def __init__(self):
        self._storylets: List[Storylet] = []

    def register(self, storylet: Storylet) -> None:
        self._storylets.append(storylet)
        # Sort by priority descending for matching
        self._storylets.sort(key=lambda s: s.priority, reverse=True)

    def match(self, state: Dict[str, Any], action: Dict[str, Any]) -> List[Tuple[Storylet, int]]:
        """
        Возвращает список (storylet, score) подходящих сторилетов, отсортированных по убыванию очков.
        """
        candidates: List[Tuple[Storylet, int]] = []
        for s in self._storylets:
            # Guard check
            try:
                if not s.guard(state, action):
                    continue
            except Exception:
                continue
            # Compute trigger score
            sc = 0
            for trig in s.triggers:
                if trig.intents and action.get("intent") not in trig.intents:
                    continue
                if trig.must_have_tag and trig.must_have_tag not in (action.get("tags") or []):
                    continue
                if trig.keywords_any:
                    text_low = _lc(action.get("text"))
                    if not any(kw in text_low for kw in trig.keywords_any):
                        continue
                sc = max(sc, trig.score)
            if sc > 0:
                candidates.append((s, sc))
        # Sort by score and priority (already sorted by priority)
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates

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

    reg.register(Storylet(
        storylet_id="cameo.slava",
        name="Камео: Слава КПСС",
        description="При упоминании 'Слава КПСС' вводится одноименный персонаж в мир.",
        strategy="analog",
        triggers=[
            Trigger(intents=[], keywords_any=["слава кпсс", "slava kpss"], score=2),
        ],
        priority=10,
        guard=_guard_cameo_slava,
        effect_builder=_effect_cameo_slava,
        narrative_builder=_narrative_cameo_slava
    ))

    return reg

# --------------------------
# ПУБЛИЧНЫЙ API МОДУЛЯ
# --------------------------

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
