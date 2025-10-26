from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

# =============================================================
# ЕДИНЫЕ СХЕМЫ ДАННЫХ ДЛЯ ПАЙПЛАЙНА (PLAN → EFFECTS → FACTS → OUTLINES → PROSE)
# -------------------------------------------------------------
# Принципы:
#  - Строгие модели (extra="forbid"), предсказуемые границы.
#  - Совместимость со старым кодом: прежние классы и поля сохранены.
#  - Новые сущности: StoryTheme, VerifiedFact, Outline.
# =============================================================

# ------------------------- БАЗОВЫЕ ТИПЫ -------------------------

class ItemEntry(BaseModel):
    """Предмет, видимый в сцене. В state.available_items допускаем только source='scene'."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=80)
    source: str = Field("scene", min_length=1, max_length=32, description="scene | npc_id | player_id")

class SoftStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    харизма: int = Field(0, ge=0, le=5)
    внимательность: int = Field(0, ge=0, le=5)
    удача: int = Field(0, ge=0, le=5)
    авторитет: int = Field(0, ge=0, le=5)

class RoleEntry(BaseModel):
    """Каталог ролей (исторически). Оставлен для совместимости."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    role: str = Field(..., validation_alias=AliasChoices("role", "name"), min_length=1, max_length=80)
    summary: str = Field(..., min_length=3, max_length=280)
    base_hp: int = Field(120, ge=60, le=300)
    soft_stats: Optional[SoftStats] = None

class NpcEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=80)
    mood: str = Field("neutral", min_length=1, max_length=24)
    hp: int = Field(100, ge=0, le=1000)
    items: List[str] = Field(default_factory=list, min_items=0, max_items=12)

class PlayerState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field("", max_length=80)
    role: Optional[str] = Field(default=None, max_length=80)
    hp: int = Field(100, ge=0, le=1000)
    max_hp: int = Field(100, ge=1, le=1000)
    status: Dict[str, Any] = Field(default_factory=dict)
    items: List[str] = Field(default_factory=list, min_items=0, max_items=20)
    position: str = Field("scene", min_length=1, max_length=32)

# ------------------------- СТАРТОВЫЙ МИР ------------------------

class VisibilityDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perception_dc: int = Field(12, ge=1, le=30)
    public_on_panic: bool = True

class StyleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tone: str = Field("mystery", min_length=1, max_length=40)
    pacing: str = Field("medium", min_length=1, max_length=40, validation_alias=AliasChoices("pacing", "pace"),
                        serialization_alias="pacing")

class StoryTheme(BaseModel):
    """Опциональная тема/жанр партии из лобби."""
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, max_length=120)
    genre: Optional[str] = Field(default=None, max_length=48)
    tone: Optional[str] = Field(default=None, max_length=32)

class InitialWorld(BaseModel):
    """Старый контракт старта мира (оставлен для обратной совместимости)."""
    model_config = ConfigDict(extra="forbid")

    setting: str = Field(..., min_length=20, max_length=2000)
    location: str = Field(..., min_length=5, max_length=400)
    world_flags: Dict[str, str] = Field(default_factory=dict)

    npcs: List[NpcEntry] = Field(default_factory=list, min_items=0, max_items=16)
    available_items: List[ItemEntry] = Field(default_factory=list, min_items=0, max_items=20)

    roles_catalog: List[RoleEntry] = Field(default_factory=list, min_items=0, max_items=12)
    visibility_defaults: Optional[VisibilityDefaults] = None
    style: Optional[StyleSpec] = None
    opening_hook: str = Field(..., min_length=10, max_length=500)

class InitialWorldV2(BaseModel):
    """Новый контракт мира без roles_catalog; роли назначаются отдельно."""
    model_config = ConfigDict(extra="forbid")

    setting: str = Field(..., min_length=20, max_length=2000)
    location: str = Field(..., min_length=5, max_length=400)
    world_flags: Dict[str, str] = Field(default_factory=dict)

    npcs: List[NpcEntry] = Field(default_factory=list, min_items=0, max_items=16)
    available_items: List[ItemEntry] = Field(default_factory=list, min_items=0, max_items=20)

    visibility_defaults: Optional[VisibilityDefaults] = None
    style: Optional[StyleSpec] = None
    opening_hook: str = Field(..., min_length=10, max_length=500)

class PlayerRoleAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: str = Field(..., min_length=1, max_length=64)
    role: str = Field(..., min_length=1, max_length=80)
    summary: str = Field(..., min_length=3, max_length=280)

class WorldAndRoles(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world: InitialWorldV2
    roles_for_players: List[PlayerRoleAssignment] = Field(..., min_items=1, max_items=64)

# --------------------------- STATE (runtime) ---------------------------

class PrivateHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: str = Field(..., min_length=1, max_length=64)
    text: str = Field(..., min_length=1, max_length=2000)
    echo_of_action: str = Field(..., min_length=1, max_length=400)
    highlights: Optional[List[str]] = Field(default=None, min_items=0, max_items=8)

class GeneralHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=4000)

# --------------------------- EFFECTS (delta) ---------------------------

class PlayerEffect(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    player_id: str = Field(validation_alias=AliasChoices("player_id", "id"), min_length=1, max_length=64)
    hp_delta: int = Field(0, ge=-300, le=300)
    hp: Optional[int] = Field(default=None, ge=0, le=1000)  # абсолютное, если задано
    status_apply: Dict[str, Any] = Field(default_factory=dict)
    items_add: List[str] = Field(default_factory=list, min_items=0, max_items=12)
    items_remove: List[str] = Field(default_factory=list, min_items=0, max_items=12)
    position: Optional[str] = Field(default=None, max_length=32)

class NpcEffect(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    npc_id: str = Field(validation_alias=AliasChoices("npc_id", "id"), min_length=1, max_length=64)
    hp_delta: int = Field(0, ge=-300, le=300)
    hp: Optional[int] = Field(default=None, ge=0, le=1000)
    mood: Optional[str] = Field(default=None, max_length=24)
    items_add: List[str] = Field(default_factory=list, min_items=0, max_items=12)
    items_remove: List[str] = Field(default_factory=list, min_items=0, max_items=12)

class Introductions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    npcs: List[NpcEntry] = Field(default_factory=list, min_items=0, max_items=10)
    items: List[ItemEntry] = Field(default_factory=list, min_items=0, max_items=12)
    locations: List[str] = Field(default_factory=list, min_items=0, max_items=10)

class EffectsDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Мир / сцена
    world_flags: Dict[str, str] = Field(default_factory=dict, description="Только плоские строковые значения")
    location: Optional[str] = Field(default=None, max_length=160)
    scene_items_add: List[ItemEntry] = Field(default_factory=list, min_items=0, max_items=12)
    scene_items_remove: List[str] = Field(default_factory=list, min_items=0, max_items=12)

    # Юниты
    players: List[PlayerEffect] = Field(default_factory=list, min_items=0, max_items=32)
    npcs: List[NpcEffect] = Field(default_factory=list, min_items=0, max_items=32)

    # Введение новых сущностей
    introductions: Introductions = Field(default_factory=Introductions)

# Совместимый псевдоним для некоторых мест кода/логов
class Effects(EffectsDelta):
    pass

# --------------------------- FACTS (normalized) ---------------------------

class FactRefs(BaseModel):
    """Связи факта с сущностями мира (для ссылочной валидности)."""
    model_config = ConfigDict(extra="forbid")
    items: List[str] = Field(default_factory=list, min_items=0, max_items=16)
    npcs: List[str] = Field(default_factory=list, min_items=0, max_items=16)
    locations: List[str] = Field(default_factory=list, min_items=0, max_items=16)

class VerifiedFact(BaseModel):
    """Нормализованный факт, полученный из Effects (public/private)."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., min_length=1, max_length=32)  # e.g. "item_move", "npc_mood", "world_flag"
    text: str = Field(..., min_length=3, max_length=280)
    public: bool = True
    player_id: Optional[str] = Field(default=None, max_length=64)
    refs: FactRefs = Field(default_factory=FactRefs)

# --------------------------- OUTLINES (bullet plans) ---------------------------

class Outline(BaseModel):
    """Скелет текста — ровно то, что нужно упомянуть в прозе."""
    model_config = ConfigDict(extra="forbid")
    bullets: List[str] = Field(..., min_items=1, max_items=24)
    must_include_ids: List[str] = Field(default_factory=list, min_items=0, max_items=24)

# --------------------------- PLAN (beats) ---------------------------

class BeatPlan(BaseModel):
    """Опорный план сцены для LLM: цель, ставки, препятствия, исходы."""
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=5, max_length=400)
    stakes: str = Field(..., min_length=5, max_length=400)
    obstacle: str = Field(..., min_length=5, max_length=400)
    twist_or_clue: Optional[str] = Field(default=None, min_length=0, max_length=200)
    scene_outcomes: List[str] = Field(..., min_items=1, max_items=3)
    must_mention: List[str] = Field(default_factory=list, min_items=0, max_items=12)
    forbidden: List[str] = Field(default_factory=list, min_items=0, max_items=20)
    open_threads_next: List[str] = Field(default_factory=list, min_items=0, max_items=4)

# --------------------------- STORIES (LLM) ---------------------------

class RawStory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=0, max_length=4000)

class EffectsAndRaw(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effects: EffectsDelta
    raw: RawStory

class PlayerStory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=2000)
    highlights: Optional[List[str]] = Field(default=None, min_items=0, max_items=8)
    echo_of_action: str = Field(..., min_length=1, max_length=400)

class GeneralStory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=4000)
    highlights: Optional[List[str]] = Field(default=None, min_items=0, max_items=8)

# --------------------------- TURN CONFIG/OUTPUTS ---------------------------

class TurnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_chat_id: int
    timeout: int = Field(60, ge=5, le=600)
    effects_n_predict: int = Field(700, ge=64, le=8192)
    private_n_predict: int = Field(700, ge=64, le=8192)
    general_n_predict: int = Field(1000, ge=64, le=8192)
    temperature_effects: float = Field(0.2, ge=0.0, le=1.0)
    temperature_private: float = Field(0.5, ge=0.0, le=1.0)
    temperature_general: float = Field(0.7, ge=0.0, le=1.0)
    llm_timeout: int = Field(480, ge=5, le=1200)

class TurnOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    effects: EffectsDelta
    raw: RawStory
    general: GeneralStory
    players_private: Dict[str, PlayerStory] = Field(default_factory=dict)
    verified_facts: List[VerifiedFact] = Field(default_factory=list)  # NEW
    general_outline: Optional[Outline] = None                           # NEW (для отладки/повторов)
    coverage: Optional[float] = Field(default=None, ge=0.0, le=1.0)     # NEW (метрика попадания фактов)
    turn: int
    telemetry: Dict[str, Any] = Field(default_factory=dict)

# --------------------------- JSON SCHEMA EXPORT ---------------------------

def as_json_schema(model_or_instance: Any) -> Dict[str, Any]:
    """Единая точка экспорта JSON Schema без санитайзеров.
    Принимает: класс/инстанс pydantic-модели или уже готовый dict.
    """
    if hasattr(model_or_instance, "model_json_schema"):
        return model_or_instance.model_json_schema()  # type: ignore[attr-defined]
    if hasattr(model_or_instance, "__class__") and hasattr(model_or_instance.__class__, "model_json_schema"):
        return model_or_instance.__class__.model_json_schema()  # type: ignore[attr-defined]
    if isinstance(model_or_instance, dict):
        return model_or_instance
    raise TypeError("as_json_schema: expected pydantic model/class or dict")

# --------------------------- АЛИАСЫ ДЛЯ СОВМЕСТИМОСТИ ---------------------------

class EffectsAndRawSchema(EffectsAndRaw):
    pass

class PlayerStorySchema(PlayerStory):
    pass

class GeneralStorySchema(GeneralStory):
    pass
