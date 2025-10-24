from __future__ import annotations

"""
SQLAlchemy 2.x модели для бота.
Совместимость: используется обычное declarative_base (sync-модели),
а сессии/движок на async настраиваются в db.py.

Экспортируемые классы:
  Base, Player, Session, SessionPlayer, Event, Item, Inventory
"""

import datetime
import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

# -----------------------------------------------------------------------------
# База
# -----------------------------------------------------------------------------

Base = declarative_base()


def gen_uuid() -> str:
    """UUID4 как строка — удобно для Item.id и Player.id."""
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Игрок
# -----------------------------------------------------------------------------

class Player(Base):
    __tablename__ = "players"

    id = Column(String, primary_key=True, default=gen_uuid)
    tg_id = Column(Integer, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)

    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Player id={self.id} tg_id={self.tg_id} name={self.name!r}>"


# -----------------------------------------------------------------------------
# Сессия (одна игра/чат)
# -----------------------------------------------------------------------------

class Session(Base):
    __tablename__ = "sessions"

    # Строковый идентификатор = обычно str(group_chat_id)
    id = Column(String, primary_key=True)

    title = Column(String, nullable=False)
    group_chat_id = Column(Integer, unique=True, index=True, nullable=False)

    # JSON-снимок мира (инициализируется и затем модифицируется движком)
    # Здесь живут: setting/location/world_flags/players/npcs/...
    # а также story_theme/beat_plans/effects_log/general_history и пр.
    state = Column(JSON, default=dict, nullable=False)

    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Session id={self.id} title={self.title!r} group_chat_id={self.group_chat_id}>"


# -----------------------------------------------------------------------------
# Игрок в сессии (состояние и метаданные)
# -----------------------------------------------------------------------------

class SessionPlayer(Base):
    __tablename__ = "session_players"

    id = Column(Integer, primary_key=True, autoincrement=True)

    session_id = Column(String, ForeignKey("sessions.id"), index=True, nullable=False)
    player_id = Column(String, ForeignKey("players.id"), index=True, nullable=False)

    # Роль отсутствует до генерации
    role = Column(String, nullable=True, default=None)

    # HP-значения (числа приходят из эффектов; здесь только хранение)
    hp = Column(Integer, nullable=False, default=100)
    max_hp = Column(Integer, nullable=False, default=100)

    # Произвольный контейнер для временных статусов/меток
    status = Column(JSON, default=dict, nullable=False)

    # scene | hidden | off | <custom>
    position = Column(String, nullable=False, default="scene")

    # --- DM (приватный канал) ---
    # chat_id лички с пользователем (обычно == tg_id); если нет/запрещено — NULL
    dm_chat_id = Column(Integer, nullable=True, default=None)
    # последний успешный контакт в DM (True) / бот заблокирован/недоступен (False)
    dm_ok = Column(Boolean, nullable=False, default=False)

    # --- Служебные поля для вариативных описаний ---
    # "healthy" | "light" | "moderate" | "critical" (или None, если не вычислялось)
    last_severity = Column(String, nullable=True, default=None)
    # сколько ходов подряд держится текущая тяжесть (для вариаций формулировок)
    last_severity_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("session_id", "player_id", name="uq_session_player"),
        Index("ix_session_players_session", "session_id"),
        Index("ix_session_players_player", "player_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        role = self.role if self.role else "-"
        dm = "ok" if self.dm_ok else "no-dm"
        return (
            f"<SessionPlayer id={self.id} session={self.session_id} "
            f"player={self.player_id} role={role!r} hp={self.hp}/{self.max_hp} {dm}>"
        )


# -----------------------------------------------------------------------------
# Журнал событий
# -----------------------------------------------------------------------------

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    session_id = Column(String, ForeignKey("sessions.id"), index=True, nullable=False)
    actor_id = Column(String, ForeignKey("players.id"), index=True, nullable=True)

    # Короткий тип: damage | move | loot | status | npc_react | world_flag | reveal | action_echo | narrative | ...
    type = Column(String, nullable=False)

    payload = Column(JSON, default=dict, nullable=False)
    source = Column(String, nullable=False, default="llm")  # 'llm' | 'kernel' | 'system'
    validated = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    __table_args__ = (
        Index("ix_events_session_created", "session_id", "created_at"),
        Index("ix_events_actor_created", "actor_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Event id={self.id} session={self.session_id} actor={self.actor_id} "
            f"type={self.type!r} created_at={self.created_at}>"
        )


# -----------------------------------------------------------------------------
# Предмет и инвентарь
# -----------------------------------------------------------------------------

class Item(Base):
    __tablename__ = "items"

    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False, index=True)
    props = Column(JSON, default=dict, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Item id={self.id} name={self.name!r}>"


class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, autoincrement=True)

    session_id = Column(String, ForeignKey("sessions.id"), index=True, nullable=False)
    player_id = Column(String, ForeignKey("players.id"), index=True, nullable=False)
    item_id = Column(String, ForeignKey("items.id"), index=True, nullable=False)

    quantity = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        Index("ix_inventory_session_player", "session_id", "player_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Inventory id={self.id} session={self.session_id} "
            f"player={self.player_id} item={self.item_id} qty={self.quantity}>"
        )


__all__ = [
    "Base",
    "Player",
    "Session",
    "SessionPlayer",
    "Event",
    "Item",
    "Inventory",
]
