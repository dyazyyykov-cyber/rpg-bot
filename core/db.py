from __future__ import annotations

"""
Async DB слой (SQLAlchemy 2.x + aiosqlite/Postgres), совместимый со старой схемой,
но с поддержкой новых полей состояния сессии:
  - open_threads: list[str]
  - clocks: dict[str, int]
  - turn_metrics: list[{turn, ts, metrics}]
  - beat_plans: list[{turn, ts, plan}]

Ключевое: в init_db() есть лёгкая авто-миграция для SQLite (ALTER TABLE … ADD COLUMN),
которая чинит ошибки вида "no such column: players.created_at".
"""

import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, and_, delete, text

from .models import Base, Player, Session, SessionPlayer, Event, Item, Inventory
from .utils import get_env

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Engine & Session
# -----------------------------------------------------------------------------

DATABASE_URL = get_env("DATABASE_URL", "sqlite+aiosqlite:///./game.db")
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# -----------------------------------------------------------------------------
# Schema init (create + lightweight migrations for SQLite)
# -----------------------------------------------------------------------------

async def _sqlite_table_has_column(conn, table: str, column: str) -> bool:
    res = await conn.exec_driver_sql(f"PRAGMA table_info({table})")
    rows = res.fetchall()
    for row in rows:
        # row = (cid, name, type, notnull, dflt_value, pk)
        if len(row) >= 2 and str(row[1]).lower() == column.lower():
            return True
    return False


async def _ensure_sqlite_columns(conn) -> None:
    """Добавляем недостающие столбцы для старых инсталляций (SQLite). Безопасно при повторных запусках."""
    # players.created_at
    if not await _sqlite_table_has_column(conn, "players", "created_at"):
        await conn.exec_driver_sql(
            "ALTER TABLE players ADD COLUMN created_at DATETIME DEFAULT (datetime('now'))"
        )
        logger.info("DB MIGRATE: added players.created_at")

    # sessions.created_at / updated_at
    if not await _sqlite_table_has_column(conn, "sessions", "created_at"):
        await conn.exec_driver_sql(
            "ALTER TABLE sessions ADD COLUMN created_at DATETIME DEFAULT (datetime('now'))"
        )
        logger.info("DB MIGRATE: added sessions.created_at")
    if not await _sqlite_table_has_column(conn, "sessions", "updated_at"):
        await conn.exec_driver_sql(
            "ALTER TABLE sessions ADD COLUMN updated_at DATETIME DEFAULT (datetime('now'))"
        )
        logger.info("DB MIGRATE: added sessions.updated_at")

    # session_players: новые поля, встречаются в коде
    if not await _sqlite_table_has_column(conn, "session_players", "status"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN status JSON DEFAULT '{}'"
        )
        logger.info("DB MIGRATE: added session_players.status")
    if not await _sqlite_table_has_column(conn, "session_players", "position"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN position TEXT DEFAULT 'scene'"
        )
        logger.info("DB MIGRATE: added session_players.position")
    if not await _sqlite_table_has_column(conn, "session_players", "dm_chat_id"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN dm_chat_id INTEGER"
        )
        logger.info("DB MIGRATE: added session_players.dm_chat_id")
    if not await _sqlite_table_has_column(conn, "session_players", "dm_ok"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN dm_ok INTEGER DEFAULT 0"
        )
        logger.info("DB MIGRATE: added session_players.dm_ok")
    if not await _sqlite_table_has_column(conn, "session_players", "last_severity"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN last_severity TEXT"
        )
        logger.info("DB MIGRATE: added session_players.last_severity")
    if not await _sqlite_table_has_column(conn, "session_players", "last_severity_count"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN last_severity_count INTEGER DEFAULT 0"
        )
        logger.info("DB MIGRATE: added session_players.last_severity_count")
    if not await _sqlite_table_has_column(conn, "session_players", "created_at"):
        await conn.exec_driver_sql(
            "ALTER TABLE session_players ADD COLUMN created_at DATETIME DEFAULT (datetime('now'))"
        )
        logger.info("DB MIGRATE: added session_players.created_at")

    # events: source/validated/created_at
    if not await _sqlite_table_has_column(conn, "events", "source"):
        await conn.exec_driver_sql(
            "ALTER TABLE events ADD COLUMN source TEXT DEFAULT 'llm'"
        )
        logger.info("DB MIGRATE: added events.source")
    if not await _sqlite_table_has_column(conn, "events", "validated"):
        await conn.exec_driver_sql(
            "ALTER TABLE events ADD COLUMN validated INTEGER DEFAULT 1"
        )
        logger.info("DB MIGRATE: added events.validated")
    if not await _sqlite_table_has_column(conn, "events", "created_at"):
        await conn.exec_driver_sql(
            "ALTER TABLE events ADD COLUMN created_at DATETIME DEFAULT (datetime('now'))"
        )
        logger.info("DB MIGRATE: added events.created_at")

    # items.props
    if not await _sqlite_table_has_column(conn, "items", "props"):
        await conn.exec_driver_sql(
            "ALTER TABLE items ADD COLUMN props JSON DEFAULT '{}'"
        )
        logger.info("DB MIGRATE: added items.props")

    # inventory.quantity
    if not await _sqlite_table_has_column(conn, "inventory", "quantity"):
        await conn.exec_driver_sql(
            "ALTER TABLE inventory ADD COLUMN quantity INTEGER DEFAULT 1"
        )
        logger.info("DB MIGRATE: added inventory.quantity")


async def init_db() -> None:
    """Создать таблицы при первом запуске и выполнить лёгкую миграцию для SQLite."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Авто-миграции только для SQLite
        if DATABASE_URL.startswith("sqlite"):
            await _ensure_sqlite_columns(conn)

        # Тюнинг SQLite (не критично, но ускоряет)
        if DATABASE_URL.startswith("sqlite"):
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")

    logger.info("DB: init ok")

# -----------------------------------------------------------------------------
# Helpers: state defaults & one-shot migration
# -----------------------------------------------------------------------------

def _ensure_state_defaults(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Добавляет недостающие поля в JSON-состояние. Возвращает (state, changed)."""
    changed = False
    if not isinstance(state, dict):
        return {}, True
    if "open_threads" not in state or not isinstance(state.get("open_threads"), list):
        state["open_threads"] = []
        changed = True
    if "clocks" not in state or not isinstance(state.get("clocks"), dict):
        state["clocks"] = {}
        changed = True
    if "turn_metrics" not in state or not isinstance(state.get("turn_metrics"), list):
        state["turn_metrics"] = []
        changed = True
    if "beat_plans" not in state or not isinstance(state.get("beat_plans"), list):
        state["beat_plans"] = []
        changed = True
    return state, changed


_JSONISH_RE = re.compile(r"^\s*[\[{]")
_CTRL_RE = re.compile("[\u0000-\u001F\u007F]")


def _clean_location_value(loc: Any) -> str:
    """Приводит location к аккуратной короткой строке. Удаляет control-символы и JSON-подобные мусорные фрагменты."""
    if loc is None:
        return ""
    if isinstance(loc, (dict, list)):
        return ""
    s = str(loc).strip()
    if not s or s.lower() in {"none", "null", "nan", "undefined"}:
        return ""
    if _JSONISH_RE.match(s):
        # в старых сессиях могли попасть сериализованные объекты
        return ""
    s = _CTRL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160]


def _migrate_state_on_read(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Одноразовая миграция при чтении state: чистим location и добавляем поля по умолчанию."""
    changed = False
    if not isinstance(state, dict):
        return {}, True

    # ensure defaults first
    state, ch = _ensure_state_defaults(state)
    changed |= ch

    # location cleanup (флажок, чтобы не делать повторно)
    mig_flag = "_migrated_location_cleanup"
    if not state.get(mig_flag):
        new_loc = _clean_location_value(state.get("location"))
        if new_loc != (state.get("location") or ""):
            state["location"] = new_loc
            changed = True
        state[mig_flag] = True
        changed = True

    return state, changed

# -----------------------------------------------------------------------------
# Sessions
# -----------------------------------------------------------------------------

async def get_session(session_id: str) -> Optional[Session]:
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Session).where(Session.id == session_id))
        s = res.scalar_one_or_none()
        return s

async def create_or_reset_session(
    session_id: str,
    group_chat_id: int,
    title: str,
    initial_state: Dict[str, Any],
) -> Session:
    """
    Полный сброс сессии:
      - чистим участников, инвентарь и события
      - записываем новое состояние (с автодобавлением новых полей)
    """
    async with AsyncSessionLocal() as db:
        # подчистка зависимых данных
        await db.execute(delete(SessionPlayer).where(SessionPlayer.session_id == session_id))
        await db.execute(delete(Inventory).where(Inventory.session_id == session_id))
        await db.execute(delete(Event).where(Event.session_id == session_id))
        await db.commit()

        res = await db.execute(select(Session).where(Session.id == session_id))
        s = res.scalar_one_or_none()

        st = dict(initial_state or {})
        st, _ = _ensure_state_defaults(st)

        if s is None:
            s = Session(
                id=session_id,
                title=title,
                group_chat_id=group_chat_id,
                state=st,
            )
            db.add(s)
        else:
            s.title = title
            s.group_chat_id = group_chat_id
            s.state = st
            db.add(s)

        await db.commit()
        await db.refresh(s)
        logger.info("DB: session %s reset ok", session_id)
        return s

async def update_session_state(session_id: str, new_state: Dict[str, Any]) -> Optional[Session]:
    """Полностью заменить JSON-состояние сессии (с авто-добавлением новых полей)."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Session).where(Session.id == session_id))
        s = res.scalar_one_or_none()
        if not s:
            return None
        st = dict(new_state or {})
        st, _ = _ensure_state_defaults(st)
        s.state = st
        db.add(s)
        await db.commit()
        await db.refresh(s)
        return s

async def get_state(session_id: str) -> Dict[str, Any]:
    """Вернуть state сессии; на лету сделать одноразовую «миграцию на чтение» (cleanup location, ensure defaults)."""
    s = await get_session(session_id)
    if not s:
        logger.warning("DB: get_state — session %s not found, returning {}", session_id)
        return {}

    st = dict(s.state or {})
    st2, changed = _migrate_state_on_read(st)
    if changed:
        # бесшумно зафиксируем миграцию
        await update_session_state(session_id, st2)
        return st2
    return st2

async def save_state(session_id: str, state: Dict[str, Any]) -> None:
    """Сохранить state сессии. Если сессия не найдена — тихо логируем."""
    s = await update_session_state(session_id, state or {})
    if not s:
        logger.warning("DB: save_state — session %s not found (state not saved)", session_id)

# -----------------------------------------------------------------------------
# New helpers for open_threads / clocks / artifacts / metrics
# -----------------------------------------------------------------------------

async def set_open_threads(session_id: str, threads: List[str]) -> Dict[str, Any]:
    st = await get_state(session_id)
    st["open_threads"] = [str(t).strip() for t in (threads or []) if str(t).strip()]
    await save_state(session_id, st)
    return st

async def add_open_thread(session_id: str, thread: str) -> Dict[str, Any]:
    st = await get_state(session_id)
    lst = st.get("open_threads") or []
    val = str(thread).strip()
    if val and val not in lst:
        lst.append(val)
        st["open_threads"] = lst
        await save_state(session_id, st)
    return st

async def close_open_thread(session_id: str, thread: str) -> Dict[str, Any]:
    st = await get_state(session_id)
    lst = [x for x in (st.get("open_threads") or []) if x != thread]
    st["open_threads"] = lst
    await save_state(session_id, st)
    return st

async def upsert_clocks(session_id: str, clocks: Dict[str, int]) -> Dict[str, Any]:
    st = await get_state(session_id)
    cur = st.get("clocks") or {}
    for k, v in (clocks or {}).items():
        try:
            cur[str(k)] = int(v)
        except Exception:
            continue
    st["clocks"] = cur
    await save_state(session_id, st)
    return st

async def set_clocks(session_id: str, clocks: Dict[str, int]) -> Dict[str, Any]:
    st = await get_state(session_id)
    clean: Dict[str, int] = {}
    for k, v in (clocks or {}).items():
        try:
            clean[str(k)] = int(v)
        except Exception:
            continue
    st["clocks"] = clean
    await save_state(session_id, st)
    return st

async def save_beat_plan(session_id: str, turn: int, plan: Dict[str, Any]) -> Dict[str, Any]:
    st = await get_state(session_id)
    arr = st.get("beat_plans") or []
    arr.append({"turn": int(turn), "ts": datetime.utcnow().isoformat() + "Z", "plan": plan or {}})
    st["beat_plans"] = arr[-200:]  # ограничим историю
    await save_state(session_id, st)
    return st

async def save_turn_metrics(session_id: str, turn: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    st = await get_state(session_id)
    arr = st.get("turn_metrics") or []
    # если запись за этот ход уже была — обновим
    found = False
    for row in arr:
        if int(row.get("turn", -1)) == int(turn):
            row["metrics"] = metrics or {}
            row["ts"] = datetime.utcnow().isoformat() + "Z"
            found = True
            break
    if not found:
        arr.append({"turn": int(turn), "ts": datetime.utcnow().isoformat() + "Z", "metrics": metrics or {}})
    st["turn_metrics"] = arr[-500:]
    await save_state(session_id, st)
    return st

async def last_beat_plan(session_id: str) -> Optional[Dict[str, Any]]:
    st = await get_state(session_id)
    arr = st.get("beat_plans") or []
    return (arr[-1] if arr else None)

async def get_turn_metrics(session_id: str, turn: int) -> Optional[Dict[str, Any]]:
    st = await get_state(session_id)
    for row in (st.get("turn_metrics") or []):
        if int(row.get("turn", -1)) == int(turn):
            return row
    return None

# -----------------------------------------------------------------------------
# Players
# -----------------------------------------------------------------------------

async def get_or_create_player(tg_id: int, name: str) -> Player:
    """Найти игрока по Telegram ID или создать. Обновляет имя, если оно изменилось."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Player).where(Player.tg_id == tg_id))
        p = res.scalar_one_or_none()
        if p:
            if p.name != name:
                p.name = name
                db.add(p)
                await db.commit()
                await db.refresh(p)
            return p
        p = Player(tg_id=tg_id, name=name)
        db.add(p)
        await db.commit()
        await db.refresh(p)
        return p

async def get_player_by_tg(tg_id: int) -> Optional[Player]:
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Player).where(Player.tg_id == tg_id))
        return res.scalar_one_or_none()

async def list_players() -> List[Player]:
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Player))
        return list(res.scalars().all())

# -----------------------------------------------------------------------------
# Session membership
# -----------------------------------------------------------------------------

async def get_session_players(session_id: str) -> List[SessionPlayer]:
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(SessionPlayer).where(SessionPlayer.session_id == session_id))
        return list(res.scalars().all())

async def join_session(
    session_id: str,
    player_id: str,
    *,
    dm_chat_id: Optional[int] = None,
    dm_ok: Optional[bool] = None,
) -> SessionPlayer:
    """Привязка игрока к сессии (idempotent). Роли не задаём — они генерируются позже."""
    async with AsyncSessionLocal() as db:
        # проверим, что сессия существует
        res_s = await db.execute(select(Session).where(Session.id == session_id))
        s = res_s.scalar_one_or_none()
        if not s:
            raise RuntimeError("Session not found")

        # уже в сессии?
        res = await db.execute(
            select(SessionPlayer).where(
                and_(SessionPlayer.session_id == session_id, SessionPlayer.player_id == player_id)
            )
        )
        sp = res.scalar_one_or_none()
        if sp:
            changed = False
            if dm_chat_id is not None and sp.dm_chat_id != dm_chat_id:
                sp.dm_chat_id = dm_chat_id
                changed = True
            if dm_ok is not None and sp.dm_ok != dm_ok:
                sp.dm_ok = dm_ok
                changed = True
            if changed:
                db.add(sp)
                await db.commit()
                await db.refresh(sp)
            return sp

        sp = SessionPlayer(session_id=session_id, player_id=player_id)
        if dm_chat_id is not None:
            sp.dm_chat_id = dm_chat_id
        if dm_ok is not None:
            sp.dm_ok = dm_ok

        db.add(sp)
        await db.commit()
        await db.refresh(sp)
        return sp

async def set_session_player_dm(
    session_id: str,
    player_id: str,
    *,
    dm_chat_id: Optional[int],
    dm_ok: bool,
) -> Optional[SessionPlayer]:
    """Обновить статус лички для игрока в рамках сессии."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(SessionPlayer).where(
                and_(SessionPlayer.session_id == session_id, SessionPlayer.player_id == player_id)
            )
        )
        sp = res.scalar_one_or_none()
        if not sp:
            return None
        sp.dm_chat_id = dm_chat_id
        sp.dm_ok = dm_ok
        db.add(sp)
        await db.commit()
        await db.refresh(sp)
        return sp

async def set_session_player_role(session_id: str, player_id: str, role: Optional[str]) -> Optional[SessionPlayer]:
    """Назначить роль игроку (после генерации мира/ролей)."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(SessionPlayer).where(
                and_(SessionPlayer.session_id == session_id, SessionPlayer.player_id == player_id)
            )
        )
        sp = res.scalar_one_or_none()
        if not sp:
            return None
        sp.role = role
        db.add(sp)
        await db.commit()
        await db.refresh(sp)
        return sp

async def set_session_player_hp(
    session_id: str,
    player_id: str,
    *,
    hp: Optional[int] = None,
    max_hp: Optional[int] = None,
) -> Optional[SessionPlayer]:
    """Обновить HP/MaxHP игрока."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(SessionPlayer).where(
                and_(SessionPlayer.session_id == session_id, SessionPlayer.player_id == player_id)
            )
        )
        sp = res.scalar_one_or_none()
        if not sp:
            return None
        if hp is not None:
            sp.hp = hp
        if max_hp is not None:
            sp.max_hp = max_hp
        db.add(sp)
        await db.commit()
        await db.refresh(sp)
        return sp

async def set_session_player_severity(
    session_id: str,
    player_id: str,
    *,
    severity: Optional[str],
    since_turns: int,
) -> Optional[SessionPlayer]:
    """
    Сохранить «тяжесть» состояния персонажа для вариативных описаний LLM.
    severity: "healthy" | "light" | "moderate" | "critical" | None
    since_turns: сколько ходов подряд держится текущая тяжесть.
    """
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(SessionPlayer).where(
                and_(SessionPlayer.session_id == session_id, SessionPlayer.player_id == player_id)
            )
        )
        sp = res.scalar_one_or_none()
        if not sp:
            return None
        sp.last_severity = severity
        sp.last_severity_count = since_turns
        db.add(sp)
        await db.commit()
        await db.refresh(sp)
        return sp

# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

async def save_event(
    session_id: str,
    actor_id: Optional[str],
    type_: str,
    payload: Dict[str, Any],
    source: str = "llm",
    validated: int = 1,
) -> Event:
    """Сохранить событие (эффект, приватка, общее и т.п.)."""
    async with AsyncSessionLocal() as db:
        ev = Event(
            session_id=session_id,
            actor_id=actor_id,
            type=type_,
            payload=payload or {},
            source=source,
            validated=validated,
        )
        db.add(ev)
        await db.commit()
        await db.refresh(ev)
        return ev

async def list_events_tail(session_id: str, limit: int = 30) -> List[Event]:
    """
    Последние события по id (для SQLite автоинкремент совпадает со временем вставки).
    """
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Event)
            .where(Event.session_id == session_id)
            .order_by(Event.id.desc())
            .limit(limit)
        )
        rows = list(res.scalars().all())
        rows.reverse()
        return rows

# -----------------------------------------------------------------------------
# Items & Inventory
# -----------------------------------------------------------------------------

async def upsert_item(name: str, props: Optional[Dict[str, Any]] = None) -> Item:
    """Upsert по имени предмета."""
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Item).where(Item.name == name))
        it = res.scalar_one_or_none()
        if it:
            return it
        it = Item(name=name, props=props or {})
        db.add(it)
        await db.commit()
        await db.refresh(it)
        return it

async def add_to_inventory(session_id: str, player_id: str, item_id: str, qty: int = 1) -> Inventory:
    async with AsyncSessionLocal() as db:
        inv = Inventory(session_id=session_id, player_id=player_id, item_id=item_id, quantity=qty)
        db.add(inv)
        await db.commit()
        await db.refresh(inv)
        return inv

async def list_inventory(session_id: str, player_id: str) -> List[Inventory]:
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Inventory).where(
                and_(Inventory.session_id == session_id, Inventory.player_id == player_id)
            )
        )
        return list(res.scalars().all())

# -----------------------------------------------------------------------------
# Snapshots for prompts (used by worker/engine)
# -----------------------------------------------------------------------------

async def build_players_snapshot(session_id: str) -> List[Dict[str, Any]]:
    """
    Снимок игроков для промптов. Ключ 'role' добавляем только если он задан.
    """
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(SessionPlayer).where(SessionPlayer.session_id == session_id))
        rows = list(res.scalars().all())

        # подтянем имена
        player_ids = [r.player_id for r in rows]
        if not player_ids:
            return []
        res_p = await db.execute(select(Player).where(Player.id.in_(player_ids)))
        players = {p.id: p for p in res_p.scalars().all()}

        snap: List[Dict[str, Any]] = []
        for r in rows:
            p = players.get(r.player_id)
            entry: Dict[str, Any] = {
                "player_id": r.player_id,
                "name": (p.name if p and p.name else (str(p.tg_id) if p else "Unknown")),
                "hp": r.hp,
                "max_hp": r.max_hp,
                "position": r.position or "scene",
                "status": r.status or {},
                "items": [],
            }
            if r.role:
                entry["role"] = r.role
            snap.append(entry)
        return snap

# -----------------------------------------------------------------------------
# Maintenance
# -----------------------------------------------------------------------------

async def purge_session_data(session_id: str) -> None:
    """Полная очистка данных сессии (участники, инвентарь, события)."""
    async with AsyncSessionLocal() as db:
        await db.execute(delete(SessionPlayer).where(SessionPlayer.session_id == session_id))
        await db.execute(delete(Inventory).where(Inventory.session_id == session_id))
        await db.execute(delete(Event).where(Event.session_id == session_id))
        await db.commit()
        logger.info("DB: purge done for session %s", session_id)

async def destroy_session(session_id: str) -> None:
    """
    Полное удаление сессии и всех её зависимостей.
    Удаляет:
      - SessionPlayer
      - Inventory
      - Event
      - Session
    """
    async with AsyncSessionLocal() as db:
        await db.execute(delete(SessionPlayer).where(SessionPlayer.session_id == session_id))
        await db.execute(delete(Inventory).where(Inventory.session_id == session_id))
        await db.execute(delete(Event).where(Event.session_id == session_id))
        await db.execute(delete(Session).where(Session.id == session_id))
        await db.commit()
        logger.info("DB: session %s destroyed", session_id)
