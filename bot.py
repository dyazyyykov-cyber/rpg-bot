from __future__ import annotations
import asyncio
import inspect
import json
import logging
import shutil
import sys
import importlib.util
import pathlib
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import redis.asyncio as aioredis
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from sqlalchemy import select, update
# --- make core importable when running `python bot.py` from repo root ---
ROOT = pathlib.Path(__file__).resolve().parent
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
    sys.path.insert(0, str(ROOT))
from core.utils import get_env, chunk_text
from core.db import AsyncSessionLocal, init_db, create_or_reset_session
# models: handle possible name collision
try:
    from core.models import Session, Player, SessionPlayer  # type: ignore
    _ = (Session, Player, SessionPlayer)
except Exception:
    spec = importlib.util.spec_from_file_location("core_db_models", CORE / "models.py")
    core_db_models = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(core_db_models)  # type: ignore[attr-defined]
    Session = core_db_models.Session        # type: ignore[attr-defined]
    Player = core_db_models.Player          # type: ignore[attr-defined]
    SessionPlayer = core_db_models.SessionPlayer  # type: ignore[attr-defined]
from core.session_init import (
    generate_world_v2,
    generate_roles_for_players,
    generate_initial_backstory,
)
# ---------------- ENV / Logging ----------------
LOGS_DIR = get_env("LOGS_DIR", "./logs")
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, str(LOG_LEVEL).upper(), logging.INFO),
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)
TG_TOKEN = get_env("TG_TOKEN", "")
if not TG_TOKEN:
    raise RuntimeError("TG_TOKEN is not set")
REDIS_URL = get_env("REDIS_URL", "redis://localhost:6379/0")
redis = aioredis.from_url(REDIS_URL)
bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()
dp.include_router(router)
# ---------------- Helpers ----------------
def _logs_base() -> Path:
    return Path(LOGS_DIR).expanduser()
async def _cleanup_session_logs(session_id: str) -> None:
    base = _logs_base().resolve()
    path = (base / f"session_{session_id}").resolve()
    try:
        if base in path.parents and path.name.startswith("session_") and path.is_dir():
            await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
            logger.info("LOGS: removed %s", path)
    except Exception as e:
        logger.warning("LOGS: failed to remove %s: %r", path, e)
def _lobby_kb(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Присоединиться", callback_data=f"join:{session_id}")],
            [InlineKeyboardButton(text="📝 Задать тему", callback_data=f"settheme:{session_id}")],
            [InlineKeyboardButton(text="🧹 Сбросить тему", callback_data=f"cleartheme:{session_id}")],
            [InlineKeyboardButton(text="🚀 Начать игру", callback_data=f"startworld:{session_id}")],
            [InlineKeyboardButton(text="👥 Участники", callback_data=f"roster:{session_id}")],
        ]
    )
async def _players_joined_list(session_id: str) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    async with AsyncSessionLocal() as dbs:
        q = (
            select(SessionPlayer, Player)
            .join(Player, SessionPlayer.player_id == Player.id)
            .where(SessionPlayer.session_id == session_id)
        )
        res = await dbs.execute(q)
        for sp, p in res.all():
            out.append((sp.player_id, p.tg_id, p.name or str(p.tg_id)))
    return out
async def _fetch_theme(session_id: str) -> str:
    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        st = dict(s.state or {}) if s else {}
        return str(st.get("story_theme") or "").strip()
async def _set_theme(session_id: str, theme: str) -> None:
    theme = theme.strip()
    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        if not s:
            return
        st = dict(s.state or {})
        st["story_theme"] = theme
        await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
        await dbs.commit()
async def _clear_theme(session_id: str) -> None:
    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        if not s:
            return
        st = dict(s.state or {})
        st.pop("story_theme", None)
        await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
        await dbs.commit()
async def _lobby_text(session_id: str) -> str:
    players = await _players_joined_list(session_id)
    theme = await _fetch_theme(session_id)
    lines = "\n".join(f"• {escape(name)}" for (_pid, _tg, name) in players) or "— пока никого"
    theme_line = f"\n\nТема: {escape(theme)}" if theme else "\n\nТема: не задана"
    return (
        "Создано лобби\n"
        "Сначала присоединитесь к игре по кнопке ниже. Когда все будут готовы — нажмите «Начать игру»."
        f"{theme_line}\n\nУчастники:\n{lines}"
    )
def _lobby_panel_key(session_id: str) -> str:
    return f"lobby_panel:{session_id}"
async def _save_lobby_panel(session_id: str, message_id: int) -> None:
    await redis.set(_lobby_panel_key(session_id), message_id, ex=7 * 24 * 3600)
async def _get_lobby_panel(session_id: str) -> Optional[int]:
    mid = await redis.get(_lobby_panel_key(session_id))
    return int(mid) if mid else None
async def _post_or_update_lobby_panel(chat_id: int, session_id: str) -> Message:
    text = await _lobby_text(session_id)
    panel_id = await _get_lobby_panel(session_id)
    if panel_id:
        try:
            return await bot.edit_message_text(
                chat_id=chat_id, message_id=panel_id, text=text, reply_markup=_lobby_kb(session_id)
            )
        except TelegramBadRequest:
            pass
    msg = await bot.send_message(chat_id, text, reply_markup=_lobby_kb(session_id))
    await _save_lobby_panel(session_id, msg.message_id)
    return msg
async def _send_chunked(chat_id: int, text: str) -> None:
    """Безопасная отправка длинных сообщений (4096-лимит Telegram)."""
    for part in chunk_text(text, hard_limit=4000):
        await bot.send_message(chat_id, part)
# ---------------- Commands ----------------
async def _register_commands() -> None:
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="newgame", description="Создать/сбросить сессию"),
                BotCommand(command="endgame", description="Завершить сессию"),
                BotCommand(command="startworld", description="Открыть лобби/запустить мир"),
                BotCommand(command="join", description="Присоединиться к лобби"),
                BotCommand(command="act", description="Личный ход (в ЛС)"),
            ]
        )
    except Exception:
        logger.exception("set_my_commands failed")
async def _maybe_await(v):
    return await v if inspect.isawaitable(v) else v
async def _send_or_edit_status(*, chat_id: int, status_msg: Optional[Message], text: str) -> Message:
    text = escape(text)
    if status_msg:
        try:
            m = await bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=text)
            logger.info("STATUS: edit -> %s", text)
            return m
        except TelegramBadRequest:
            logger.debug("STATUS: edit failed, will resend")
    m = await bot.send_message(chat_id, text)
    logger.info("STATUS: send -> %s", text)
    return m
async def _delete_status_silent(msg: Optional[Message]) -> None:
    if not msg:
        return
    try:
        await bot.delete_message(msg.chat.id, msg.message_id)
        logger.info("STATUS: deleted")
    except TelegramBadRequest:
        logger.debug("STATUS: already gone")
async def _ensure_player(*, tg_id: int, name: str) -> Player:
    async with AsyncSessionLocal() as dbs:
        p: Player | None = (
            await dbs.execute(select(Player).where(Player.tg_id == tg_id))
        ).scalar_one_or_none()
        if p is None:
            p = Player(tg_id=tg_id, name=name)
            dbs.add(p)
            await dbs.commit()
            await dbs.refresh(p)
            logger.info("PLAYER: created tg=%s name=%s", tg_id, name)
        return p
async def _join_session(*, session_id: str, tg_id: int, name: str) -> bool:
    player = await _ensure_player(tg_id=tg_id, name=name)
    async with AsyncSessionLocal() as dbs:
        sp: SessionPlayer | None = (
            await dbs.execute(
                select(SessionPlayer).where(
                    SessionPlayer.session_id == session_id, SessionPlayer.player_id == str(player.id)
                )
            )
        ).scalar_one_or_none()
        if sp:
            return False
        dbs.add(SessionPlayer(session_id=session_id, player_id=str(player.id), role=""))
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        if s:
            st = dict(s.state or {})
            players = list(st.get("players") or [])
            if not any(str(p.get("player_id")) == str(player.id) for p in players if isinstance(p, dict)):
                players.append({"player_id": str(player.id), "name": player.name or str(player.tg_id), "role": "", "role_summary": ""})
                st["players"] = players
                await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
        await dbs.commit()
    logger.info("JOIN: session=%s user=%s (%s)", session_id, tg_id, name)
    return True
async def _find_active_session_for_user(tg_id: int) -> Optional[str]:
    async with AsyncSessionLocal() as dbs:
        p: Player | None = (
            await dbs.execute(select(Player).where(Player.tg_id == tg_id))
        ).scalar_one_or_none()
        if not p:
            return None
        rows = await dbs.execute(
            select(Session)
            .join(SessionPlayer, SessionPlayer.session_id == Session.id)
            .where(SessionPlayer.player_id == str(p.id))
        )
        sessions: List[Session] = list(rows.scalars().all())
    for s in sessions:
        st = dict(s.state or {})
        ph = (st.get("phase") or "").lower()
        if ph == "running":
            return str(s.id)
    for s in sessions:
        st = dict(s.state or {})
        ph = (st.get("phase") or "").lower()
        if ph != "ended":
            return str(s.id)
    return None
# ---------------- Start world flow ----------------
async def _start_world_flow(*, session_id: str, group_chat_id: int, reply_target: Union[Message, CallbackQuery]) -> None:
    joined = await _players_joined_list(session_id)
    if not joined:
        await bot.send_message(group_chat_id, "Никто не присоединился. Сначала «Присоединиться», затем «Начать игру».")
        return
    logger.info("STARTWORLD: players=%d session=%s", len(joined), session_id)
    status_msg: Optional[Message] = await _send_or_edit_status(chat_id=group_chat_id, status_msg=None, text="Генерирую мир…")
    async with AsyncSessionLocal() as dbs:
        sess: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        title = (sess.title if sess else "") or "Игра"
        story_theme = (dict(sess.state or {}).get("story_theme") or "") if sess else ""
    players_payload = [{"player_id": pid, "name": name} for (pid, _tg, name) in joined]
    world_state: Dict[str, Any] = await _maybe_await(
        generate_world_v2(
            group_chat_id=group_chat_id,
            title=title,
            players=players_payload,
            story_theme=story_theme,
            timeout=None,
        )
    )
    if story_theme and not world_state.get("story_theme"):
        world_state["story_theme"] = story_theme
    async with AsyncSessionLocal() as dbs:
        await dbs.execute(update(Session).where(Session.id == session_id).values(state=world_state))
        await dbs.commit()
    logger.info("STARTWORLD: world generated")
    status_msg = await _send_or_edit_status(chat_id=group_chat_id, status_msg=status_msg, text="Распределяю роли…")
    roles = await _maybe_await(
        generate_roles_for_players(
            state=world_state,
            players=players_payload,
            story_theme=story_theme,
            timeout=None,
        )
    )
    roles_by_pid: Dict[str, Dict[str, str]] = {r["player_id"]: r for r in roles}
    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        st = dict(s.state or {}) if s else {}
        st_players = st.get("players") or []
        by_id = {str(p.get("player_id")): p for p in st_players if isinstance(p, dict)}
        for pid, _tg, name in joined:
            slot = by_id.get(pid) or {"player_id": pid, "name": name}
            r = roles_by_pid.get(pid) or {}
            if r:
                slot["role"] = r.get("role") or slot.get("role") or ""
                slot["role_summary"] = r.get("summary") or slot.get("role_summary") or ""
            by_id[pid] = slot
        st["players"] = list(by_id.values())
        await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
        for pid, _tg, _name in joined:
            r = roles_by_pid.get(pid)
            if r:
                await dbs.execute(
                    update(SessionPlayer)
                    .where(SessionPlayer.session_id == session_id, SessionPlayer.player_id == pid)
                    .values(role=r.get("role") or "")
                )
        await dbs.commit()
    logger.info("STARTWORLD: roles assigned")
    status_msg = await _send_or_edit_status(chat_id=
