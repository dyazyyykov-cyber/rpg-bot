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
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
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
from core.db import (
    AsyncSessionLocal,
    init_db,
    create_or_reset_session,
    get_or_create_player,
    get_player_by_tg,
    join_session,
)

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
    theme_line = f"\n\n<b>Тема:</b> {escape(theme)}" if theme else "\n\n<b>Тема:</b> не задана"
    return (
        "<b>Создано лобби</b>\n"
        "Сначала присоединитесь к игре по кнопке ниже. Когда все будут готовы — нажмите «Начать игру»."
        f"{theme_line}\n\n<b>Участники:</b>\n{lines}"
    )

def _lobby_panel_key(session_id: str) -> str:
    return f"lobby_panel:{session_id}"

async def _save_lobby_panel(session_id: str, message_id: int) -> None:
    await redis.set(_lobby_panel_key(session_id), message_id, ex=7 * 24 * 3600)

async def _get_lobby_panel(session_id: str) -> Optional[int]:
    val = await redis.get(_lobby_panel_key(session_id))
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None

async def _send_or_edit_status(*, chat_id: int, status_msg: Optional[Message], text: str) -> Message:
    """Send a new status message or update the existing one."""
    if status_msg is not None:
        try:
            return await bot.edit_message_text(
                text,
                chat_id=chat_id,
                message_id=status_msg.message_id,
            )
        except TelegramBadRequest:
            pass
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("STATUS: failed to edit message %s: %r", status_msg.message_id, exc)
    sent = await bot.send_message(chat_id, text)
    return sent

async def _delete_status_silent(status_msg: Optional[Message]) -> None:
    if not status_msg:
        return
    try:
        await bot.delete_message(chat_id=status_msg.chat.id, message_id=status_msg.message_id)
    except TelegramBadRequest:
        pass
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning(
            "STATUS: failed to delete message %s in chat %s: %r",
            status_msg.message_id,
            status_msg.chat.id,
            exc,
        )

async def _send_chunked(chat_id: int, text: str) -> None:
    for part in chunk_text(text, hard_limit=4000):
        await bot.send_message(chat_id, part)

async def _post_or_update_lobby_panel(chat_id: int, session_id: str) -> Message:
    """Post a lobby panel message or update the cached one if possible."""
    text = await _lobby_text(session_id)
    keyboard = _lobby_kb(session_id)
    msg_id = await _get_lobby_panel(session_id)
    if msg_id is not None:
        try:
            return await bot.edit_message_text(
                text,
                chat_id=chat_id,
                message_id=msg_id,
                reply_markup=keyboard,
            )
        except TelegramBadRequest:
            pass
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning(
                "LOBBY: failed to edit panel sid=%s msg=%s: %r",
                session_id,
                msg_id,
                exc,
            )
    sent = await bot.send_message(chat_id, text, reply_markup=keyboard)
    return sent

async def _find_active_session_for_user(tg_id: int) -> Optional[str]:
    player = await get_player_by_tg(tg_id)
    if not player:
        return None

    try:
        active_raw = await redis.smembers("sessions:active")
        active_ids = []
        for raw in active_raw or []:
            if isinstance(raw, bytes):
                active_ids.append(raw.decode("utf-8"))
            else:
                active_ids.append(str(raw))
    except Exception:
        active_ids = []

    async with AsyncSessionLocal() as dbs:
        res = await dbs.execute(
            select(SessionPlayer.session_id).where(SessionPlayer.player_id == player.id)
        )
        joined_ids = list(res.scalars().all())
        if not joined_ids:
            return None

        res_sessions = await dbs.execute(
            select(Session.id, Session.state).where(Session.id.in_(joined_ids))
        )
        session_states = {sid: dict(state or {}) for sid, state in res_sessions.all()}

    for sid in active_ids:
        st = session_states.get(sid)
        if st and st.get("phase") == "running":
            return sid

    for sid in joined_ids:
        st = session_states.get(sid)
        if st and st.get("phase") == "running":
            return sid

    return None

async def _join_session(
    *,
    session_id: str,
    tg_id: int,
    name: str,
    dm_chat_id: Optional[int] = None,
    dm_ok: Optional[bool] = None,
) -> bool:
    """Register a Telegram user in the session. Returns True if newly added."""
    clean_name = (name or "").strip() or str(tg_id)
    player = await get_or_create_player(tg_id=tg_id, name=clean_name)

    async with AsyncSessionLocal() as dbs:
        sess = (
            await dbs.execute(select(Session).where(Session.id == session_id))
        ).scalar_one_or_none()
        if not sess:
            raise RuntimeError(f"Session {session_id} not found")
        state = dict(sess.state or {})
        join_locked = bool(state.get("join_locked"))

    async with AsyncSessionLocal() as dbs:
        res = await dbs.execute(
            select(SessionPlayer).where(
                SessionPlayer.session_id == session_id,
                SessionPlayer.player_id == player.id,
            )
        )
        existing = res.scalar_one_or_none()

    if join_locked and existing is None:
        logger.info("JOIN: session %s is locked", session_id)
        return False

    await join_session(
        session_id=session_id,
        player_id=player.id,
        dm_chat_id=dm_chat_id,
        dm_ok=dm_ok,
    )

    async with AsyncSessionLocal() as dbs:
        sess = (
            await dbs.execute(select(Session).where(Session.id == session_id))
        ).scalar_one_or_none()
        if sess:
            state = dict(sess.state or {})
            players = list(state.get("players") or [])
            updated = False
            found = False
            for idx, info in enumerate(players):
                if str(info.get("player_id")) == player.id:
                    found = True
                    new_info = dict(info)
                    entry_changed = False
                    if new_info.get("name") != clean_name:
                        new_info["name"] = clean_name
                        entry_changed = True
                    if new_info.get("tg_id") != tg_id:
                        new_info["tg_id"] = tg_id
                        entry_changed = True
                    if entry_changed:
                        players[idx] = new_info
                        updated = True
                    break
            if not found:
                players.append({
                    "player_id": player.id,
                    "name": clean_name,
                    "tg_id": tg_id,
                })
                updated = True
            if updated:
                state["players"] = players
                await dbs.execute(
                    update(Session).where(Session.id == session_id).values(state=state)
                )
                await dbs.commit()

    return existing is None

# ---------------- Start world flow ----------------

def _format_generation_error(stage: str, exc: BaseException) -> str:
    """Return a human-friendly message for generation failures."""
    if isinstance(exc, asyncio.TimeoutError):
        return f"⏳ {stage}: время ожидания ответа модели истекло. Попробуйте позже."
    return f"❌ {stage}: произошла ошибка. Попробуйте позже."


async def _handle_generation_error(
    *,
    stage: str,
    exc: BaseException,
    chat_id: int,
    status_msg: Optional[Message],
) -> Message:
    if isinstance(exc, asyncio.TimeoutError):
        logger.warning("STARTWORLD: %s timed out: %r", stage, exc)
    else:
        logger.exception("STARTWORLD: %s failed", stage)
    return await _send_or_edit_status(
        chat_id=chat_id,
        status_msg=status_msg,
        text=_format_generation_error(stage, exc),
    )


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

    world_state_result = generate_world_v2(
        group_chat_id=group_chat_id,
        title=title,
        players=players_payload,
        story_theme=story_theme,
        timeout=None,
    )
    try:
        world_state: Dict[str, Any] = await world_state_result if inspect.isawaitable(world_state_result) else world_state_result
    except Exception as exc:
        status_msg = await _handle_generation_error(
            stage="Генерация мира",
            exc=exc,
            chat_id=group_chat_id,
            status_msg=status_msg,
        )
        return
    if story_theme and not world_state.get("story_theme"):
        world_state["story_theme"] = story_theme

    async with AsyncSessionLocal() as dbs:
        await dbs.execute(update(Session).where(Session.id == session_id).values(state=world_state))
        await dbs.commit()
    logger.info("STARTWORLD: world generated")

    status_msg = await _send_or_edit_status(chat_id=group_chat_id, status_msg=status_msg, text="Распределяю роли…")

    roles_result = generate_roles_for_players(
        state=world_state,
        players=players_payload,
        story_theme=story_theme,
        timeout=None,
    )
    try:
        roles = await roles_result if inspect.isawaitable(roles_result) else roles_result
    except Exception as exc:
        status_msg = await _handle_generation_error(
            stage="Распределение ролей",
            exc=exc,
            chat_id=group_chat_id,
            status_msg=status_msg,
        )
        return
    roles_by_pid: Dict[str, Dict[str, str]] = {r["player_id"]: r for r in roles}

    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        st = dict(s.state or {}) if s else {}
        st_players = st.get("players") or []
        by_id = {str(p.get("player_id")): p for p in st_players if isinstance(p, dict)}

        for pid, tg, name in joined:
            slot = by_id.get(pid) or {"player_id": pid}
            # Важно: сохраняем tg_id игрока в state, иначе воркер не распознаёт его ходы.
            slot["tg_id"] = tg
            slot["name"] = name
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

    status_msg = await _send_or_edit_status(chat_id=group_chat_id, status_msg=status_msg, text="Готовлю личные прологи…")

    sent_dm = 0
    for pid, tg, name in joined:
        pl_dict = {"player_id": pid, "name": name}
        story_result = generate_initial_backstory(
            state=world_state,
            player=pl_dict,
            story_theme=story_theme,
            timeout=None,
        )
        try:
            story = await story_result if inspect.isawaitable(story_result) else story_result
        except Exception as exc:
            status_msg = await _handle_generation_error(
                stage=f"Пролог для {name}",
                exc=exc,
                chat_id=group_chat_id,
                status_msg=status_msg,
            )
            try:
                await bot.send_message(
                    group_chat_id,
                    _format_generation_error(stage=f"Пролог для {escape(name)}", exc=exc),
                )
            except Exception:
                pass
            continue
        role_line = roles_by_pid.get(pid, {}).get("role") or ""
        dm_text = f"<b>Ваша роль:</b> {escape(role_line)}\n<b>Пролог</b>\n{escape(story.get('text') or '')}"
        try:
            # длинные прологи — чанковано
            for part in chunk_text(dm_text, hard_limit=4000):
                await bot.send_message(tg, part)
            sent_dm += 1
        except Exception as e:
            logger.warning("DM to %s failed: %r", tg, e)

        async with AsyncSessionLocal() as dbs:
            s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
            if s:
                st = dict(s.state or {})
                priv = list(st.get("private_history") or [])
                priv.append({"player_id": pid, "text": story.get('text') or "", "echo_of_action": story.get('echo_of_action') or ""})
                st["private_history"] = priv[-20:]
                await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
                await dbs.commit()

    logger.info("STARTWORLD: preludes sent dm=%d/%d", sent_dm, len(joined))

    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        if s:
            st = dict(s.state or {})
            st["phase"] = "running"
            await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
            await dbs.commit()
    try:
        await redis.sadd("sessions:active", session_id)
        logger.info("STARTWORLD: session %s activated in Redis", session_id)
    except Exception as e:
        logger.warning("STARTWORLD: failed to activate session in Redis: %r", e)

    await _delete_status_silent(status_msg)

    players_block = "\n".join([f"• {escape(name)} — {escape(roles_by_pid.get(pid, {}).get('role') or '')}" for pid, _tg, name in joined])
    setting = escape(world_state.get("setting") or "")
    location = escape(world_state.get("location") or "")
    theme_line = escape(world_state.get("story_theme") or "") if world_state.get("story_theme") else "не задана"
    opening_hook = escape(world_state.get("opening_hook") or "")

    final_text = (
        f"<b>Сеттинг:</b> {setting}\n"
        f"<b>Стартовая локация:</b> {location}\n"
        f"<b>Тема:</b> {theme_line}" +
        (f"\n<b>Начальная ситуация:</b> {opening_hook}" if opening_hook else "") +
        f"\n\nРоли и персональные прологи разосланы в личные сообщения ({sent_dm}/{len(joined)}).\n\n"
        f"<b>Участники:</b>\n{players_block}"
    )
    await _send_chunked(group_chat_id, final_text)
    logger.info("STARTWORLD: final world post sent")

    # Подсказка игрокам о начале игры
    try:
        await bot.send_message(group_chat_id, "Игра началась! Теперь отправляйте свои действия командой /act мне в личные сообщения.")
    except Exception as e:
        logger.warning("Failed to send start reminder: %r", e)

# ---- THEME in group flow ----

def _await_theme_key(chat_id: int, user_id: int) -> str:
    return f"await_theme:{chat_id}:{user_id}"

@router.callback_query(F.data.startswith("settheme:"))
async def cb_settheme(call: CallbackQuery):
    """Ждём СЛЕДУЮЩЕЕ сообщение от нажавшего пользователя и используем его как тему.
    Важно: НЕ перерисовываем лобби здесь, чтобы не плодить дубликаты.
    """
    session_id = call.data.split(":", 1)[1]
    await redis.setex(_await_theme_key(call.message.chat.id, call.from_user.id), 600, session_id)
    help_text = (
        "Напишите тему одним сообщением (например: <i>больной боб</i> или <i>случай в Санкт-Петербурге</i>).\n"
        "Будет учтена тема от нажавшего кнопку игрока. (Тема опциональна.)"
    )
    try:
        await call.message.reply(help_text)
        await call.answer("Жду тему здесь, в чате")
    except Exception:
        pass
    logger.info("SETTHEME: sid=%s user=%s chat=%s", session_id, call.from_user.id, call.message.chat.id)

@router.callback_query(F.data.startswith("cleartheme:"))
async def cb_cleartheme(call: CallbackQuery):
    session_id = call.data.split(":", 1)[1]
    await _clear_theme(session_id)
    try:
        await call.answer("Тема сброшена")
    except Exception:
        pass
    try:
        await _post_or_update_lobby_panel(call.message.chat.id, session_id)
    except Exception:
        pass
    logger.info("CLEARTHEME: sid=%s", session_id)

# ---- Group message handler to capture theme text ----
# ВАЖНО: ограничиваем ТОЛЬКО групповыми чатами, иначе этот хендлер съедает приватные апдейты и /act не доходит.
@router.message(F.chat.type.in_({"group", "supergroup"}), ~F.text.startswith("/"))
async def on_group_message(message: Message):
    text = (message.text or "").strip()
    if not text:
        return
    pending_sid = await redis.get(_await_theme_key(message.chat.id, message.from_user.id))
    if not pending_sid:
        return
    session_id = pending_sid.decode("utf-8")
    await _set_theme(session_id, text)
    await redis.delete(_await_theme_key(message.chat.id, message.from_user.id))
    await message.reply(f"Тема установлена: <b>{escape(text)}</b>")
    try:
        await _post_or_update_lobby_panel(message.chat.id, session_id)
    except Exception:
        pass
    logger.info("THEME SET: sid=%s by user=%s theme=%s", session_id, message.from_user.id, text)

# ---------------- DM: /act ----------------

@router.message(Command("act"))
async def cmd_act(message: Message):
    if message.chat.type != "private":
        await message.answer("Эту команду нужно отправлять в личку боту.")
        return

    raw = message.text or ""
    parts = raw.split(maxsplit=1)
    act_text = parts[1].strip() if len(parts) > 1 else ""
    if not act_text:
        await message.answer("Пустой ход. Напишите текст после /act.")
        return

    session_id = await _find_active_session_for_user(message.from_user.id)
    if not session_id:
        await message.answer("Нет активной сессии. Зайдите в группу, присоединитесь и дождитесь запуска игры.")
        logger.info("ACT: no active session for tg=%s", message.from_user.id)
        return

    payload = {"player_tg": message.from_user.id, "player_name": message.from_user.full_name, "text": act_text}
    try:
        await redis.rpush(f"session:{session_id}:actions", json.dumps(payload, ensure_ascii=False))
        logger.info("ACT: accepted for sid=%s tg=%s text=%s", session_id, message.from_user.id, act_text)
        await message.answer("Принято.")
    except Exception as e:
        logger.warning("ACT: redis push failed sid=%s err=%r", session_id, e)
        await message.answer("Не удалось отправить ход, попробуйте ещё раз.")

# ---------------- Commands ----------------

@router.message(Command("newgame"))
async def cmd_newgame(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        await message.answer("Команду нужно вызывать в групповом чате.")
        return

    session_id = str(message.chat.id)
    title = message.chat.title or "Игра"

    init_state = {
        "turn": 0,
        "setting": "",
        "location": "",
        "world_flags": {},
        "npcs": [],
        "available_items": [],
        "opening_hook": "",
        "visibility_defaults": {},
        "style": {},
        "title": title,
        "phase": "init",
        "join_locked": False,
        "players": [],
        "effects_log": [],
        "raw_history": [],
        "private_history": [],
        "general_history": [],
        "story_theme": "",
    }

    await create_or_reset_session(session_id=session_id, group_chat_id=message.chat.id, title=title, initial_state=init_state)
    try:
        await redis.srem("sessions:active", session_id)
    except Exception:
        pass

    await _join_session(session_id=session_id, tg_id=message.from_user.id, name=message.from_user.full_name)
    msg = await _post_or_update_lobby_panel(message.chat.id, session_id)
    await _save_lobby_panel(session_id, msg.message_id)
    logger.info("NEWGAME: session=%s reset", session_id)

@router.message(Command("endgame"))
async def cmd_endgame(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        await message.answer("Команду нужно вызывать в групповом чате.")
        return
    session_id = str(message.chat.id)
    async with AsyncSessionLocal() as dbs:
        s: Session | None = (await dbs.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        if s is None:
            await message.answer("Сессия не найдена.")
            return
        st = dict(s.state or {})
        st["phase"] = "ended"
        st["join_locked"] = True
        await dbs.execute(update(Session).where(Session.id == session_id).values(state=st))
        await dbs.commit()
    try:
        await redis.srem("sessions:active", session_id)
    except Exception:
        pass
    await _cleanup_session_logs(session_id)
    await message.answer("Игра завершена. /newgame — чтобы начать новую.")
    logger.info("ENDGAME: session=%s ended", session_id)

@router.message(Command("startworld"))
async def cmd_startworld(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        await message.answer("Команду нужно вызвать в групповом чате.")
        return
    session_id = str(message.chat.id)
    msg = await _post_or_update_lobby_panel(message.chat.id, session_id)
    await _save_lobby_panel(session_id, msg.message_id)
    logger.info("STARTWORLD CMD: chat=%s", session_id)

@router.message(Command("join"))
async def cmd_join(message: Message):
    if message.chat.type not in ("group", "supergroup"):
        await message.answer("Жмите /start в ЛС и используйте /act для хода.")
        return
    added = await _join_session(session_id=str(message.chat.id), tg_id=message.from_user.id, name=message.from_user.full_name)
    note = "✅ Вы в лобби." if added else "Вы уже в лобби."
    await message.answer(note)
    await _post_or_update_lobby_panel(message.chat.id, str(message.chat.id))
    logger.info("JOIN CMD: chat=%s user=%s added=%s", message.chat.id, message.from_user.id, added)

# ---------------- Callbacks ----------------

@router.callback_query(F.data.startswith("join:"))
async def cb_join(call: CallbackQuery):
    session_id = call.data.split(":", 1)[1]
    added = await _join_session(session_id=session_id, tg_id=call.from_user.id, name=call.from_user.full_name)
    try:
        await call.answer("✅ Присоединился" if added else "Вы уже в лобби")
    except Exception:
        pass
    try:
        msg = await _post_or_update_lobby_panel(call.message.chat.id, session_id)
        await _save_lobby_panel(session_id, msg.message_id)
    except Exception:
        pass
    logger.info("JOIN CB: chat=%s user=%s added=%s", call.message.chat.id, call.from_user.id, added)

@router.callback_query(F.data.startswith("roster:"))
async def cb_roster(call: CallbackQuery):
    session_id = call.data.split(":", 1)[1]
    players = await _players_joined_list(session_id)
    if not players:
        txt = "Никто не присоединился."
    else:
        lines = [f"• {escape(name)}" for (_pid, _tg, name) in players]
        txt = "<b>Участники:</b>\n" + "\n".join(lines)
    await call.message.answer(txt)
    try:
        await call.answer()
    except Exception:
        pass
    logger.info("ROSTER: session=%s count=%d", session_id, len(players))

@router.callback_query(F.data.startswith("startworld:"))
async def cb_startworld(call: CallbackQuery):
    session_id = call.data.split(":", 1)[1]
    try:
        await call.answer()
    except Exception:
        pass
    await _start_world_flow(session_id=session_id, group_chat_id=call.message.chat.id, reply_target=call)

# ---------------- Справка/Правила ----------------

@router.message(Command("rules"))
async def cmd_rules(message: Message):
    help_text = (
        "📜 <b>Правила игры и управление:</b>\n"
        "• После начала игры отправляйте свои действия ботu <b>в личные сообщения</b> командой <code>/act</code> и описанием действия. Например: <code>/act осматриваюсь вокруг</code>.\n"
        "• Ваши действия обрабатываются и влияют на общую историю, разворачивающуюся в групповом чате. Вы будете получать личное описание последствий своих действий, а в группу бот публикует общее продолжение истории.\n"
        "• Команда /newgame в группе создаёт новую сессию игры, /join — присоединяет к лобби, и по нажатию «Начать игру» запускается сюжет.\n"
        "• Тему игры можно задать через кнопку «📝 Задать тему» перед стартом игры.\n"
        "• Игра является текстовой RPG с нарративом, старайтесь описывать действия персонажа понятно и по ситуации. Успех и неудача действий определяются системой (частично случайно, частично по логике)."
    )
    await message.answer(help_text)

@router.message(Command("help"))
async def cmd_help(message: Message):
    # Alias to /rules
    await cmd_rules(message)

# ---------------- Команды бота ----------------


async def _register_commands() -> None:
    group_commands = [
        BotCommand(command="newgame", description="Создать новое лобби"),
        BotCommand(command="join", description="Присоединиться к игре"),
        BotCommand(command="startworld", description="Начать игру"),
        BotCommand(command="endgame", description="Завершить игру"),
        BotCommand(command="rules", description="Правила и помощь"),
    ]
    private_commands = [
        BotCommand(command="act", description="Отправить действие персонажа"),
        BotCommand(command="rules", description="Правила и помощь"),
        BotCommand(command="help", description="Краткая справка"),
    ]

    try:
        await bot.set_my_commands(group_commands, scope=BotCommandScopeAllGroupChats())
        await bot.set_my_commands(private_commands, scope=BotCommandScopeAllPrivateChats())
    except TelegramBadRequest as exc:  # pragma: no cover - зависит от API Telegram
        logger.warning("BOT: failed to register commands: %s", exc)

# ---------------- Entry ----------------

async def main():
    await init_db()
    try:
        await bot.set_my_short_description("Сторителлер для кооперативной текстовой RPG.")
    except Exception:
        pass
    await _register_commands()
    logger.info("BOT: started")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
