import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.db import (
    get_async_sessionmaker,
    get_redis,
    get_session,
    load_session_state,
    save_session_state,
)
from core.turn_runner import (
    TurnIOCallbacks,
    PlayerAction,
    run_turn_loop,
)
from core.schemas import TurnConfig

# -------------------- Логирование --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = True

# -------------------- Конфиг по умолчанию --------------------
DEFAULT_TURN_CFG = TurnConfig(
    group_chat_id=0,  # будет переопределён реальным chat_id при запуске сессии
    timeout=60,       # время ожидания хода
)

# -------------------- Redis-ключи и утилиты --------------------

def _redis_session_actions_key(session_id: int) -> str:
    return f"session:{session_id}:actions"  # список входящих действий (json в каждой записи)

def _redis_session_active_key() -> str:
    return "sessions:active"  # SET с id сессий, для которых надо крутить луп

def _redis_session_control_key(session_id: int) -> str:
    return f"session:{session_id}:control"  # каналы управления (например, cancel)

# -------------------- Приём действий игроков --------------------

async def _await_actions_for_session(
    session_id: int,
    expect_actions_hint: int,
    timeout_sec: float,
    poll_sec: float = 0.5,
) -> List[PlayerAction]:
    """
    Забираем с Redis-очереди все действия, пришедшие в «окно» хода.
    Логика:
      - читаем состояние, определяем состав игроков (state["players"])
      - собираем N (кол-во) действий *по одному от каждого участника*;
      - окно закрывается либо при получении всех, либо по таймауту.
    """
    redis = await get_redis()
    state = await load_session_state(session_id)
    players: List[Dict[str, Any]] = state.get("players", [])
    need_tg_ids = [p.get("tg_id") for p in players if "tg_id" in p]
    need_tg_ids = [x for x in need_tg_ids if x is not None]

    logger.info(
        "Turn[%s]: load_state keys=%s",
        session_id,
        list(state.keys())[:18],
    )

    if not need_tg_ids:
        # если нет игроков — сразу пустой список
        return []

    # Мапа tg_id -> PlayerAction (берем по одному действию от игрока)
    collected: Dict[int, PlayerAction] = {}

    started = time.time()
    while True:
        # забираем все сообщения, которые накопились
        while True:
            raw = await redis.lpop(_redis_session_actions_key(session_id))
            if raw is None:
                break
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            # проверяем, что это ход от участника окна
            tg = obj.get("player_tg")
            txt = (obj.get("text") or "").strip()
            if tg in need_tg_ids and txt:
                collected[tg] = PlayerAction(
                    player_id=obj.get("player_id"),
                    player_tg=tg,
                    player_name=obj.get("player_name") or str(tg),
                    text=txt,
                )

        # если собрали по одному от каждого — закрываем окно
        if len(collected) >= len(need_tg_ids):
            break

        if (time.time() - started) >= timeout_sec:
            break

        await asyncio.sleep(poll_sec)

    # возвращаем в фиксированном порядке: по списку участников
    result: List[PlayerAction] = []
    for tg in need_tg_ids:
        if tg in collected:
            result.append(collected[tg])

    logger.info(
        "Turn[%s]: awaiting actions (NO TIMEOUT), expect=%d players=%d",
        session_id, len(need_tg_ids), len(need_tg_ids),
    )
    return result

# -------------------- Взаимодействие с БД --------------------

async def _load_state(session_id: int) -> Dict[str, Any]:
    return await load_session_state(session_id)

async def _save_state(session_id: int, state: Dict[str, Any]) -> None:
    await save_session_state(session_id, state)

# -------------------- Паблиш в Telegram (через бота) --------------------

# Здесь только интерфейсы; реализация у бота подписана на Redis/БД и подхватывает события.
# Для простоты — пишем в лог (в реальном коде — складируем в БД/Redis, либо дергаем вебхук бота).

async def _notify_generation_started(session_id: int, actual: int = 0, expected: int = 0) -> None:
    logger.info("Turn[%s]: collecting actions, expect_hint=%d timeout=%ss", session_id, expected, int(DEFAULT_TURN_CFG.timeout))

async def _notify_effects(session_id: int, effects_obj: Any) -> None:
    try:
        # для дебага: складываем «валидные» эффекты рядом с логами сессии
        sid = str(session_id)
        turn_dir = os.path.join(".", "logs", f"session_{sid}")
        os.makedirs(turn_dir, exist_ok=True)
        with open(os.path.join(turn_dir, "last_effects.json"), "w", encoding="utf-8") as f:
            if hasattr(effects_obj, "model_dump"):
                json.dump(effects_obj.model_dump(exclude_none=True), f, ensure_ascii=False, indent=2)  # type: ignore
            else:
                json.dump(effects_obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    logger.info("Turn[%s]: effects applied", session_id)

async def _notify_raw(session_id: int, raw_obj: Any) -> None:
    text = raw_obj.text if hasattr(raw_obj, "text") else str(raw_obj)
    logger.info("Turn[%s]: raw story ready | raw(%d): %s", session_id, len(text), (text[:155] + "...") if len(text) > 155 else text)

async def _notify_private(session_id: int, player_id: str, story_obj: Any) -> None:
    text = story_obj.text if hasattr(story_obj, "text") else str(story_obj)
    echo = getattr(story_obj, "echo_of_action", None)
    logger.info(
        "Turn[%s]: private for pid=%s | text(%d): %s | echo=%s",
        session_id, player_id, len(text),
        (text[:120] + "...") if len(text) > 120 else text,
        f"echo({len(echo)}): {echo}" if echo else "—",
    )

async def _notify_general(session_id: int, story_obj: Any) -> None:
    text = story_obj.text if hasattr(story_obj, "text") else str(story_obj)
    logger.info("Turn[%s]: general | text(%d): %s", session_id, len(text), (text[:155] + "...") if len(text) > 155 else text)

async def _notify_telemetry(session_id: int, data: Dict[str, Any]) -> None:
    logger.info("Turn[%s]: telemetry json(%d): %s", session_id, len(json.dumps(data, ensure_ascii=False)), json.dumps(data, ensure_ascii=False))

# -------------------- Коллбеки для раннера --------------------

def make_callbacks_for_session(session_id: int) -> TurnIOCallbacks:
    return TurnIOCallbacks(
        load_state=lambda: _load_state(session_id),
        save_state=lambda st: _save_state(session_id, st),
        await_actions=lambda exp, timeout, poll: _await_actions_for_session(session_id, exp, timeout, poll),
        notify_generation_started=lambda actual=0, expected=0: _notify_generation_started(session_id, actual, expected),
        notify_effects=lambda eff: _notify_effects(session_id, eff),
        notify_raw_story=lambda raw: _notify_raw(session_id, raw),
        notify_private_story=lambda pid, st: _notify_private(session_id, pid, st),
        notify_general_story=lambda st: _notify_general(session_id, st),
        telemetry=lambda data: _notify_telemetry(session_id, data),
    )

# -------------------- Управляющая логика воркера --------------------

_running_tasks: Dict[int, asyncio.Task] = {}  # session_id -> task
_monitor_task: Optional[asyncio.Task] = None

async def _start_turn_loop_for_session(session_id: int, cfg: TurnConfig = DEFAULT_TURN_CFG) -> None:
    """
    Создаёт и запускает бесконечный цикл ходов для одной сессии.
    """
    # выясним, кого ждать — по состоянию
    state = await load_session_state(session_id)
    players = state.get("players", [])
    expect_actions = max(1, len(players))

    # Переопределим конфиг для конкретной сессии
    session_cfg = cfg.model_copy()
    session_group_chat_id: Optional[int] = None

    # 1) Попробуем взять chat_id из JSON-состояния (мог быть сохранён ботом)
    try:
        maybe_from_state = state.get("group_chat_id") or (state.get("session") or {}).get("group_chat_id")
        if maybe_from_state is not None:
            session_group_chat_id = int(maybe_from_state)
    except Exception:
        session_group_chat_id = None

    # 2) Если в state его нет — загрузим саму сессию из БД
    if session_group_chat_id is None:
        session_row = await get_session(str(session_id))
        if session_row and session_row.group_chat_id is not None:
            session_group_chat_id = int(session_row.group_chat_id)

    if session_group_chat_id is not None:
        session_cfg = session_cfg.model_copy(update={"group_chat_id": session_group_chat_id})
    else:
        logger.warning("Worker: session %s has no group_chat_id, using default=%s", session_id, session_cfg.group_chat_id)

    logger.info(
        "Worker: start run_turn_loop sid=%s expect_actions=%s joined=%s",
        session_id, expect_actions,
        [(p.get("id"), p.get("tg_id")) for p in players],
    )

    callbacks = make_callbacks_for_session(session_id)
    try:
        await run_turn_loop(
            session_id=session_id,
            callbacks=callbacks,
            cfg=session_cfg,
            expect_actions=expect_actions,
            idle_sleep=0.5,
        )
    except asyncio.CancelledError:
        pass

async def _monitor_active_sessions() -> None:
    """
    Следит за Redis SET sessions:active.
    На каждую сессию поднимает отдельный таск с бесконечным циклом ходов.
    """
    redis = await get_redis()
    logger.info("Worker: bootstrap, watching Redis set %s", _redis_session_active_key())

    # начальные
    known: set[int] = set()
    while True:
        try:
            ids_raw = await redis.smembers(_redis_session_active_key())
            ids = set()
            for b in ids_raw:
                try:
                    ids.add(int(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b))
                except Exception:
                    continue

            # новые -> запустить
            for sid in ids - known:
                logger.info("Worker: spawned loop for sid=%s", sid)
                # проверим, что сессия вообще существует в БД
                st = await load_session_state(sid)
                if not st:
                    logger.warning("Worker: session=%s not found", sid)
                    continue

                if sid in _running_tasks and not _running_tasks[sid].done():
                    continue

                _running_tasks[sid] = asyncio.create_task(_start_turn_loop_for_session(sid))

            # удаленные -> отменить
            for sid in list(known - ids):
                t = _running_tasks.get(sid)
                if t and not t.done():
                    t.cancel()
                _running_tasks.pop(sid, None)

            known = ids
            await asyncio.sleep(3.0)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Worker: monitor loop failed; sleep")
            await asyncio.sleep(2.0)

# -------------------- main --------------------

def _handle_sigterm():
    global _monitor_task

    if _monitor_task and not _monitor_task.done():
        try:
            _monitor_task.cancel()
        except Exception:
            pass

    for t in list(_running_tasks.values()):
        try:
            t.cancel()
        except Exception:
            pass

async def _amain() -> None:
    # прогреем соединения
    _ = await get_async_sessionmaker()
    _ = await get_redis()

    global _monitor_task
    _monitor_task = asyncio.create_task(_monitor_active_sessions())

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigterm)
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    except NotImplementedError:
        # windows etc.
        pass

    try:
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass
    finally:
        for t in list(_running_tasks.values()):
            try:
                t.cancel()
            except Exception:
                pass

        if _monitor_task and not _monitor_task.done():
            try:
                _monitor_task.cancel()
            except Exception:
                pass

        _monitor_task = None

def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    # точка входа воркера
    from core.db import init_db  # опционально, если нужно мигрировать
    logger.info("DB: init ok")
    main()
