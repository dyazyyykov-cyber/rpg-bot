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
    timeout=60.0,   # время ожидания хода
    seed=None,      # фиксированный сид генерации (для отладки) или None
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
    timeout: float,
) -> List[PlayerAction]:
    """
    Ждёт, пока в Redis list накопится хотя бы expect_actions_hint действий,
    либо истечёт timeout.
    Возвращает список PlayerAction (json→pydantic).
    """
    redis = await get_redis()
    k = _redis_session_actions_key(session_id)

    start = time.monotonic()
    collected = []
    while (time.monotonic() - start) < timeout:
        raw = await redis.lpop(k)
        if raw:
            try:
                data = json.loads(raw)
                collected.append(PlayerAction(**data))
            except Exception as e:
                logger.warning(f"Turn[{session_id}]: invalid action json={raw}, {e}")
                continue

        if len(collected) >= expect_actions_hint:
            break

        # если не набрали, немного спим
        if not raw:
            await asyncio.sleep(0.1)

    return collected

# -------------------- Эмиттеры событий --------------------
@dataclass
class WorkerCallbacks(TurnIOCallbacks):
    session_id: int

    async def notify_started(self) -> None:
        logger.info("Turn[%s]: turn started", self.session_id)

    async def notify_generation_started(self, expected: int = 0) -> None:
        logger.info(
            "Turn[%s]: collecting actions, expect_hint=%d timeout=%ss",
            self.session_id,
            expected,
            int(DEFAULT_TURN_CFG.timeout)
        )

    async def notify_generation_ended(self) -> None:
        logger.info("Turn[%s]: generation ended", self.session_id)

    async def notify_ended(self) -> None:
        logger.info("Turn[%s]: turn ended", self.session_id)

    async def load_state(self) -> Dict[str, Any]:
        session_maker = await get_async_sessionmaker()
        async with session_maker() as db:
            state = await load_session_state(db, self.session_id)
        logger.debug("Turn[%s]: load_state keys=%s", self.session_id, list(state.keys()))
        return state

    async def save_state(self, state: Dict[str, Any]) -> None:
        logger.debug("Turn[%s]: save_state keys=%s", self.session_id, list(state.keys()))
        session_maker = await get_async_sessionmaker()
        async with session_maker() as db:
            await save_session_state(db, self.session_id, state)
            await db.commit()

    async def load_pending_actions(self, expected: int) -> List[PlayerAction]:
        logger.debug("Turn[%s]: load_pending_actions expected=%d", self.session_id, expected)
        actions = await _await_actions_for_session(
            self.session_id,
            expect_actions_hint=expected,
            timeout=DEFAULT_TURN_CFG.timeout,
        )
        logger.info("Turn[%s]: loaded %d actions", self.session_id, len(actions))
        return actions

    async def emit_response(self, text: str, context: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Turn[%s]: emit_response len=%d", self.session_id, len(text))
        # TODO: сохранить ответ в БД, или отправить через websocket, и т.д.

# -------------------- Основной рабочий цикл --------------------
_running_tasks: Dict[int, asyncio.Task] = {}

async def _run_session_loop(session_id: int) -> None:
    """
    Бесконечный луп для одной сессии:
      1. Загружаем состояние
      2. Ждём действия игроков
      3. Генерируем ответ (run_turn_loop)
      4. Сохраняем состояние
      5. Повторяем

    Цикл останавливается, если:
      - в Redis появился control signal (cancel)
      - возникло исключение
    """
    logger.info(f"Worker: start session loop {session_id}")
    callbacks = WorkerCallbacks(session_id=session_id)

    try:
        await run_turn_loop(
            callbacks=callbacks,
            turn_cfg=DEFAULT_TURN_CFG,
        )
    except asyncio.CancelledError:
        logger.info(f"Worker: session loop {session_id} cancelled")
        raise
    except Exception:
        logger.exception(f"Worker: session loop {session_id} failed")
    finally:
        logger.info(f"Worker: stop session loop {session_id}")

async def _monitor_active_sessions() -> None:
    """
    Мониторит Redis SET `sessions:active`.
    Для каждой новой сессии запускает _run_session_loop().
    Для исчезнувшей сессии — отменяет таску.
    """
    logger.info("Worker: monitor active sessions started")
    known: set = set()

    while True:
        try:
            redis = await get_redis()
            k = _redis_session_active_key()
            ids_raw = await redis.smembers(k)
            ids = {int(x) for x in ids_raw}

            # новые сессии
            for sid in (ids - known):
                logger.info(f"Worker: discovered new session {sid}")
                task = asyncio.create_task(_run_session_loop(sid))
                _running_tasks[sid] = task

            # исчезнувшие сессии
            for sid in (known - ids):
                logger.info(f"Worker: session {sid} no longer active, cancel task")
                t = _running_tasks.get(sid)
                if t:
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
    for t in list(_running_tasks.values()):
        try:
            t.cancel()
        except Exception:
            pass

async def _amain() -> None:
    # прогреем соединения
    _ = await get_async_sessionmaker()
    _ = await get_redis()

    task = asyncio.create_task(_monitor_active_sessions())
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigterm)
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    except NotImplementedError:
        # windows etc.
        pass

    try:
        await task
    finally:
        for t in list(_running_tasks.values()):
            try:
                t.cancel()
            except Exception:
                pass

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
