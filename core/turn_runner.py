from __future__ import annotations

import os
import time
import json
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, Awaitable

from .utils import ensure_dir
from .engine import run_turn_with_llm
from .schemas import (
    TurnConfig,
    TurnOutputs,
    EffectsDelta,
    RawStory,
    PlayerStory,
    GeneralStory,
)

logger = logging.getLogger(__name__)
logger.propagate = True


# --------- Внешние типы для взаимодействия (воркер/БД/бот) ---------

@dataclass
class PlayerAction:
    player_id: Optional[str]   # внутриигровой id игрока, если уже известен
    player_tg: int             # TG user id
    player_name: str           # видимое имя
    text: str                  # текст действия как есть


@dataclass
class TurnIOCallbacks:
    """
    Коллбеки воркера. Реализация — в worker.py.

    Важно:
    - await_actions ждёт «всех» из стартового окна (формируемого по state["players"]).
    - Возвращает 0..N действий (N == размер окна). Если 0 — генерация не запускается.
    """
    # загрузка/сохранение состояния
    load_state: Callable[[], Awaitable[Dict[str, Any]]]
    save_state: Callable[[Dict[str, Any]], Awaitable[None]]

    # ожидание действий игроков:
    await_actions: Callable[[int, float, float], Awaitable[List[PlayerAction]]]

    # уведомления (бот/БД)
    notify_generation_started: Optional[Callable[..., Awaitable[None]]] = None
    notify_effects: Callable[[EffectsDelta], Awaitable[None]] = lambda _: asyncio.sleep(0)
    notify_raw_story: Callable[[RawStory], Awaitable[None]] = lambda _: asyncio.sleep(0)
    notify_private_story: Callable[[str, PlayerStory], Awaitable[None]] = lambda _pid, _st: asyncio.sleep(0)
    notify_general_story: Callable[[GeneralStory], Awaitable[None]] = lambda _: asyncio.sleep(0)

    # опционально: телеметрия
    telemetry: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None


# -------------------- Утилиты логирования --------------------

def _write_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            if hasattr(obj, "model_dump"):
                json.dump(obj.model_dump(exclude_none=True), f, ensure_ascii=False, indent=2)  # type: ignore
            else:
                json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _write_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


# -------------------- Основной раннер одного хода --------------------

async def run_single_turn(
    session_id: int,
    callbacks: TurnIOCallbacks,
    cfg: TurnConfig,
    expect_actions: int,
    wait_timeout: float,
    poll_sec: float = 0.5,
) -> Optional[TurnOutputs]:
    """
    1) Загружает текущее состояние
    2) Ждёт действий игроков (await_actions) — там же фиксируется ростер окна.
    3) Если действий нет — пропускаем. Если >0 — запускаем LLM-пайплайн.
    4) Рассылаем результаты, сохраняем state, логируем.
    """
    sid = str(session_id)

    # --- директория логов хода
    state = await callbacks.load_state()
    turn_no = int(state.get("turn", 0))
    turn_dir = os.path.join(".", "logs", f"session_{sid}", f"turn_{turn_no:04d}")
    ensure_dir(turn_dir)

    # --- ожидание действий
    logger.info(
        "Turn[%s]: collecting actions, expect_hint=%d timeout=%ss",
        sid, expect_actions, int(wait_timeout)
    )
    actions: List[PlayerAction] = await callbacks.await_actions(expect_actions, wait_timeout, poll_sec)
    for a in actions:
        logger.info(
            "Turn[%s]: action from tg=%s (pid=%s) text_preview=%r",
            sid, a.player_tg, a.player_id or "?", a.text[:80]
        )

    _write_json(os.path.join(turn_dir, "actions.json"), [a.__dict__ for a in actions])

    # --- если никто не походил — скипаем; «ждать всех» гарантируется внутри await_actions
    if not actions:
        logger.info("Turn[%s]: no actions -> skip", sid)
        return None

    # --- уведомление: начинаем генерацию
    try:
        if callbacks.notify_generation_started:
            try:
                await callbacks.notify_generation_started(len(actions), expect_actions)
            except TypeError:
                await callbacks.notify_generation_started()
        _write_text(os.path.join(turn_dir, "generation_started.txt"),
                    f"started; actions={len(actions)}")
    except Exception:
        logger.exception("Turn[%s]: notify_generation_started failed", sid)

    # --- запускаем оркестровку хода (LLM)
    actions_payload = [
        {
            "player_id": a.player_id,
            "player_tg": a.player_tg,
            "player_name": a.player_name,
            "text": a.text,
        }
        for a in actions
    ]

    t0 = time.time()
    outputs: TurnOutputs = await run_turn_with_llm(
        state=state,
        actions=actions_payload,
        cfg=cfg,
    )
    dt = time.time() - t0
    logger.info("Turn[%s #%d]: LLM done in %.1fs", sid, turn_no, dt)

    # --- логируем и рассылаем

    # эффекты
    effects_obj = outputs.effects
    _write_json(os.path.join(turn_dir, "effects_valid.json"), effects_obj)
    try:
        if isinstance(effects_obj, dict):
            effects_obj = EffectsDelta.model_validate(effects_obj)
        await callbacks.notify_effects(effects_obj)  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("Turn[%s]: notify_effects failed: %r", sid, e)

    # сырая история
    raw_obj = outputs.raw
    _write_json(os.path.join(turn_dir, "raw_story.json"), raw_obj)
    try:
        if isinstance(raw_obj, dict):
            raw_obj = RawStory.model_validate(raw_obj)
        await callbacks.notify_raw_story(raw_obj)  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("Turn[%s]: notify_raw_story failed: %r", sid, e)

    # личные истории
    for pid, story_val in (outputs.players_private or {}).items():
        if isinstance(story_val, dict):
            try:
                story_obj = PlayerStory.model_validate(story_val)
            except Exception:
                story_obj = PlayerStory(**story_val)
        else:
            story_obj = story_val

        _write_json(
            os.path.join(turn_dir, f"private_story_{pid}.json"),
            story_obj
        )
        try:
            await callbacks.notify_private_story(pid, story_obj)
        except Exception as e:
            logger.exception("Turn[%s]: notify_private_story(%s) failed: %r", sid, pid, e)

    # общая история (GENERAL обязателен — fail-stop гарантируется внутри engine)
    general_obj = outputs.general
    _write_json(os.path.join(turn_dir, "general_story.json"), general_obj)

    if general_obj is None:
        logger.error("Turn[%s]: GENERAL missing (engine should never return None) -> abort turn", sid)
        raise RuntimeError("General story is required but missing")
    try:
        if isinstance(general_obj, dict):
            general_obj = GeneralStory.model_validate(general_obj)
        if not (general_obj.text or "").strip():
            logger.error("Turn[%s]: GENERAL empty text -> abort turn", sid)
            raise RuntimeError("General story text is empty")
        await callbacks.notify_general_story(general_obj)  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("Turn[%s]: notify_general_story failed: %r", sid, e)
        raise

    # --- сохраняем состояние (run_turn_with_llm уже мутировал state)
    await callbacks.save_state(state)
    logger.info("Turn[%s #%d]: state saved (turn now %s)", sid, turn_no, state.get("turn"))

    # --- телеметрия
    if callbacks.telemetry:
        try:
            await callbacks.telemetry({
                "session_id": session_id,
                "turn": state.get("turn"),
                "llm_time_sec": dt,
                "actions_cnt": len(actions),
                "private_cnt": len(outputs.players_private or {}),
                "general_present": True,
            })
        except Exception:
            logger.exception("Turn[%s]: telemetry failed", sid)

    return outputs


# -------------------- Цикл с таймером ожидания --------------------

async def run_turn_loop(
    session_id: int,
    callbacks: TurnIOCallbacks,
    cfg: TurnConfig,
    expect_actions: int,
    idle_sleep: float,
) -> None:
    """
    Бесконечный цикл ходов для одной сессии (воркер вызывает в таске).
    await_actions сам решает, кого ждать, исходя из state["players"].
    """
    sid = str(session_id)
    logger.info("TurnLoop[%s]: started", sid)
    while True:
        try:
            res = await run_single_turn(
                session_id=session_id,
                callbacks=callbacks,
                cfg=cfg,
                expect_actions=expect_actions,  # только хинт для логов
                wait_timeout=cfg.timeout,
                poll_sec=idle_sleep,
            )
            if res is None:
                await asyncio.sleep(idle_sleep)
        except asyncio.CancelledError:
            logger.info("TurnLoop[%s]: cancelled", sid)
            break
        except Exception:
            logger.exception("TurnLoop[%s]: iteration failed", sid)
            await asyncio.sleep(idle_sleep)
