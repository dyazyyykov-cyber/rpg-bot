from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .engine import run_turn_with_llm
from .schemas import GeneralStory, TurnConfig, TurnOutputs
from .utils import ensure_dir

logger = logging.getLogger(__name__)
logger.propagate = True


@dataclass
class PlayerAction:
    player_id: Optional[str]
    player_tg: int
    player_name: str
    text: str


@dataclass
class TurnIOCallbacks:
    load_state: Callable[[], Awaitable[Dict[str, Any]]]
    save_state: Callable[[Dict[str, Any]], Awaitable[None]]
    await_actions: Callable[[int, float, float], Awaitable[List[PlayerAction]]]
    notify_generation_started: Optional[Callable[..., Awaitable[None]]] = None
    notify_general_story: Callable[[GeneralStory], Awaitable[None]] = lambda _: asyncio.sleep(0)
    telemetry: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None


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


async def run_single_turn(
    session_id: int,
    callbacks: TurnIOCallbacks,
    cfg: TurnConfig,
    expect_actions: int,
    wait_timeout: float,
    poll_sec: float = 0.5,
) -> Optional[TurnOutputs]:
    sid = str(session_id)

    state = await callbacks.load_state()
    turn_no = int(state.get("turn", 0))
    turn_dir = os.path.join(".", "logs", f"session_{sid}", f"turn_{turn_no:04d}")
    ensure_dir(turn_dir)

    logger.info(
        "Turn[%s]: collecting actions, expect_hint=%d timeout=%ss",
        sid,
        expect_actions,
        int(wait_timeout),
    )
    actions: List[PlayerAction] = await callbacks.await_actions(expect_actions, wait_timeout, poll_sec)
    for action in actions:
        logger.info(
            "Turn[%s]: action from tg=%s (pid=%s) text_preview=%r",
            sid,
            action.player_tg,
            action.player_id or "?",
            action.text[:80],
        )

    _write_json(os.path.join(turn_dir, "actions.json"), [a.__dict__ for a in actions])

    if not actions:
        logger.info("Turn[%s]: no actions -> skip", sid)
        return None

    try:
        if callbacks.notify_generation_started:
            try:
                await callbacks.notify_generation_started(len(actions), expect_actions)
            except TypeError:
                await callbacks.notify_generation_started()
            _write_text(os.path.join(turn_dir, "generation_started.txt"), f"started; actions={len(actions)}")
    except Exception:
        logger.exception("Turn[%s]: notify_generation_started failed", sid)

    t0 = time.time()
    outputs = await run_turn_with_llm(
        state=state,
        actions=[
            {
                "player_id": a.player_id,
                "player_tg": a.player_tg,
                "player_name": a.player_name,
                "text": a.text,
            }
            for a in actions
        ],
        cfg=cfg,
    )
    dt = time.time() - t0
    logger.info("Turn[%s #%d]: LLM done in %.1fs", sid, turn_no, dt)

    general_obj = outputs.general
    _write_json(os.path.join(turn_dir, "general_story.json"), general_obj)

    try:
        if isinstance(general_obj, dict):
            general_obj = GeneralStory.model_validate(general_obj)
        if not (general_obj.text or "").strip():
            raise RuntimeError("General story text is empty")
        await callbacks.notify_general_story(general_obj)  # type: ignore[arg-type]
    except Exception:
        logger.exception("Turn[%s]: notify_general_story failed", sid)
        raise

    await callbacks.save_state(state)
    logger.info("Turn[%s #%d]: state saved (turn now %s)", sid, turn_no, state.get("turn"))

    if callbacks.telemetry:
        payload = {
            "session_id": session_id,
            "turn": state.get("turn"),
            "llm_time_sec": dt,
            "actions_cnt": len(actions),
            "general_present": True,
        }
        payload.update(outputs.telemetry or {})
        try:
            await callbacks.telemetry(payload)
        except Exception:
            logger.exception("Turn[%s]: telemetry failed", sid)

    return outputs


async def run_turn_loop(
    session_id: int,
    callbacks: TurnIOCallbacks,
    cfg: TurnConfig,
    expect_actions: int,
    idle_sleep: float,
) -> None:
    sid = str(session_id)
    logger.info("TurnLoop[%s]: started", sid)
    while True:
        try:
            result = await run_single_turn(
                session_id=session_id,
                callbacks=callbacks,
                cfg=cfg,
                expect_actions=expect_actions,
                wait_timeout=cfg.timeout,
                poll_sec=idle_sleep,
            )
            if result is None:
                await asyncio.sleep(idle_sleep)
        except asyncio.CancelledError:
            logger.info("TurnLoop[%s]: cancelled", sid)
            break
        except Exception:
            logger.exception("TurnLoop[%s]: iteration failed", sid)
            await asyncio.sleep(idle_sleep)
