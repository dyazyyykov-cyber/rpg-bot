from __future__ import annotations

import os
import time
import json
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, Awaitable

from .utils import ensure_dir
from . import storylets
from .engine import run_turn_with_llm, apply_effects, push_raw_story, push_private_story, push_general_story
from .schemas import TurnConfig, TurnOutputs, EffectsDelta, RawStory, PlayerStory, GeneralStory
from .fact_bank import FactBank

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

    # --- детерминированная обработка через Storylets, если применимо
    if len(actions) == 1:
        try:
            result = storylets.propose_storylet_resolution(state, {
                "player_id": actions[0].player_id,
                "player_tg": actions[0].player_tg,
                "player_name": actions[0].player_name,
                "text": actions[0].text
            })
        except Exception as e:
            result = None
            logger.warning("Turn[%s]: storylet resolution exception: %r", sid, e)
        if result and result.get("confidence", 0) >= 0.6:
            eff = result["effects"]
            # Очистка эффектов от служебных полей и корректировка схемы
            for pd in eff.get("players", []) or []:
                if "flags" in pd:
                    flags = pd.pop("flags")
                    if "status" in flags:
                        pd["status_apply"] = {"status": flags["status"]}
            for nd in eff.get("npcs", []) or []:
                if "flags" in nd:
                    flags = nd.pop("flags")
                    # status flags for NPCs are ignored (no schema field)
            for item in (eff.get("introductions") or {}).get("items", []):
                if isinstance(item, dict):
                    item.pop("meta", None)
            # Применяем эффекты к состоянию
            apply_effects(state, eff)
            old_turn = int(state.get("turn", 0))
            state["turn"] = old_turn + 1
            new_turn = state["turn"]
            # Формируем тексты результата
            raw_text = result["raw"].get("text", "")
            private_text = result["private"].get("text", "")
            general_text = result["general"].get("text", "")
            player_id = str(actions[0].player_id or "")
            echo_text = actions[0].text or ""
            push_raw_story(state, {"text": raw_text})
            push_private_story(state, player_id, {"text": private_text, "echo_of_action": echo_text})
            push_general_story(state, {"text": general_text})
            # Обновляем FactBank фактами эффекта
            fb = FactBank.from_state(state)
            fb.apply_effects(eff, new_turn, source="storylet")
            state["fact_bank"] = fb.export()
            # Логирование и рассылка
            try:
                effects_obj = EffectsDelta.model_validate(eff)
            except Exception as e:
                logger.error("Turn[%s]: EffectsDelta validation failed for storylet effects: %s", sid, e)
                effects_obj = eff  # use dict if validation fails
            try:
                await callbacks.notify_effects(effects_obj)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception("Turn[%s]: notify_effects failed: %r", sid, e)
            try:
                raw_obj = RawStory.model_validate({"text": raw_text})
            except Exception:
                raw_obj = RawStory(text=raw_text)
            try:
                await callbacks.notify_raw_story(raw_obj)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception("Turn[%s]: notify_raw_story failed: %r", sid, e)
            private_obj = PlayerStory(text=private_text or " ", highlights=None, echo_of_action=echo_text or " ")
            try:
                await callbacks.notify_private_story(player_id, private_obj)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception("Turn[%s]: notify_private_story(%s) failed: %r", sid, player_id, e)
            general_obj = GeneralStory(text=general_text or " ", highlights=None)
            try:
                if not (general_obj.text or "").strip():
                    logger.error("Turn[%s]: GENERAL empty text in storylet result -> abort turn", sid)
                    raise RuntimeError("General story text is empty")
                await callbacks.notify_general_story(general_obj)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception("Turn[%s]: notify_general_story failed: %r", sid, e)
                raise
            await callbacks.save_state(state)
            logger.info("Turn[%s #%d]: state saved (turn now %s)", sid, old_turn, state.get("turn"))
            if callbacks.telemetry:
                try:
                    await callbacks.telemetry({
                        "session_id": session_id,
                        "turn": state.get("turn"),
                        "llm_time_sec": 0.0,
                        "actions_cnt": len(actions),
                        "private_cnt": 1,
                        "general_present": True,
                    })
                except Exception:
                    logger.exception("Turn[%s]: telemetry failed", sid)
            outputs = TurnOutputs(
                effects=effects_obj if isinstance(effects_obj, EffectsDelta) else EffectsDelta.model_validate(eff),
                raw=raw_obj if isinstance(raw_obj, RawStory) else RawStory.model_validate({"text": raw_text}),
                general=general_obj,
                players_private={player_id: private_obj},
                verified_facts=[],
                general_outline=None,
                coverage=None,
                turn=new_turn,
                telemetry={}
            )
            return outputs

    # --- запускаем оркестровку хода (LLM)
    t0 = time.time()
    outputs: TurnOutputs = await run_turn_with_llm(
        state=state,
        actions=[{
            "player_id": a.player_id,
            "player_tg": a.player_tg,
            "player_name": a.player_name,
            "text": a.text,
        } for a in actions],
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
