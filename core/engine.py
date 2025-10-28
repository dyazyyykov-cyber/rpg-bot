from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .llm import generate_text, find_forbidden_hits
from .prompts import BANLIST_DEFAULT, build_story_prompt
from .schemas import GeneralStory, TurnConfig, TurnOutputs
from .utils import ensure_dir, get_env

logger = logging.getLogger(__name__)
logger.propagate = True

GENERAL_TEMPERATURE = get_env("GENERAL_TEMPERATURE", 0.7, float)
GENERAL_TOP_P = get_env("GENERAL_TOP_P", 0.9, float)
MAX_HISTORY_ENTRIES = int(get_env("GENERAL_HISTORY_LIMIT", 80, int) or 80)

SYSTEM_PROMPT = (
    "Ты — рассказчик и мастер настольной RPG. Говори на русском языке,"
    " описывая события художественным текстом от третьего лица."
    " Не признавайся, что ты ИИ, и не упоминай правила или формат игры."
)


def _normalize_action(action: Any) -> Dict[str, str]:
    if isinstance(action, dict):
        return {
            "player_id": str(action.get("player_id") or ""),
            "player_name": str(action.get("player_name") or ""),
            "text": str(action.get("text") or ""),
        }
    return {
        "player_id": str(getattr(action, "player_id", "") or ""),
        "player_name": str(getattr(action, "player_name", "") or ""),
        "text": str(getattr(action, "text", "") or ""),
    }


def _collect_actions(actions: List[Any]) -> List[Dict[str, str]]:
    normed: List[Dict[str, str]] = []
    for action in actions or []:
        item = _normalize_action(action)
        if item["text"].strip():
            normed.append(item)
    return normed


def _banlist_from_state(state: Dict[str, Any]) -> List[str]:
    extra: List[str] = []
    try:
        raw = state.get("forbidden") or []
        if isinstance(raw, list):
            extra = [str(x) for x in raw if str(x).strip()]
    except Exception:
        extra = []
    return list(dict.fromkeys([*BANLIST_DEFAULT, *extra]))


def _append_general_history(state: Dict[str, Any], text: str, keep_last: int = MAX_HISTORY_ENTRIES) -> None:
    entry = {"text": text}
    history = state.get("general_history")
    if not isinstance(history, list):
        history = []
    history.append(entry)
    if keep_last and len(history) > keep_last:
        history = history[-keep_last:]
    state["general_history"] = history


async def run_turn_with_llm(
    state: Dict[str, Any],
    actions: List[Dict[str, Any]],
    cfg: TurnConfig,
) -> TurnOutputs:
    norm_actions = _collect_actions(actions)
    banlist = _banlist_from_state(state)
    theme = (state.get("story_theme") or "").strip() or None

    sid = str(getattr(cfg, "group_chat_id", "unknown"))
    turn_no = int(state.get("turn", 0))
    turn_dir = os.path.join(".", "logs", f"session_{sid}", f"turn_{turn_no:04d}")
    ensure_dir(turn_dir)
    llm_dir = os.path.join(turn_dir, "llm")
    ensure_dir(llm_dir)

    prompt = build_story_prompt(state=state, actions=norm_actions, forbidden=banlist, theme=theme)
    try:
        with open(os.path.join(turn_dir, "story_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
    except Exception:
        pass

    temperature = float(getattr(cfg, "temperature_general", GENERAL_TEMPERATURE))
    top_p = float(getattr(cfg, "top_p_general", GENERAL_TOP_P))
    timeout = getattr(cfg, "llm_timeout", None)

    max_attempts = 2
    last_hits: List[str] = []
    final_text: Optional[str] = None
    prompt_attempt = prompt
    attempts_used = 0

    for attempt in range(1, max_attempts + 1):
        logger.info("TURN[%s #%d]: request narrative attempt %d", sid, turn_no, attempt)
        attempts_used = attempt
        story_text = await generate_text(
            prompt=prompt_attempt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            log_prefix=f"GENERAL_{attempt}",
            log_dir=llm_dir,
        )
        story_text = story_text.strip()
        if not story_text:
            last_hits = ["empty_response"]
        else:
            last_hits = find_forbidden_hits(story_text, banlist)

        if story_text and not last_hits:
            final_text = story_text
            break

        if attempt < max_attempts:
            reminder = (
                "\n\nНапоминание: отвечай художественным описанием без мета-комментариев,"
                " избегай слов из запрета и сделай ответ содержательным."
            )
            prompt_attempt = prompt + reminder
            continue

        if story_text:
            final_text = story_text
        break

    if not final_text:
        raise RuntimeError("LLM не вернула текст сцены")

    try:
        with open(os.path.join(turn_dir, "general_story.txt"), "w", encoding="utf-8") as f:
            f.write(final_text)
    except Exception:
        pass

    if last_hits:
        logger.warning("TURN[%s #%d]: final narration contains forbidden tokens %s", sid, turn_no, last_hits)

    _append_general_history(state, final_text)
    state["turn"] = int(state.get("turn", 0)) + 1

    general_story = GeneralStory(text=final_text)
    telemetry = {
        "llm_attempts": attempts_used,
        "forbidden_hits": last_hits,
    }

    return TurnOutputs(general=general_story, turn=state["turn"], telemetry=telemetry)


__all__ = [
    "run_turn_with_llm",
]
