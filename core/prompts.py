from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, List, Optional

BANLIST_DEFAULT: List[str] = [
    "четвёртая стена",
    "четвертая стена",
    "мета-комментарий",
    "метакомментарий",
    "бот",
    "чат-бот",
    "чат бот",
    "чатгпт",
    "chatgpt",
]


def _split_sents(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace("\r", " ").split(". ") if p.strip()]
    return parts


def _recap_from_tail(text: str, *, sentences: int = 2, max_chars: int = 320) -> str:
    pieces = _split_sents(text)
    if not pieces:
        return ""
    snippet = " ".join(pieces[-sentences:])
    if len(snippet) > max_chars:
        snippet = snippet[-max_chars:].lstrip()
    return snippet


def _recent_history_summary(state: Dict[str, Any]) -> str:
    history = state.get("general_history") or []
    if isinstance(history, list) and history:
        for entry in reversed(history):
            text = ""
            if isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
            elif isinstance(entry, str):
                text = entry.strip()
            if text:
                recap = _recap_from_tail(text)
                if recap:
                    return recap
    parts: List[str] = []
    opening = str(state.get("opening_hook") or "").strip()
    if opening:
        parts.append(opening)
    location = str(state.get("location") or "").strip()
    if location:
        parts.append(f"Текущая локация: {location}.")
    setting = str(state.get("setting") or "").strip()
    if setting and len(setting) <= 240:
        parts.append(setting)
    return " ".join(parts)


def _format_actions(actions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, action in enumerate(actions, start=1):
        name = str(action.get("player_name") or action.get("player_id") or f"Игрок {idx}").strip()
        text = str(action.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"- {name}: {text}")
    return "\n".join(lines)


def build_story_prompt(
    *,
    state: Dict[str, Any],
    actions: List[Dict[str, Any]],
    forbidden: Optional[List[str]] = None,
    theme: Optional[str] = None,
) -> str:
    context = _recent_history_summary(state)
    if not context:
        context = "Герои стоят на пороге новой сцены."

    actions_block = _format_actions(actions)
    if not actions_block:
        actions_block = "- Игроки осматриваются и пытаются сориентироваться."

    theme_block = ""
    theme_text = theme or state.get("story_theme")
    if theme_text:
        theme_block = f"Тон сцены: {str(theme_text).strip()}\n"

    forbidden_block = ""
    words = list(dict.fromkeys([*(forbidden or []), *BANLIST_DEFAULT]))
    meta_words = [w for w in words if w]
    if meta_words:
        forbidden_block = (
            "Не упоминай, что происходящее — игра или симуляция, избегай мета-комментариев"
            " и слов вроде: " + ", ".join(meta_words[:12]) + ".\n"
        )

    return dedent(
        f"""
        {theme_block}Предыстория: {context}

        Действия игроков текущего хода:
        {actions_block}

        Продолжи художественное повествование от третьего лица. Учитывай последствия каждого действия,
        поддерживай непрерывность сцены и атмосферы. Ответ должен быть образным, динамичным и оставаться
        в рамках игрового мира. Не используй списки, не давай системных комментариев и не ломай четвёртую
        стену. {forbidden_block}Изложение должно занимать несколько связных предложений на русском языке.
        """
    ).strip()


__all__ = ["BANLIST_DEFAULT", "build_story_prompt"]
