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


def _format_players(players: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, player in enumerate(players, start=1):
        pid = str(player.get("player_id") or f"player-{idx}").strip()
        name = str(player.get("name") or f"Игрок {idx}").strip()
        if name:
            lines.append(f"- {name} (id: {pid})")
        else:
            lines.append(f"- Игрок {idx} (id: {pid})")
    if not lines:
        lines.append("- Игроки пока не заданы")
    return "\n".join(lines)


def _format_npcs(npcs: Any) -> str:
    if not isinstance(npcs, list) or not npcs:
        return "- NPC пока не определены"
    lines: List[str] = []
    for npc in npcs[:8]:
        if not isinstance(npc, dict):
            continue
        name = str(npc.get("name") or "NPC").strip()
        mood = str(npc.get("mood") or "").strip()
        description = f", настроение: {mood}" if mood else ""
        lines.append(f"- {name}{description}")
    return "\n".join(lines) if lines else "- NPC пока не определены"


def _format_world_flags(flags: Any) -> str:
    if not isinstance(flags, dict) or not flags:
        return "- Нет активных флагов"
    parts: List[str] = []
    for key, value in list(flags.items())[:8]:
        parts.append(f"- {key}: {value}")
    return "\n".join(parts) if parts else "- Нет активных флагов"


def build_world_prompt_with_players(
    *,
    title: str,
    players: List[Dict[str, Any]],
    story_theme: str = "",
) -> str:
    title_s = (title or "Приключение").strip()
    theme_block = (
        f"Жанровая тема партии: {story_theme.strip()}\n"
        if story_theme and story_theme.strip()
        else ""
    )

    players_block = _format_players(players)

    return dedent(
        f"""
        Ты — мастер настольной ролевой игры. Сформируй стартовую сцену (turn=0) для кооперативного приключения.
        Название приключения: {title_s}
        {theme_block}Участники кампании:
        {players_block}

        Опиши мир на русском языке и верни **строго валидный JSON** без пояснений и комментариев. JSON должен соответствовать структуре:
        {{
          "turn": 0,
          "title": "{title_s}",
          "setting": "Краткое, но образное описание общего сеттинга (не менее 20 символов)",
          "location": "Текущее место, где находятся герои",
          "world_flags": {{"ключ": значение, ...}} — важные факты или индикаторы состояния,
          "npcs": [
            {{
              "id": "уникальный идентификатор",
              "name": "имя NPC",
              "mood": "настроение или отношение",
              "hp": число,
              "items": ["перечень предметов, которыми владеет NPC"]
            }}
          ],
          "available_items": [{{"name": "предмет", "source": "происхождение"}}],
          "opening_hook": "Динамичный вводный абзац с зацепкой сюжета",
          "visibility_defaults": {{...}} — параметры восприятия по умолчанию,
          "style": {{...}} — художественные настройки повествования,
          "story_theme": "Повтори переданную тему, если она указана"
        }}

        Требования:
        - Ключевые элементы (setting, location, opening_hook) должны быть оригинальными и связанными общей атмосферой.
        - Избегай мета-комментариев, упоминаний чата и технологий. Не ломай четвёртую стену.
        - NPC и предметы должны быть разнообразными и подходить выбранной теме.
        - Если тему не передали, придумай гармоничный жанровый акцент самостоятельно.
        - Используй естественный литературный русский язык.
        - Верни только JSON.
        """
    ).strip()


def build_roles_for_players_prompt(
    *,
    state: Dict[str, Any],
    players: List[Dict[str, Any]],
    story_theme: str = "",
) -> str:
    title = str(state.get("title") or "Приключение").strip()
    setting = str(state.get("setting") or "").strip()
    location = str(state.get("location") or "").strip()
    opening = str(state.get("opening_hook") or "").strip()
    theme_block = (
        f"Жанровая тема: {story_theme.strip()}\n"
        if story_theme and story_theme.strip()
        else (f"Жанровая тема: {str(state.get('story_theme') or '').strip()}\n" if state.get("story_theme") else "")
    )

    players_block = _format_players(players)
    npcs_block = _format_npcs(state.get("npcs"))
    flags_block = _format_world_flags(state.get("world_flags"))

    return dedent(
        f"""
        Ты — ведущий мастер ролевой игры. На основе стартовой сцены создай яркие роли для игроков.
        Название приключения: {title}
        {theme_block}Текущее местоположение: {location or 'уточни сам, используя сеттинг'}
        Краткий сеттинг:
        {setting}

        Стартовый хук:
        {opening}

        Важные факты и флаги мира:
        {flags_block}

        Доступные NPC поблизости:
        {npcs_block}

        Участники партии:
        {players_block}

        Необходимо вернуть **строго валидный JSON** вида:
        {{
          "roles_for_players": [
            {{"player_id": "id игрока", "role": "роль или амплуа", "summary": "2–3 предложения о прошлом и мотивации"}}
          ]
        }}

        Требования:
        - Количество элементов должно совпадать с числом игроков и соблюдать их порядок.
        - Каждая роль должна быть уникальной, соответствовать сеттингу и стимулировать взаимодействие с текущей локацией.
        - Summary пишется от третьего лица, без обращения к игроку напрямую и без мета-комментариев.
        - Упоминай сюжетные зацепки или NPC, если это уместно.
        - Верни только JSON, без пояснений и лишнего текста.
        """
    ).strip()


def build_initial_backstory_prompt(
    *,
    state: Dict[str, Any],
    player: Dict[str, Any],
    story_theme: str = "",
) -> str:
    name = str(player.get("name") or "Игрок").strip()
    pid = str(player.get("player_id") or "").strip()
    title = str(state.get("title") or "Приключение").strip()
    setting = str(state.get("setting") or "").strip()
    location = str(state.get("location") or "").strip()
    opening = str(state.get("opening_hook") or "").strip()
    theme = story_theme.strip() or str(state.get("story_theme") or "").strip()

    npcs_block = _format_npcs(state.get("npcs"))
    flags_block = _format_world_flags(state.get("world_flags"))
    items = state.get("available_items") if isinstance(state, dict) else []
    items_lines: List[str] = []
    if isinstance(items, list):
        for item in items[:8]:
            if isinstance(item, dict):
                name_it = str(item.get("name") or "").strip()
                source_it = str(item.get("source") or "").strip()
                if name_it:
                    if source_it:
                        items_lines.append(f"- {name_it} (источник: {source_it})")
                    else:
                        items_lines.append(f"- {name_it}")
            else:
                val = str(item).strip()
                if val:
                    items_lines.append(f"- {val}")
    items_block = "\n".join(items_lines) if items_lines else "- Особых предметов пока не обнаружено"

    theme_line = f"Жанровая тема: {theme}\n" if theme else ""

    return dedent(
        f"""
        Ты пишешь личный пролог для игрока. Нужно передать его персонажу чувство вовлечённости в стартовую сцену.
        Название приключения: {title}
        {theme_line}Игрок: {name} (id: {pid})
        Общий сеттинг:
        {setting}

        Текущая локация: {location}
        Стартовый хук для всей группы:
        {opening}

        Важные флаги мира:
        {flags_block}

        Доступные предметы сцены:
        {items_block}

        NPC поблизости:
        {npcs_block}

        Сформируй ответ в виде **валидного JSON**:
        {{
          "text": "Художественное вступление 2–4 абзаца от третьего лица, раскрывающее героя и вводящее его в сцену",
          "echo_of_action": "Короткая фраза, отражающая естественное первое действие персонажа",
          "highlights": ["Ключевые тезисы, 0–5 штук"]
        }}

        Требования:
        - Пиши на литературном русском языке, без обращения к игроку и мета-комментариев.
        - Упоминай детали окружения и связку с общей завязкой, избегай противоречий миру.
        - "echo_of_action" — одно предложение, мотивирующее героя сделать первый шаг.
        - Если у героя ещё нет определённой роли, намекни на его склонности через действия и воспоминания.
        - Верни только JSON без пояснений.
        """
    ).strip()


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


__all__ = [
    "BANLIST_DEFAULT",
    "build_world_prompt_with_players",
    "build_roles_for_players_prompt",
    "build_initial_backstory_prompt",
    "build_story_prompt",
]
