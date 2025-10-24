from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata as _ud
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar

T = TypeVar("T")

# =============================================================================
# .env loader (автозапуск при импорте) — совместимо с текущим проектом
# =============================================================================

_DOTENV_LOADED = False

_ENV_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:export\s+)?                # допускаем 'export KEY=...'
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?P<val>.*?)
    \s*$
""",
    re.VERBOSE,
)


def _strip_quotes(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    return val


def load_env_file(path: str | Path | None = None) -> None:
    """
    Простой загрузчик .env: KEY=VALUE, строки с # — комментарии.
    Не трогает уже заданные переменные окружения.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    env_path = Path(path or os.getenv("ENV_FILE", ".env"))
    if not env_path.exists():
        _DOTENV_LOADED = True
        return

    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            m = _ENV_LINE_RE.match(s)
            if not m:
                continue
            key = m.group("key").strip()
            val = _strip_quotes(m.group("val").strip())
            if key and key not in os.environ:
                os.environ[key] = val
    finally:
        _DOTENV_LOADED = True


# Автозагрузка при импорте
load_env_file()

# =============================================================================
# ENV
# =============================================================================

def _to_bool(val: str) -> bool:
    v = val.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot cast to bool: {val!r}")


def get_env(
    name: str,
    default: Optional[T] = None,
    cast: Optional[Callable[[str], T]] = None,
    required: bool = False,
) -> T | None:
    """
    Читает переменную окружения (с учётом уже подгружённого .env).
    - cast: функция приведения (если None — вернёт строку).
            Особый случай: cast=bool поддерживает 1/0, true/false, yes/no, on/off.
    - required=True — бросит RuntimeError, если переменная отсутствует и default is None.
    """
    raw = os.getenv(name, None)
    if raw is None:
        if required and default is None:
            raise RuntimeError(f"ENV {name} is required but not set")
        return default

    if cast is None:
        # type: ignore[return-value]
        return raw

    try:
        if cast is bool:
            # type: ignore[return-value]
            return _to_bool(raw)
        # type: ignore[return-value]
        return cast(raw)
    except Exception as e:
        if default is not None:
            return default
        raise RuntimeError(f"ENV {name} cast error: {e}") from e


# =============================================================================
# ФАЙЛЫ / ЛОГИ
# =============================================================================

def _ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    """Создаёт директорию (или родительскую директорию файла)."""
    p = Path(path)
    if p.suffix:
        _ensure_parent_dir(p)
    else:
        p.mkdir(parents=True, exist_ok=True)


def log_text(path: str | Path, content: str) -> None:
    """Пишет текст в файл (UTF-8), перезаписывая."""
    p = Path(path)
    _ensure_parent_dir(p)
    p.write_text(content if isinstance(content, str) else str(content), encoding="utf-8")


def log_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Пишет JSON в файл (UTF-8, без ASCII-эскейпа)."""
    p = Path(path)
    _ensure_parent_dir(p)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def purge_session_logs(session_id: str, base_dir: str | Path = "./logs") -> None:
    """
    Удаляет каталог логов конкретной сессии: ./logs/session_<sid>/
    Безопасно игнорирует отсутствие.
    """
    try:
        p = Path(base_dir) / f"session_{session_id}"
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        # не пробрасываем, чистка логов — best-effort
        pass


def purge_global_logs_older_than(
    days: int,
    base_dir: str | Path = "./logs",
    patterns: Iterable[str] = ("EFFECTS_", "PRIVATE_", "GENERAL_"),
) -> int:
    """
    Удаляет файлы верхнего уровня в ./logs, старше N дней, чьи имена начинаются
    с заданных префиксов. Возвращает количество удалённых файлов.

    Использовать опционально как фоновую санитацию глобальных логов.
    """
    deleted = 0
    try:
        root = Path(base_dir)
        if not root.exists():
            return 0
        threshold = datetime.utcnow() - timedelta(days=days)
        for entry in root.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            if not any(name.startswith(p) for p in patterns):
                continue
            try:
                mtime = datetime.utcfromtimestamp(entry.stat().st_mtime)
                if mtime < threshold:
                    entry.unlink(missing_ok=True)
                    deleted += 1
            except Exception:
                # best-effort
                continue
    except Exception:
        # best-effort
        return deleted
    return deleted


# =============================================================================
# ХЕЛПЕРЫ ДЛЯ РАССЫЛКИ ДЛИННЫХ СООБЩЕНИЙ
# =============================================================================

def chunk_text(
    text: str,
    hard_limit: int = 4000,
    breakpoints: Iterable[str] = ("\n\n", "\n", " "),
) -> List[str]:
    """Делит длинный текст на части <= hard_limit. Разрезает по breakpoints, если возможно."""
    if not text:
        return [""]

    chunks: List[str] = []
    buf = text

    while len(buf) > hard_limit:
        cut = -1
        window = buf[:hard_limit]
        for sep in breakpoints:
            pos = window.rfind(sep)
            if pos > cut:
                cut = pos
        if cut <= 0:
            cut = hard_limit
        part = buf[:cut].rstrip()
        chunks.append(part)
        buf = buf[cut:].lstrip()

    if buf:
        chunks.append(buf)

    return chunks


# =============================================================================
# МЯГКАЯ НОРМАЛИЗАЦИЯ СТРОК + ФИЛЬТР МУСОРА
# =============================================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-Z0-9\"«(])")
_ZERO_WIDTH_RE = re.compile("[\u200B-\u200D\uFEFF]")
_CTRL_RE = re.compile("[\u0000-\u001F\u007F]")

# Некоторые артефакты, замеченные в логах (например, \"в空气...\")
_ARTIFACT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"в空气\S*", flags=re.IGNORECASE),  # удаляем китайский фрагмент-мойжибэйк
]


def normalize_text(s: str, *, collapse_spaces: bool = True) -> str:
    """Unicode NFC + зачистка управляющих, нулевой ширины и известных артефактов."""
    if s is None:
        return ""
    # NFC
    s = _ud.normalize("NFC", str(s))
    # убрать zero-width и control
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _CTRL_RE.sub(" ", s)
    # убрать артефакты
    for patt in _ARTIFACT_PATTERNS:
        s = patt.sub("", s)
    # схлопнуть пробелы
    if collapse_spaces:
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = s.strip()
    return s


def as_str(x: Any, *, max_len: int | None = None) -> str:
    """
    Безопасно приводит значение к строке:
    - dict -> dict.get('text') если есть, иначе JSON;
    - list -> '; '.join(map(str, ...));
    - None -> ''.
    Никаких смысловых замен. Только форма.
    """
    if x is None:
        s = ""
    elif isinstance(x, str):
        s = x
    elif isinstance(x, dict):
        if "text" in x and isinstance(x["text"], str):
            s = x["text"]
        else:
            try:
                s = json.dumps(x, ensure_ascii=False)
            except Exception:
                s = str(x)
    elif isinstance(x, (list, tuple, set)):
        try:
            s = "; ".join(str(i) for i in x)
        except Exception:
            s = str(list(x))
    else:
        s = str(x)

    if max_len is not None and max_len > 0 and len(s) > max_len:
        return s[:max_len]
    return s


def trim_sentences(text: str, max_sents: int = 9, keep_min: int = 1) -> str:
    """
    Обрезает текст до max_sents предложений.
    Не меняет порядок, не редактирует содержимое.
    """
    if not text:
        return ""
    sents = _SENT_SPLIT_RE.split(text.strip())
    if len(sents) <= max_sents:
        return text.strip()
    kept = sents[:max(keep_min, max_sents)]
    return " ".join(seg.strip() for seg in kept if seg.strip())


# =============================================================================
# BANLIST / FORBIDDEN
# =============================================================================

def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_strings(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)


def find_forbidden_hits(obj: Any, banlist: Optional[List[str]]) -> List[str]:
    if not banlist:
        return []
    hits: List[str] = []
    low_forb = [s.lower() for s in banlist if isinstance(s, str) and s]
    if not low_forb:
        return []
    for s in _iter_strings(obj):
        sl = str(s).lower()
        for w in low_forb:
            if w and w in sl:
                hits.append(w)
    # уникализируем сохраняя порядок
    seen: Set[str] = set()
    out: List[str] = []
    for w in hits:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def no_forbidden(obj: Any, banlist: Optional[List[str]]) -> bool:
    return len(find_forbidden_hits(obj, banlist)) == 0


# =============================================================================
# EFFECTS LINKING / COVERAGE
# =============================================================================

def _extract_effect_referents(effects: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Выдёргивает имена предметов/ключей флагов/имен NPC/имён игроков, затронутых эффектами."""
    refs: Set[str] = set()
    if not isinstance(effects, dict):
        return refs

    # world / location
    if effects.get("location") not in (None, ""):
        loc = normalize_text(str(effects.get("location")))
        if loc:
            refs.add(loc.lower())
    wf = effects.get("world_flags") or {}
    if isinstance(wf, dict):
        for k in wf.keys():
            if k:
                refs.add(str(k).lower())

    # scene items
    for arr_key in ("scene_items_add", "scene_items_remove"):
        arr = effects.get(arr_key)
        if isinstance(arr, list):
            for it in arr:
                name = None
                if isinstance(it, dict):
                    name = it.get("name")
                else:
                    name = str(it)
                if name:
                    refs.add(normalize_text(str(name)).lower())

    # units
    id_to_name: Dict[str, str] = {}
    if snapshot:
        for p in (snapshot.get("players") or []):
            pid = str(p.get("player_id")) if p and p.get("player_id") is not None else None
            nm = normalize_text(str(p.get("name"))) if p and p.get("name") is not None else ""
            if pid and nm:
                id_to_name[pid] = nm
        for n in (snapshot.get("npcs") or []):
            nid = str(n.get("id")) if n and n.get("id") is not None else None
            nm = normalize_text(str(n.get("name"))) if n and n.get("name") is not None else ""
            if nid and nm:
                id_to_name[nid] = nm

    for coll_key in ("players", "npcs"):
        arr = effects.get(coll_key)
        if isinstance(arr, list):
            for ent in arr:
                if not isinstance(ent, dict):
                    continue
                # по id восстановим имя для проверки упоминаний в тексте
                key = "player_id" if coll_key == "players" else "npc_id"
                uid = ent.get(key) or ent.get("id")
                if uid and id_to_name.get(str(uid)):
                    refs.add(id_to_name[str(uid)].lower())
                # предметы
                for it_key in ("items_add", "items_remove"):
                    it = ent.get(it_key)
                    if isinstance(it, list):
                        for name in it:
                            if name:
                                refs.add(normalize_text(str(name)).lower())
    return refs


def links_to_effects(
    general_text: str,
    effects: Optional[Dict[str, Any]],
    *,
    snapshot: Optional[Dict[str, Any]] = None,
    min_refs: int = 2,
) -> bool:
    """
    Эвристическая проверка: General должен упоминать минимум min_refs сущностей/фактов,
    затронутых в effects (предметы/флаги/имена/локации).
    """
    if not effects:
        return True  # нечего проверять
    text = normalize_text(general_text or "").lower()
    if not text:
        return False
    refs = _extract_effect_referents(effects, snapshot=snapshot)
    if not refs:
        return True
    hits = 0
    for r in refs:
        if not r or len(r) < 2:
            continue
        # грубая проверка вхождения слова/фразы
        if r in text:
            hits += 1
            if hits >= max(1, min_refs):
                return True
    return False


def compute_effects_coverage(
    actions: Optional[List[Dict[str, Any]]],
    effects: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Считаем грубую метрику покрытия: сколько осмысленных дельт против числа действий."""
    total_actions = 0
    for a in (actions or []):
        if isinstance(a, dict) and str(a.get("text") or "").strip():
            total_actions += 1

    deltas = 0
    details: Dict[str, int] = defaultdict(int)
    e = effects or {}

    def bump(k: str, n: int = 1):
        nonlocal deltas
        if n > 0:
            details[k] += n
            deltas += n

    if isinstance(e, dict):
        if e.get("location") not in (None, ""):
            bump("location")
        wf = e.get("world_flags") or {}
        if isinstance(wf, dict):
            bump("world_flags", len(wf))
        for key in ("scene_items_add", "scene_items_remove"):
            arr = e.get(key)
            if isinstance(arr, list):
                bump(key, len(arr))
        for coll_key in ("players", "npcs"):
            arr = e.get(coll_key)
            if isinstance(arr, list):
                for ent in arr:
                    if not isinstance(ent, dict):
                        continue
                    changed = 0
                    if ent.get("hp_delta") not in (None, 0):
                        changed += 1
                    if ent.get("hp") is not None:
                        changed += 1
                    for it_key in ("items_add", "items_remove"):
                        it = ent.get(it_key)
                        if isinstance(it, list) and it:
                            changed += len(it)
                    if ent.get("position"):
                        changed += 1
                    if coll_key == "npcs" and ent.get("mood"):
                        changed += 1
                    bump(f"{coll_key}", changed)
        intro = e.get("introductions") or {}
        if isinstance(intro, dict):
            for k in ("npcs", "items", "locations"):
                v = intro.get(k)
                if isinstance(v, list) and v:
                    bump(f"intro_{k}", len(v))

    ratio = (float(deltas) / float(total_actions)) if total_actions > 0 else (1.0 if deltas == 0 else 0.0)
    return {
        "actions": total_actions,
        "deltas": deltas,
        "ratio": ratio,
        "details": dict(details),
    }


# =============================================================================
# ИНВЕНТАРЬ / ИНВАРИАНТЫ
# =============================================================================

def _normalize_item_entry(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        name = normalize_text(str(x.get("name", "")))
        src = str(x.get("source", "scene") or "scene").strip() or "scene"
        return {"name": name[:80], "source": src[:32]}
    name = normalize_text(str(x))
    return {"name": name[:80], "source": "scene"}


def _scene_items_by_name(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in (state.get("available_items") or []):
        ent = _normalize_item_entry(it)
        key = ent["name"].lower()
        if key:
            out[key] = ent
    return out


def _find_player(state: Dict[str, Any], player_id: str) -> Optional[Dict[str, Any]]:
    for p in (state.get("players") or []):
        if str(p.get("player_id")) == str(player_id):
            return p
    return None


def add_item_to_player(state: Dict[str, Any], player_id: str, item_name: str) -> bool:
    """Идempotent: перемещает предмет из сцены в инвентарь игрока. Возврат True, если было изменение."""
    if not item_name:
        return False
    item_name_n = normalize_text(item_name)
    if not item_name_n:
        return False

    p = _find_player(state, player_id)
    if not p:
        return False

    # Удалим из сцены (если есть)
    scene = state.get("available_items") or []
    idx = None
    for i, it in enumerate(scene):
        ent = _normalize_item_entry(it)
        if ent["name"].lower() == item_name_n.lower():
            idx = i
            break
    if idx is not None:
        scene.pop(idx)

    # Добавим игроку, если нет
    items = p.setdefault("items", [])
    for nm in items:
        if normalize_text(str(nm)).lower() == item_name_n.lower():
            return idx is not None  # уже был у игрока, но удалили со сцены — тоже изменение
    items.append(item_name_n)
    return True


def move_item_scene_to_player(state: Dict[str, Any], player_id: str, item_name: str) -> bool:
    return add_item_to_player(state, player_id, item_name)


def ensure_inventory_invariants(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Правила:
    - available_items — только предметы сцены (source="scene").
    - У предмета всегда есть name!=\"\"; имена нормализованы.
    - Один и тот же предмет (по имени) не должен одновременно лежать в сцене и в чьём-то инвентаре.
      (Дубликаты в инвентарях разных акторов допускаются как разные экземпляры.)
    Возвращает state (тот же объект) для удобства чейнинга.
    """
    # нормализуем сцену
    scene_items: List[Dict[str, Any]] = []
    for it in (state.get("available_items") or []):
        ent = _normalize_item_entry(it)
        ent["source"] = "scene"
        if ent["name"]:
            scene_items.append(ent)
    state["available_items"] = scene_items

    # собрать все имена, занятые инвентарями
    held: Set[str] = set()
    for key in ("players", "npcs"):
        for actor in (state.get(key) or []):
            lst = actor.get("items") or []
            clean: List[str] = []
            for nm in lst:
                nm_n = normalize_text(str(nm))
                if not nm_n:
                    continue
                clean.append(nm_n)
                held.add(nm_n.lower())
            actor["items"] = clean

    # вычесть занятые из сцены
    state["available_items"] = [it for it in state["available_items"] if it["name"].lower() not in held]

    return state


# =============================================================================
# __all__
# =============================================================================

__all__ = [
    # env & fs
    "load_env_file",
    "get_env",
    "ensure_dir",
    "log_text",
    "log_json",
    "purge_session_logs",
    "purge_global_logs_older_than",
    # text
    "normalize_text",
    "chunk_text",
    "as_str",
    "trim_sentences",
    # banlist
    "find_forbidden_hits",
    "no_forbidden",
    # effects
    "links_to_effects",
    "compute_effects_coverage",
    # inventory
    "add_item_to_player",
    "move_item_scene_to_player",
    "ensure_inventory_invariants",
]
