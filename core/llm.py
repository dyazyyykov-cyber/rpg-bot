# core/llm.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


class LLMRequestTimeoutError(asyncio.TimeoutError):
    """Timeout while waiting for the LLM HTTP endpoint."""

    def __init__(self, *, url: str, timeout: Optional[float]):
        self.url = url
        self.timeout = timeout
        if timeout is not None:
            message = f"LLM request to {url} timed out after {timeout:.0f}s"
        else:
            message = f"LLM request to {url} timed out"
        super().__init__(message)

from .utils import get_env
from .schemas import as_json_schema  # единая точка экспорта схем

logger = logging.getLogger(__name__)
logger.propagate = True

# ===================== ENV / CONFIG =====================
LLM_URL = get_env("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL_DEFAULT = get_env("LLM_MODEL", "qwen3")
LLM_HTTP_TIMEOUT = int(get_env("LLM_HTTP_TIMEOUT", "180"))

LOG_LLM_FULL = str(get_env("LOG_LLM_FULL", "1")).strip().lower() in ("1", "true", "yes", "on")
LLM_LOG_CONTENT = str(get_env("LLM_LOG_CONTENT", "1")).strip().lower() in ("1", "true", "yes", "on")

# ===================== FS HELPERS =====================

def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        return None

def _ts() -> str:
    # 20251012-184238-735
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

def _dump_json(path: str, obj: Any) -> None:
    if not LOG_LLM_FULL:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(obj, (dict, list)):
                json.dump(obj, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(obj))
    except Exception:
        pass

# ===================== HTTP =====================

async def _http_post_json(payload: Dict[str, Any], timeout: Optional[int]) -> Dict[str, Any]:
    effective_timeout: Optional[int]
    if timeout is None or (isinstance(timeout, (int, float)) and timeout <= 0):
        effective_timeout = LLM_HTTP_TIMEOUT
    else:
        effective_timeout = timeout
    client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

    # минимальная отладочная сводка
    dbg = {
        "url": LLM_URL,
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "seed": payload.get("seed"),
        "has_response_format": bool(payload.get("response_format")),
    }
    try:
        msgs = payload.get("messages", [])
        if isinstance(msgs, list) and msgs:
            prev = []
            for m in msgs[:2]:
                c = str(m.get("content", ""))[:180].replace("\n", "\\n")
                prev.append({"role": m.get("role", "?"), "content": c})
            dbg["messages_preview"] = prev
    except Exception:
        pass
    logger.info("LLM POST: %s", dbg)

    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as sess:
            async with sess.post(LLM_URL, json=payload) as resp:
                text = await resp.text()
                if LOG_LLM_FULL:
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        parsed = {"status_code": resp.status, "text": text[:20000]}
                    _dump_json(os.path.join(".", "logs", f"HTTP_{_ts()}.json"), parsed)
                if resp.status != 200:
                    raise RuntimeError(f"LLM HTTP {resp.status}: {text[:1500]}")
                try:
                    return json.loads(text)
                except Exception:
                    logger.exception("LLM: invalid JSON in HTTP response body (not message.content)")
                    raise
    except asyncio.TimeoutError as exc:
        logger.warning(
            "LLM HTTP timeout after %ss (url=%s)",
            "∞" if effective_timeout is None else effective_timeout,
            LLM_URL,
        )
        raise LLMRequestTimeoutError(url=LLM_URL, timeout=effective_timeout) from exc

# ===================== PROMPT / MESSAGES =====================

_SYSTEM_JSON = (
    "Ты — генератор СТРОГОГО JSON. Верни РОВНО ОДИН JSON-объект; "
    "НИКАКОГО текста вне JSON, без бэктиков и комментариев. "
    "НЕ добавляй лишние поля и не меняй имена ключей. Не сериализуй массивы/объекты строками. Язык: русский."
)

def _build_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": _SYSTEM_JSON}, {"role": "user", "content": prompt}]

# ===================== JSON DIAG =====================

def _extract_json_for_diag(text: str) -> Any:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

# ===================== SCHEMA (NO SANITIZE) =====================

def _inline_local_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Безопасно инлайнит локальные #/$defs, не меняя типы/границы.
    Это не «санитайзер», а только раскрытие ссылок, чтобы движок JSON-schema
    на стороне модели не споткнулся на $ref. Если что-то не удаётся — возвращаем как есть.
    """
    try:
        root = deepcopy(schema)

        def resolve(path: str) -> Optional[Dict[str, Any]]:
            if not (isinstance(path, str) and path.startswith("#/")):
                return None
            node: Any = root
            for p in path[2:].split("/"):
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    return None
            return node if isinstance(node, dict) else None

        def walk(n: Any) -> Any:
            if isinstance(n, dict):
                if "$ref" in n and isinstance(n["$ref"], str):
                    tgt = resolve(n["$ref"])
                    if tgt:
                        merged = deepcopy(tgt)
                        extras = {k: v for k, v in n.items() if k != "$ref"}
                        merged.update(extras)
                        return walk(merged)
                return {k: walk(v) for k, v in n.items()}
            if isinstance(n, list):
                return [walk(x) for x in n]
            return n

        res = walk(root)
        if isinstance(res, dict):
            res.pop("$defs", None)
            res.pop("definitions", None)
        return res if isinstance(res, dict) else schema
    except Exception:
        return schema

# ===================== FORBIDDEN / EFFECTS CHECKS =====================

def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_strings(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)


def _find_forbidden_hits(obj: Any, forbidden: Optional[List[str]]) -> List[str]:
    if not forbidden:
        return []
    hits: List[str] = []
    low_forb = [s.lower() for s in forbidden if isinstance(s, str) and s]
    if not low_forb:
        return []
    for s in _iter_strings(obj):
        sl = s.lower()
        for w in low_forb:
            if w and w in sl:
                hits.append(w)
    # уникализируем сохраняя порядок
    seen, out = set(), []
    for w in hits:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _effects_has_delta(obj: Dict[str, Any]) -> bool:
    """Проверка, что effects содержит хотя бы одну дельту.
    Поддерживает как корневую форму EffectsDelta, так и {effects: EffectsDelta}.
    """
    eff = obj.get("effects") if isinstance(obj, dict) and "effects" in obj else obj
    if not isinstance(eff, dict):
        return False

    if eff.get("location") not in (None, ""):
        return True

    wf = eff.get("world_flags")
    if isinstance(wf, dict) and len(wf) > 0:
        return True

    for key in ("scene_items_add", "scene_items_remove"):
        arr = eff.get(key)
        if isinstance(arr, list) and len(arr) > 0:
            return True

    def _has_unit_deltas(coll_key: str) -> bool:
        arr = eff.get(coll_key)
        if not isinstance(arr, list):
            return False
        for ent in arr:
            if not isinstance(ent, dict):
                continue
            if ent.get("hp_delta") not in (None, 0):
                return True
            if ent.get("hp") is not None:
                return True
            for it_key in ("items_add", "items_remove"):
                it = ent.get(it_key)
                if isinstance(it, list) and it:
                    return True
            if ent.get("position"):
                return True
            if coll_key == "npcs" and ent.get("mood"):
                return True
        return False

    if _has_unit_deltas("players") or _has_unit_deltas("npcs"):
        return True

    intro = eff.get("introductions")
    if isinstance(intro, dict):
        for k in ("npcs", "items", "locations"):
            v = intro.get(k)
            if isinstance(v, list) and v:
                return True
    return False

# ===================== CORE CALL =====================

async def chat_json_schema(
    *,
    model: str,
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
    temperature: float = 0.2,
    timeout: Optional[int] = None,
    log_prefix: Optional[str] = None,
    log_dir: Optional[str] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    inline_refs: bool = True,
) -> Dict[str, Any]:
    """
    Один жёсткий путь: JSON Schema → llama.cpp json_schema(strict).
    ВАЖНО: max_tokens НЕ отправляем (чтобы не обрывать JSON).
    Без «санитайзеров»: типы и границы не меняем. Разрешено только безопасное раскрытие $ref.
    """
    req_id = _ts()
    if log_dir:
        _ensure_dir(log_dir)

    schema_src = as_json_schema(json_schema)
    schema_final = _inline_local_refs(schema_src) if inline_refs else schema_src

    schema_name = (isinstance(schema_final, dict) and (schema_final.get("title") or schema_final.get("$id"))) or "Schema"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "temperature": float(temperature),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema_name),
                "schema": schema_final,
                "strict": True,
            },
        },
    }
    if top_p is not None:
        try:
            payload["top_p"] = float(top_p)
        except Exception:
            payload["top_p"] = top_p
    if seed is not None:
        try:
            payload["seed"] = int(seed)
        except Exception:
            payload["seed"] = seed

    if log_dir:
        base = (log_prefix or "LLM").replace(":", "_")
        _dump_json(os.path.join(log_dir, f"{base}_request_{req_id}.json"), payload)
        _dump_json(os.path.join(log_dir, f"{base}_schema_original_{req_id}.json"), schema_src)
        _dump_json(os.path.join(log_dir, f"{base}_schema_sanitized_{req_id}.json"), schema_final)  # фактически inline-референсы

    # Вызов
    data = await _http_post_json(payload, timeout=timeout)

    # usage / latency лог
    try:
        usage = data.get("usage") or {}
        logger.info("LLM USAGE[%s]: %s", log_prefix or "-", json.dumps(usage, ensure_ascii=False))
    except Exception:
        pass

    # Контент
    try:
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise RuntimeError(f"LLM response has no content: {str(data)[:600]}")

    if LLM_LOG_CONTENT:
        logger.info("LLM RAW content[%s]: %s", log_prefix or "-", content[:1200])

    if log_dir:
        base = (log_prefix or "LLM").replace(":", "_")
        try:
            parsed = json.loads(content)
            _dump_json(os.path.join(log_dir, f"{base}_content_{req_id}.json"), parsed)
        except Exception:
            _dump_json(os.path.join(log_dir, f"{base}_content_{req_id}.json"), {"content": content})
        _dump_json(os.path.join(log_dir, f"{base}_http_response_{req_id}.json"), data)

    # Жёсткий парсинг
    try:
        obj = json.loads(content)
    except Exception:
        probe = _extract_json_for_diag(content)
        if probe is not None:
            logger.error("Strict expected pure JSON object; got extra text around JSON — refusing by policy.")
        else:
            logger.error("Strict expected pure JSON; got non-JSON. Head: %s", content[:400])
        raise RuntimeError("LLM returned non-JSON content under strict json_schema")

    if not isinstance(obj, dict):
        raise RuntimeError(f"LLM returned non-object root: {type(obj)}")

    return obj

# ===================== RETRYING WRAPPER =====================

async def generate_json(
    *,
    model: Optional[str],
    schema: Any,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
    seed: Optional[int] = None,
    max_retries: int = 1,
    timeout: Optional[int] = None,
    log_prefix: Optional[str] = None,
    log_dir: Optional[str] = None,
    top_p: Optional[float] = None,
    forbidden: Optional[List[str]] = None,
    require_nonempty_effects: bool = False,
    actions_for_effects: Optional[List[Dict[str, Any]]] = None,
    inline_refs: bool = True,
) -> Any:
    """
    Унифицированная точка генерации строго-валидного JSON.
    Триггеры для ретраев: невалидный JSON (исключение), forbidden-хиты, пустые эффекты при наличии действий.
    Если `schema` — pydantic-модель/класс: выполняется `.model_validate(...)` перед возвратом.
    """
    if not messages:
        if not prompt:
            raise TypeError("generate_json: provide either `messages` or `prompt`")
        messages = _build_messages(prompt)

    model_name = model or LLM_MODEL_DEFAULT
    retries = max(0, int(max_retries))

    def _with_retry_hint(msgs: List[Dict[str, str]], reason: str) -> List[Dict[str, str]]:
        new_msgs = [dict(m) for m in msgs]
        # добавим к последнему пользовательскому сообщению короткий hint
        for i in range(len(new_msgs) - 1, -1, -1):
            if new_msgs[i].get("role") == "user":
                new_msgs[i] = dict(new_msgs[i])
                new_msgs[i]["content"] = (
                    str(new_msgs[i]["content"]) +
                    "\n\nИсправь предыдущие ошибки: " + reason +
                    ". Верни СТРОГИЙ JSON, ровно по схеме."
                )
                return new_msgs
        # если нет user-сообщения — добавим новое
        new_msgs.append({"role": "user", "content": "Исправь ошибки и верни строгий JSON по схеме."})
        return new_msgs

    attempts = 0
    last_err: Optional[BaseException] = None
    while True:
        attempts += 1
        try:
            obj = await chat_json_schema(
                model=model_name,
                messages=messages,
                json_schema=as_json_schema(schema),
                temperature=temperature,
                timeout=timeout,
                log_prefix=log_prefix,
                log_dir=log_dir,
                top_p=top_p,
                seed=(None if seed is None else int(seed) + (attempts - 1)),
                inline_refs=inline_refs,
            )
        except Exception as e:
            last_err = e
            if attempts > retries:
                raise
            # Перепробуем без модификации сообщений — это ошибка транспорта/парсинга
            logger.warning("Retry %d/%d after error: %s", attempts, retries, e)
            continue

        # Валидация pydantic'ом (если применимо)
        try:
            if hasattr(schema, "model_validate"):
                obj_validated = schema.model_validate(obj)  # type: ignore[attr-defined]
            else:
                obj_validated = obj
        except Exception as e:
            last_err = e
            if attempts > retries:
                raise
            # подскажем схеме где больно
            try:
                logger.error("Schema validation failed: %s", e)
                logger.error("Offending object: %s", json.dumps(obj, ensure_ascii=False)[:1200])
            except Exception:
                pass
            messages = _with_retry_hint(messages, "объект не соответствует схеме")
            continue

        # Forbidden
        hits = _find_forbidden_hits(obj_validated, forbidden)
        if hits:
            if attempts > retries:
                raise RuntimeError(f"Forbidden content: {hits}")
            logger.warning("Forbidden hits %s — retrying", hits)
            messages = _with_retry_hint(messages, "обнаружены запрещённые слова: " + ", ".join(hits))
            continue

        # Пусть эффекты не пустые, если есть действия
        if require_nonempty_effects and actions_for_effects:
            try:
                if not _effects_has_delta(obj_validated):
                    if attempts > retries:
                        raise RuntimeError("Effects are empty while actions present")
                    logger.warning("Empty effects with actions — retrying")
                    messages = _with_retry_hint(messages, "нужна минимум одна осмысленная дельта effect")
                    continue
            except Exception:
                # если форма неожиданная — лучше пропустить проверку на этом шаге
                pass

        return obj_validated

# ===================== BACKWARD-COMPAT API =====================

async def request_llm_chat_json(
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    schema: Any,
    n_predict: Optional[int] = None,  # совместимость — игнорируем
    max_tokens: Optional[int] = None,  # совместимость — игнорируем
    temperature: float = 0.2,
    timeout: Optional[int] = None,
    timeout_s: Optional[int] = None,
    model: Optional[str] = None,
    cache_prompt: bool = False,   # совместимость — не используется
    log_prefix: Optional[str] = None,
    log_dir: Optional[str] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> Any:
    del cache_prompt, n_predict, max_tokens
    if messages is None:
        if not prompt:
            raise TypeError("request_llm_chat_json: provide either `messages` or `prompt`")
        messages = _build_messages(prompt)

    return await generate_json(
        model=model,
        schema=schema,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_retries=0,  # поведение как раньше: без автоперегенераций
        timeout=(timeout_s if timeout_s is not None else timeout),
        log_prefix=log_prefix,
        log_dir=log_dir,
        top_p=top_p,
        forbidden=None,
        require_nonempty_effects=False,
        actions_for_effects=None,
    )
