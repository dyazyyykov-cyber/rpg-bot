from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .utils import get_env

logger = logging.getLogger(__name__)
logger.propagate = True


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


LLM_URL = get_env("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL_DEFAULT = get_env("LLM_MODEL", "qwen3")
LLM_HTTP_TIMEOUT = int(get_env("LLM_HTTP_TIMEOUT", "480"))

LOG_LLM_FULL = str(get_env("LOG_LLM_FULL", "1")).strip().lower() in ("1", "true", "yes", "on")
LLM_LOG_CONTENT = str(get_env("LLM_LOG_CONTENT", "1")).strip().lower() in ("1", "true", "yes", "on")


def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        return None


def _ts() -> str:
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


async def _http_post_json(payload: Dict[str, Any], timeout: Optional[int]) -> Dict[str, Any]:
    effective_timeout: Optional[int]
    if timeout is None or (isinstance(timeout, (int, float)) and timeout <= 0):
        effective_timeout = LLM_HTTP_TIMEOUT
    else:
        effective_timeout = int(timeout)

    client_timeout = aiohttp.ClientTimeout(total=effective_timeout)

    dbg = {
        "url": LLM_URL,
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "seed": payload.get("seed"),
    }
    try:
        msgs = payload.get("messages", [])
        if isinstance(msgs, list) and msgs:
            preview: List[Dict[str, Any]] = []
            for msg in msgs[:2]:
                preview.append(
                    {
                        "role": msg.get("role"),
                        "content": str(msg.get("content", ""))[:180].replace("\n", "\\n"),
                    }
                )
            dbg["messages_preview"] = preview
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
                    logger.exception("LLM: invalid JSON in HTTP response body")
                    raise
    except asyncio.TimeoutError as exc:
        logger.warning(
            "LLM HTTP timeout after %ss (url=%s)",
            "∞" if effective_timeout is None else effective_timeout,
            LLM_URL,
        )
        raise LLMRequestTimeoutError(url=LLM_URL, timeout=effective_timeout) from exc


async def generate_text(
    *,
    prompt: str,
    system_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: Optional[int] = None,
    log_prefix: Optional[str] = None,
    log_dir: Optional[str] = None,
    seed: Optional[int] = None,
    extra_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Send a plain-text prompt to the chat completion endpoint."""

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model or LLM_MODEL_DEFAULT,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if seed is not None:
        try:
            payload["seed"] = int(seed)
        except Exception:
            payload["seed"] = seed

    req_id = _ts()
    if log_dir:
        _ensure_dir(log_dir)
        base = (log_prefix or "LLM").replace(":", "_")
        _dump_json(os.path.join(log_dir, f"{base}_request_{req_id}.json"), payload)

    data = await _http_post_json(payload, timeout=timeout)

    try:
        usage = data.get("usage") or {}
        logger.info("LLM USAGE[%s]: %s", log_prefix or "-", json.dumps(usage, ensure_ascii=False))
    except Exception:
        pass

    try:
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise RuntimeError(f"LLM response has no content: {str(data)[:600]}")

    if LLM_LOG_CONTENT:
        logger.info("LLM RAW content[%s]: %s", log_prefix or "-", content[:1200])

    if log_dir:
        base = (log_prefix or "LLM").replace(":", "_")
        _dump_json(os.path.join(log_dir, f"{base}_response_{req_id}.json"), data)
        with open(os.path.join(log_dir, f"{base}_text_{req_id}.txt"), "w", encoding="utf-8") as f:
            f.write(content)

    return content


def find_forbidden_hits(text: str, forbidden: Optional[List[str]]) -> List[str]:
    if not forbidden:
        return []
    hits: List[str] = []
    lowered = text.lower()
    for word in forbidden:
        if not word:
            continue
        if word.lower() in lowered:
            if word not in hits:
                hits.append(word)
    return hits


__all__ = [
    "generate_text",
    "find_forbidden_hits",
    "LLMRequestTimeoutError",
    "LLM_MODEL_DEFAULT",
]
