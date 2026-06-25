from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from openai import AsyncOpenAI
from ..config import settings
from ..logger import ExperimentLogger, LLMCallEntry
from .prompts import FRAMEWORK_PREAMBLE
from .scheduler import is_speculative, critical_path, speculative_slot


def _with_preamble(system: str, *, use_preamble: bool) -> str:
    if not use_preamble:
        return system
    return FRAMEWORK_PREAMBLE + "\n\n---\n\n" + system


def _dispatch_gate():
    """Pick the scheduler gate for this call based on the speculative ContextVar.
    Speculative calls wait for slack + a budget slot; critical calls hold the gate."""
    return speculative_slot() if is_speculative() else critical_path()


_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0


def _record(
    logger: Optional[ExperimentLogger],
    *,
    call_site: str,
    model: str,
    started_at: datetime,
    duration_ms: int,
    tokens_in: int,
    tokens_out: int,
    temperature: Optional[float],
    max_tokens: Optional[int],
    system: str,
    user: str,
    response_text: str,
    response_json: Optional[dict[str, Any]],
    is_json_mode: bool,
    ok: bool,
    error: Optional[str] = None,
) -> None:
    if logger is None:
        return
    logger.record_llm_call(LLMCallEntry(
        call_site=call_site or "unknown",
        model=model,
        started_at=started_at,
        duration_ms=duration_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system,
        user_prompt=user,
        response_text=response_text,
        response_json=response_json,
        is_json_mode=is_json_mode,
        ok=ok,
        error=error,
    ))


async def chat(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 800,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "unknown",
    use_preamble: bool = True,
) -> LLMResult:
    started_at = datetime.utcnow()
    t0 = time.perf_counter()
    text = ""
    tokens_in = 0
    tokens_out = 0
    err: Optional[str] = None
    ok = True
    full_system = _with_preamble(system, use_preamble=use_preamble)
    try:
        async with _dispatch_gate():
            r = await client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        text = r.choices[0].message.content or ""
        usage = r.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        _record(
            logger,
            call_site=call_site, model=model, started_at=started_at, duration_ms=duration_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
            temperature=temperature, max_tokens=max_tokens,
            system=full_system, user=user,
            response_text=text, response_json=None, is_json_mode=False,
            ok=ok, error=err,
        )
    return LLMResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out, duration_ms=duration_ms)


async def chat_json(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "unknown",
    use_preamble: bool = True,
) -> tuple[dict[str, Any], LLMResult]:
    started_at = datetime.utcnow()
    t0 = time.perf_counter()
    content = "{}"
    parsed: dict[str, Any] = {}
    tokens_in = 0
    tokens_out = 0
    err: Optional[str] = None
    ok = True
    full_system = _with_preamble(system, use_preamble=use_preamble)
    try:
        async with _dispatch_gate():
            r = await client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        content = r.choices[0].message.content or "{}"
        usage = r.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as je:
            err = f"JSONDecodeError: {je}"
            parsed = {}
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        _record(
            logger,
            call_site=call_site, model=model, started_at=started_at, duration_ms=duration_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
            temperature=temperature, max_tokens=max_tokens,
            system=full_system, user=user,
            response_text=content, response_json=parsed if parsed else None, is_json_mode=True,
            ok=ok, error=err,
        )
    res = LLMResult(text=content, tokens_in=tokens_in, tokens_out=tokens_out, duration_ms=duration_ms)
    return parsed, res
