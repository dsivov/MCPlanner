"""Speculative-work scheduler: bounded budget + critical-path preemption.

Adopted from PASTE's slack-scheduling design (Sui et al., 2026, arXiv:2603.18897):
speculative LLM work runs only on "slack" and within a bounded concurrency budget, so a
mispredicted speculation wastes at most a bounded amount of API work and never delays the
live turn.

In our system the contended resource is OpenAI API concurrency/rate. Background MCTS
pondering fires many rollout LLM calls; the live critical-path turn (classify + response
generation) must not be queued behind that speculative load. Two rules implement this:

  1. Bounded budget. At most ``SPECULATIVE_BUDGET`` speculative LLM calls run at once.
  2. Strict preemption. While any critical-path LLM call is in flight, NEW speculative
     calls are held; they proceed only once the critical path is idle. (An already
     in-flight HTTP request cannot be torn down mid-call, but the next-turn cancellation
     of pondering — see PonderingScheduler.cancel_all — handles that cleanup; the gate
     here prevents speculative calls from competing with the live turn for API budget.)

A call marks itself speculative via the ``speculative_mode`` ContextVar rather than a
threaded ``priority`` argument, so deep call chains (run_mcts -> rollouts) inherit it for
free: asyncio.create_task copies the current context, so setting the var at the top of a
pondering/instruction task propagates to every nested LLM call.
"""
from __future__ import annotations
import asyncio
import contextvars

from ..config import settings

# ContextVar: True inside a speculative (background) task tree, False on the live path.
speculative_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "speculative_mode", default=False
)


def is_speculative() -> bool:
    return speculative_mode.get()


class _State:
    """Process-wide gate state. Lazily binds asyncio primitives to the running loop."""

    def __init__(self) -> None:
        self._sem: asyncio.Semaphore | None = None
        self._idle: asyncio.Event | None = None
        self._critical_active = 0
        self._lock: asyncio.Lock | None = None
        # Lightweight telemetry for panels / analysis.
        self.spec_started = 0
        self.spec_held_for_critical = 0

    def _ensure(self) -> None:
        if self._sem is None:
            budget = max(1, int(getattr(settings, "SPECULATIVE_BUDGET", 4)))
            self._sem = asyncio.Semaphore(budget)
            self._idle = asyncio.Event()
            self._idle.set()  # idle until a critical op begins
            self._lock = asyncio.Lock()

    async def enter_critical(self) -> None:
        self._ensure()
        async with self._lock:  # type: ignore[union-attr]
            self._critical_active += 1
            self._idle.clear()  # type: ignore[union-attr]

    async def exit_critical(self) -> None:
        async with self._lock:  # type: ignore[union-attr]
            self._critical_active -= 1
            if self._critical_active <= 0:
                self._critical_active = 0
                self._idle.set()  # type: ignore[union-attr]

    async def acquire_speculative(self) -> None:
        self._ensure()
        # Preemption: don't start while the live path is using the LLM.
        if not self._idle.is_set():  # type: ignore[union-attr]
            self.spec_held_for_critical += 1
        await self._idle.wait()  # type: ignore[union-attr]
        await self._sem.acquire()  # type: ignore[union-attr]
        # If a critical op slipped in between the wait and the acquire, yield the slot and
        # wait again — keeps the live path strictly ahead of speculative work.
        while not self._idle.is_set():  # type: ignore[union-attr]
            self._sem.release()  # type: ignore[union-attr]
            await self._idle.wait()  # type: ignore[union-attr]
            await self._sem.acquire()  # type: ignore[union-attr]
        self.spec_started += 1

    def release_speculative(self) -> None:
        self._sem.release()  # type: ignore[union-attr]


_state = _State()


class critical_path:
    """Async context manager wrapping one critical-path LLM call. While held, new
    speculative calls are blocked from starting."""

    async def __aenter__(self) -> "critical_path":
        await _state.enter_critical()
        return self

    async def __aexit__(self, *exc) -> None:
        await _state.exit_critical()


class speculative_slot:
    """Async context manager wrapping one speculative LLM call: wait for the live path to
    be idle, then take a bounded budget slot."""

    async def __aenter__(self) -> "speculative_slot":
        await _state.acquire_speculative()
        return self

    async def __aexit__(self, *exc) -> None:
        _state.release_speculative()


def stats() -> dict:
    return {
        "budget": int(getattr(settings, "SPECULATIVE_BUDGET", 4)),
        "spec_started": _state.spec_started,
        "spec_held_for_critical": _state.spec_held_for_critical,
    }
