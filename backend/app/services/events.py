"""Event bus for Server-Sent Events: thread-safe publish, one asyncio queue per client."""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(eq=False)
class Subscriber:
    queue: asyncio.Queue
    kind: str  # "panel" | "team" | "history"


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"Not serializable: {type(obj)}")


def sse_message(event: str, data) -> str:
    payload = json.dumps(data, default=_json_default, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


class EventBus:
    def __init__(self) -> None:
        self._subs: set[Subscriber] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, kind: str = "panel") -> Subscriber:
        sub = Subscriber(asyncio.Queue(maxsize=200), kind)
        with self._lock:
            self._subs.add(sub)
        self.publish("clients", self.counts())
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)
        self.publish("clients", self.counts())

    def counts(self) -> dict:
        with self._lock:
            team = sum(1 for s in self._subs if s.kind == "team")
            panel = sum(1 for s in self._subs if s.kind == "panel")
            return {"team": team, "panel": panel}

    def publish(self, event: str, data) -> None:
        """Safe to call from any thread."""
        if self._loop is None or self._loop.is_closed():
            return
        msg = sse_message(event, data)
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            self._loop.call_soon_threadsafe(self._put, sub, msg)

    @staticmethod
    def _put(sub: Subscriber, msg: str) -> None:
        try:
            sub.queue.put_nowait(msg)
        except asyncio.QueueFull:
            log.warning("SSE client too slow, dropping message")
