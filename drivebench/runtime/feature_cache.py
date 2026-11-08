"""Fixed-interval approximate block reuse with explicit candidate/CFG scopes.

No torch import is needed until the backend actually executes a model. Cache
entries never survive a candidate. A scope denotes the complete ordered CFG
batch, so unconditional and conditional features retain their separate slots.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator


class FixedIntervalFeatureCache:
    def __init__(self, interval: int = 1):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("Feature cache interval must be a positive integer")
        self.interval = interval
        self.hits = 0
        self.misses = 0
        self._scope: Any = None
        self._entries: dict[tuple[Any, ...], tuple[int, Any]] = {}

    @contextmanager
    def candidate(self, candidate_id: Any) -> Iterator["FixedIntervalFeatureCache"]:
        if self._scope is not None:
            raise RuntimeError("Feature cache candidate scopes cannot be nested or shared concurrently")
        self._scope = candidate_id
        self._entries.clear()
        try:
            yield self
        finally:
            self._entries.clear()
            self._scope = None

    def evaluate(
        self, *, block: str, step: int, cfg_scope: str, signature: tuple[Any, ...],
        compute: Callable[[], Any],
    ) -> Any:
        if self._scope is None:
            raise RuntimeError("Feature reuse requires an active candidate scope")
        if step < 0:
            raise ValueError("Denoiser evaluation index cannot be negative")
        key = (self._scope, cfg_scope, block, signature)
        previous = self._entries.get(key)
        if previous is not None and step > previous[0] and step % self.interval != 0:
            self.hits += 1
            return _clone(previous[1])
        value = compute()
        self.misses += 1
        self._entries[key] = (step, _clone(value))
        return value

    @property
    def entry_count(self) -> int:
        return len(self._entries)


def _clone(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if hasattr(value, "copy"):
        return value.copy()
    return value
