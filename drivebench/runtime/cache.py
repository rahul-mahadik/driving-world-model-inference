"""Byte-bounded LRU for deterministic, action-independent prepared contexts."""
from collections import OrderedDict
from dataclasses import fields, is_dataclass

import numpy as np


def size_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        # A tiny slice keeps its whole backing allocation alive. Counting only
        # the slice's nbytes would allow a byte-bounded cache to retain gigabytes.
        if value.base is not None:
            return size_bytes(value.base, seen)
        return int(value.nbytes)
    if isinstance(value, memoryview):
        return size_bytes(value.obj, seen)
    if hasattr(value, "untyped_storage"):
        storage = value.untyped_storage()
        key = ("tensor-storage", str(value.device), storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(size_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(size_bytes(v, seen) for v in value)
    if is_dataclass(value):
        return sum(size_bytes(getattr(value, f.name), seen) for f in fields(value))
    if isinstance(value, (str, bytes)):
        return len(value)
    return 0


class ContextCache:
    def __init__(self, max_bytes=0):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cache byte budget must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.items = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        item = self.items.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        self.items.move_to_end(key)
        return item[0]

    def put(self, key, value):
        count = size_bytes(value)
        if key in self.items:
            self.bytes -= self.items.pop(key)[1]
        if self.max_bytes == 0 or count > self.max_bytes:
            return
        while self.items and self.bytes + count > self.max_bytes:
            _, (_, old_size) = self.items.popitem(last=False)
            self.bytes -= old_size
            self.evictions += 1
        self.items[key] = (value, count)
        self.bytes += count

    def clear(self):
        self.items.clear()
        self.bytes = 0

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "bytes": self.bytes, "entries": len(self.items), "hit_rate": self.hits / total if total else 0.0}
