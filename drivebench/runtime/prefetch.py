"""Ordered background loading with backpressure and exception propagation."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def prefetch(items, load, depth=2):
    if depth <= 0:
        for item in items:
            yield load(item)
        return
    source = iter(items)
    with ThreadPoolExecutor(max_workers=min(depth, 4), thread_name_prefix="drivebench-load") as pool:
        pending = deque()
        for _ in range(depth):
            try:
                pending.append(pool.submit(load, next(source)))
            except StopIteration:
                break
        try:
            while pending:
                result = pending.popleft().result()
                yield result
                try:
                    pending.append(pool.submit(load, next(source)))
                except StopIteration:
                    pass
        finally:
            for future in pending:
                future.cancel()
