"""Two reusable pinned buffers with explicit cross-stream CUDA dependencies.

This utility is opt-in. The caller owns the consumer stream and must release the
slot after consumption, so the copy stream cannot overwrite a live input.
"""
from contextlib import contextmanager


class PinnedDoubleBuffer:
    def __init__(self, shape, dtype, device="cuda"):
        import torch
        self.torch = torch
        self.copy_stream = torch.cuda.Stream(device=device)
        self.host = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(2)]
        self.device = [torch.empty(shape, dtype=dtype, device=device) for _ in range(2)]
        self.done = [None, None]
        self.index = 0

    @contextmanager
    def transfer(self, array):
        torch = self.torch
        i = self.index
        self.index = 1 - i
        # CPU source buffer must not be overwritten before previous H2D completes.
        if self.done[i] is not None:
            self.done[i].synchronize()
        self.host[i].copy_(torch.as_tensor(array))
        consumer = torch.cuda.current_stream(self.device[i].device)
        with torch.cuda.stream(self.copy_stream):
            self.device[i].copy_(self.host[i], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        consumer.wait_event(ready)
        try:
            yield self.device[i]
        finally:
            self.done[i] = torch.cuda.Event()
            self.done[i].record(consumer)
