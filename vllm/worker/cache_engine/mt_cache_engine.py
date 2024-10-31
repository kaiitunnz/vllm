"""CacheEngine class for managing the multi-tier KV cache."""
from queue import SimpleQueue
from threading import Semaphore, Thread
from typing import List, Optional, Tuple

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.worker.cache_engine.cache_engine import CacheEngine
from vllm.worker.cache_engine.cache_engine import CacheOp, SwapManagerBase


class MTSwapManager(SwapManagerBase):

    def __init__(self, stream: torch.cuda.Stream,
                 num_attention_layers: int) -> None:
        self._stream = stream
        self._num_attention_layers = num_attention_layers
        self._swap_events = [
            torch.cuda.Event() for _ in range(num_attention_layers)
        ]
        self._counter: Optional[int] = None

    def wait(self) -> None:
        assert self._counter is not None
        self._swap_events[self._counter].synchronize()
        self._counter += 1
        if self._counter >= self._num_attention_layers:
            self._counter = None

    def record_event(self):
        if self._counter is None:
            self._counter = 0
        self._stream.record_event(self._swap_events[self._counter])
        self._counter = (self._counter + 1) % self._num_attention_layers

    @property
    def is_active(self) -> bool:
        return self._counter is not None


class MTAsyncSwapManager(SwapManagerBase):
    Message = Tuple[CacheOp, Optional[torch.Tensor]]

    def __init__(self, wait_sema: Semaphore) -> None:
        self._wait_sema = wait_sema

    def wait(self):
        self._wait_sema.acquire()


class MTCacheEngine(CacheEngine):
    """Multi-tier KV cache engine."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config,
                         device_config)

        self._stream = torch.cuda.Stream()
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self._swap_manager = MTSwapManager(self._stream,
                                           self.num_attention_layers)

        self._to_swap_in: Optional[torch.Tensor] = None
        self._to_swap_out: Optional[torch.Tensor] = None
        self._to_copy: Optional[torch.Tensor] = None
        self._to_prefetch: Optional[torch.Tensor] = None
        self._to_unload: Optional[torch.Tensor] = None

        self._do_cache_ops: bool = False

    @property
    def swap_manager(self) -> Optional[MTSwapManager]:
        if self._swap_manager.is_active:
            return self._swap_manager
        return None

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        assert self._to_swap_in is None
        self._to_swap_in = src_to_dst
        self._do_cache_ops = True

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        assert self._to_swap_out is None
        self._to_swap_out = src_to_dst
        self._do_cache_ops = True

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        assert self._to_copy is None
        self._to_copy = src_to_dsts
        self._do_cache_ops = True

    def prefetch(self, src_to_dst: torch.Tensor) -> None:
        assert self._to_prefetch is None
        self._to_prefetch = src_to_dst
        self._do_cache_ops = True

    def unload(self, src_to_dst: torch.Tensor) -> None:
        assert self._to_unload is None
        self._to_unload = src_to_dst
        self._do_cache_ops = True

    def begin_cache_ops(self) -> None:
        assert not self._swap_manager.is_active
        do_cache_ops, self._do_cache_ops = self._do_cache_ops, False

        if not do_cache_ops:
            return

        stream = self._stream
        attn_backend = self.attn_backend
        gpu_cache = self.gpu_cache
        cpu_cache = self.cpu_cache

        prefetch_event, self._prefetch_event = self._prefetch_event, None
        to_swap_in, self._to_swap_in = self._to_swap_in, None
        to_swap_out, self._to_swap_out = self._to_swap_out, None
        to_copy, self._to_copy = self._to_copy, None
        to_prefetch, self._to_prefetch = self._to_prefetch, None
        to_unload, self._to_unload = self._to_unload, None

        with torch.cuda.stream(stream):
            if prefetch_event is not None:
                prefetch_event.synchronize()
            if not ((to_swap_in is None) and (to_swap_out is None) and
                    (to_copy is None)):
                for i in range(self.num_attention_layers):
                    if to_swap_out is not None:
                        attn_backend.swap_blocks(gpu_cache[i], cpu_cache[i],
                                                 to_swap_out)
                    if to_swap_in is not None:
                        attn_backend.swap_blocks(cpu_cache[i], gpu_cache[i],
                                                 to_swap_in)
                    if to_copy is not None:
                        attn_backend.copy_blocks(gpu_cache, to_copy)
                    self._swap_manager.record_event()
            if not ((to_prefetch is None) and (to_unload is None)):
                for i in range(self.num_attention_layers):
                    if to_unload is not None:
                        attn_backend.swap_blocks(gpu_cache[i], cpu_cache[i],
                                                 to_unload)
                    if to_prefetch is not None:
                        attn_backend.swap_blocks(cpu_cache[i], gpu_cache[i],
                                                 to_prefetch)
                self._prefetch_event = stream.record_event()


class MTAsyncCacheEngine(CacheEngine):
    """Asynchronous version of multi-tier KV cache engine."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config,
                         device_config)

        self._swap_queue: SimpleQueue[Optional[MTAsyncSwapManager.Message]] = (
            SimpleQueue())
        self._wait_sema: Semaphore = Semaphore(0)
        self._swap_manager = MTAsyncSwapManager(self._wait_sema)
        self._swap_thread = Thread(
            target=_swap_thread_func,
            kwargs=dict(queue=self._swap_queue,
                        wait_sema=self._wait_sema,
                        num_attention_layers=self.num_attention_layers,
                        attn_backend=self.attn_backend,
                        gpu_cache=self.gpu_cache,
                        cpu_cache=self.cpu_cache))
        self._start_swap_thread()

        self._do_cache_ops = False

    @property
    def swap_manager(self) -> Optional[MTAsyncSwapManager]:
        do_cache_ops, self._do_cache_ops = self._do_cache_ops, False
        if do_cache_ops:
            return self._swap_manager
        return None

    def _start_swap_thread(self) -> None:
        self._swap_thread.daemon = True
        self._swap_thread.start()

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self._swap_queue.put((CacheOp.SWAP_IN, src_to_dst))
        self._do_cache_ops = True

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self._swap_queue.put((CacheOp.SWAP_OUT, src_to_dst))
        self._do_cache_ops = True

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self._swap_queue.put((CacheOp.COPY, src_to_dsts))
        self._do_cache_ops = True

    def prefetch(self, src_to_dst: torch.Tensor) -> None:
        self._swap_queue.put((CacheOp.PREFETCH, src_to_dst))
        # Not set _do_cache_ops to True, because prefetching is not a necessary
        # cache operation.

    def unload(self, src_to_dst: torch.Tensor) -> None:
        self._swap_queue.put((CacheOp.UNLOAD, src_to_dst))
        # Not set _do_cache_ops to True, because unloading is not a necessary
        # cache operation.

    def begin_cache_ops(self) -> None:
        self._swap_queue.put((CacheOp.START, None))

    def __del__(self):
        self._swap_queue.put(None)
        self._swap_thread.join()


def _swap_thread_func(queue: SimpleQueue[Optional[MTAsyncSwapManager.Message]],
                      wait_sema: Semaphore, num_attention_layers: int,
                      attn_backend: AttentionBackend,
                      gpu_cache: List[torch.Tensor],
                      cpu_cache: List[torch.Tensor]) -> None:
    stream = torch.cuda.Stream()
    pending_event: Optional[torch.cuda.Event] = None
    to_swap_in = to_swap_out = to_copy = to_prefetch = to_unload = None

    with torch.cuda.stream(stream):
        while True:
            msg = queue.get()
            if msg is None:
                break
            op, src_to_dst = msg
            if op == CacheOp.SWAP_IN:
                assert to_swap_in is None and src_to_dst is not None
                to_swap_in = src_to_dst
            elif op == CacheOp.SWAP_OUT:
                assert to_swap_out is None and src_to_dst is not None
                to_swap_out = src_to_dst
            elif op == CacheOp.COPY:
                assert to_copy is None and src_to_dst is not None
                to_copy = src_to_dst
            elif op == CacheOp.PREFETCH:
                assert to_prefetch is None and src_to_dst is not None
                to_prefetch = src_to_dst
            elif op == CacheOp.UNLOAD:
                assert to_unload is None and src_to_dst is not None
                to_unload = src_to_dst
            elif op == CacheOp.START:
                if pending_event is not None:
                    pending_event.synchronize()
                    pending_event = None
                if not ((to_swap_in is None) and (to_swap_out is None) and
                        (to_copy is None)):
                    assert src_to_dst is None
                    assert wait_sema._value == 0
                    for i in range(num_attention_layers):
                        if to_swap_out is not None:
                            attn_backend.swap_blocks(gpu_cache[i],
                                                     cpu_cache[i], to_swap_out)
                        if to_swap_in is not None:
                            attn_backend.swap_blocks(cpu_cache[i],
                                                     gpu_cache[i], to_swap_in)
                        if to_copy is not None:
                            attn_backend.copy_blocks(gpu_cache, to_copy)
                        stream.record_event().synchronize()
                        wait_sema.release()
                    to_swap_in = to_swap_out = to_copy = None
                if not ((to_prefetch is None) and (to_unload is None)):
                    assert src_to_dst is None
                    for i in range(num_attention_layers):
                        if to_unload is not None:
                            attn_backend.swap_blocks(gpu_cache[i],
                                                     cpu_cache[i], to_unload)
                        if to_prefetch is not None:
                            attn_backend.swap_blocks(cpu_cache[i],
                                                     gpu_cache[i], to_prefetch)
                    pending_event = stream.record_event()
                    to_prefetch = to_unload = None
