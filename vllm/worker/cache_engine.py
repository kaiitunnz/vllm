"""CacheEngine class for managing the KV cache."""
import enum
from queue import SimpleQueue
from threading import Semaphore, Thread
from typing import List, Optional, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)

logger = init_logger(__name__)


class CacheOp(enum.Enum):
    """Enumeration of cache operations."""
    SWAP_IN = enum.auto()
    SWAP_OUT = enum.auto()
    COPY = enum.auto()
    START = enum.auto()


class SwapManager:

    def __init__(self, wait_sema: Semaphore) -> None:
        self._wait_sema = wait_sema

    def wait(self):
        self._wait_sema.acquire()


_SwapMessage = Tuple[CacheOp, Optional[torch.Tensor]]


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    @classmethod
    def from_config(
        cls,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> "CacheEngine":
        if cache_config.enable_multi_tier_prefix_caching:
            return AsyncCacheEngine(cache_config, model_config,
                                    parallel_config, device_config)
        return cls(cache_config, model_config, parallel_config, device_config)

    @property
    def swap_manager(self) -> Optional[SwapManager]:
        return None

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    def begin_cache_ops(self) -> None:
        # No-op for synchronous cache engine.
        pass

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


class AsyncCacheEngine(CacheEngine):
    """Asynchronous version of CacheEngine."""

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config,
                         device_config)

        self._swap_queue: SimpleQueue[Optional[_SwapMessage]] = SimpleQueue()
        self._wait_sema: Semaphore = Semaphore(0)
        self._swap_manager = SwapManager(self._wait_sema)
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
    def swap_manager(self) -> Optional[SwapManager]:
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

    def begin_cache_ops(self) -> None:
        self._swap_queue.put((CacheOp.START, None))

    def __del__(self):
        self._swap_queue.put(None)
        self._swap_thread.join()


def _swap_thread_func(queue: SimpleQueue[Optional[_SwapMessage]],
                      wait_sema: Semaphore, num_attention_layers: int,
                      attn_backend: AttentionBackend,
                      gpu_cache: List[torch.Tensor],
                      cpu_cache: List[torch.Tensor]) -> None:
    stream = torch.cuda.Stream()
    to_swap_in = to_swap_out = to_copy = None
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
            elif op == CacheOp.START:
                if ((to_swap_in is None) and (to_swap_out is None)
                        and (to_copy is None)):
                    continue
                assert src_to_dst is None
                assert wait_sema._value == 0
                for i in range(num_attention_layers):
                    if to_swap_out is not None:
                        attn_backend.swap_blocks(gpu_cache[i], cpu_cache[i],
                                                 to_swap_out)
                    if to_swap_in is not None:
                        attn_backend.swap_blocks(cpu_cache[i], gpu_cache[i],
                                                 to_swap_in)
                    if to_copy is not None:
                        attn_backend.copy_blocks(gpu_cache, to_copy)
                    stream.record_event().synchronize()
                    wait_sema.release()
                to_swap_in = to_swap_out = to_copy = None
