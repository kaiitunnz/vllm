from logging import Logger
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from vllm.core.block.common import MTCacheMetricData
from vllm.core.block.interfaces import (Block, BlockId, BlockState,
                                        EvictedBlockMetaData)
from vllm.core.block.mt_interfaces import (MTAllocationOutput,
                                           MTBlockAllocator,
                                           MTDeviceAwareBlockAllocator)
from vllm.core.block.mt_prefix_caching_block import (
    MTPrefixCachingBlockAllocator, PrefixCache)
from vllm.utils import Device


class BlockMover:
    Entry = Tuple[Device, int]
    Record = Dict[Entry, Entry]
    PLACEHOLDER: Entry = (Device.CPU, -1)

    def __init__(self):
        self._record: BlockMover.Record = {}
        self._pending: BlockMover.Record = {}

    def move(self, src: "BlockMover.Entry",
             dst: Optional["BlockMover.Entry"]) -> None:
        if dst is BlockMover.PLACEHOLDER:
            assert src not in self._pending
            self._pending[src] = self._record.pop(src, BlockMover.PLACEHOLDER)
        elif dst is not None:
            src_record = self._pending if src in self._pending else self._record
            original_src = src_record.pop(src, BlockMover.PLACEHOLDER)
            self._record[dst] = (src if original_src is BlockMover.PLACEHOLDER
                                 else original_src)
            assert dst[0] != self._record[dst][0]
        else:
            self._record.pop(src, None)
            self._pending.pop(src, None)

    def get_and_reset_record(self) -> "BlockMover.Record":
        assert len(self._pending) == 0
        record, self._record = self._record, {}
        return record


# class BlockMover:
#     Entry = Tuple[Device, int]
#     Record = Dict[Entry, Entry]
#     PLACEHOLDER: Entry = (Device.CPU, -1)

#     def __init__(self):
#         self._record: Dict[BlockMover.Entry, List[BlockMover.Entry]] = {}
#         self._pending: Dict[BlockMover.Entry, List[BlockMover.Entry]] = {}

#     def move(self, src: "BlockMover.Entry",
#              dst: Optional["BlockMover.Entry"]) -> None:
#         if dst == BlockMover.PLACEHOLDER:
#             assert src not in self._pending
#             self._pending[src] = self._record.pop(src, [])
#         elif dst is not None:
#             src_record = (self._pending
#                           if src in self._pending else self._record)
#             original_src = src_record.pop(src, [])
#             original_src.append(src)
#             self._record[dst] = original_src
#             assert dst[0] != original_src[0][0], (
#                 f"[noppanat] src={src}, "
#                 f"dst={dst}, "
#                 f"original_src={original_src}"
#             )
#         else:
#             self._record.pop(src, None)
#             self._pending.pop(src, None)

#     def get_and_reset_record(self) -> "BlockMover.Record":
#         assert len(self._pending) == 0
#         record = {k: v[0] for k, v in self._record.items()}
#         self._record = {}
#         return record

#     @property
#     def record(self) -> Dict["BlockMover.Entry", List["BlockMover.Entry"]]:
#         return self._record


class MTPrefixAwareBlockAllocator(MTDeviceAwareBlockAllocator):
    """A block allocator that can allocate blocks on both CPU and GPU memory.

    This class implements the `DeviceAwareBlockAllocator` interface and provides
    functionality for allocating and managing blocks of memory on both CPU and
    GPU devices.

    The `MTBlockAllocator` maintains separate memory pools for CPU and GPU
    blocks, and allows for allocation, deallocation, forking, and swapping of
    blocks across these memory pools.
    """

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> MTDeviceAwareBlockAllocator:
        """Creates a MTBlockAllocator instance with the specified
        configuration.

        This static method creates and returns a MTBlockAllocator instance
        based on the provided parameters. It initializes the CPU and GPU block
        allocators with the specified number of blocks, block size, and
        allocator type.

        Args:
            allocator_type (str): The type of block allocator to use for CPU
                and GPU blocks. Currently supported values are "naive" and
                "prefix_caching".
            num_gpu_blocks (int): The number of blocks to allocate for GPU
                memory.
            num_cpu_blocks (int): The number of blocks to allocate for CPU
                memory.
            block_size (int): The size of each block in number of tokens.

        Returns:
            DeviceAwareBlockAllocator: A MTBlockAllocator instance with the
                specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        prefix_cache = PrefixCache()

        metric_data = MTCacheMetricData([Device.GPU, Device.CPU])

        if allocator_type == "prefix_caching":
            gpu_allocator = MTPrefixCachingBlockAllocator(
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                metric_data=metric_data.for_device(Device.GPU),
                block_ids=gpu_block_ids,
                hit_count_threshold=1,
                prefix_cache=prefix_cache,
            )

            cpu_allocator = MTPrefixCachingBlockAllocator(
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                metric_data=metric_data.for_device(Device.CPU),
                block_ids=cpu_block_ids,
                prefix_cache=prefix_cache,
                hit_count_threshold=10,
                block_pool=gpu_allocator.block_pool,
            )
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return MTPrefixAwareBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
            prefix_cache=prefix_cache,
        )

    def __init__(
        self,
        cpu_block_allocator: MTBlockAllocator,
        gpu_block_allocator: MTBlockAllocator,
        prefix_cache: PrefixCache,
    ):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        self._device_tier = [Device.GPU, Device.CPU]  # Highest tier first.
        self._allocators = {
            Device.CPU: cpu_block_allocator,
            Device.GPU: gpu_block_allocator,
        }
        self._prefix_cache = prefix_cache

        self._swap_mapping: Dict[int, int] = {}
        self._block_mover = BlockMover()
        self._null_block: Optional[Block] = None

        self._block_ids_to_allocator: Dict[int, MTBlockAllocator] = {}
        self._block_ids_to_device: Dict[int, Device] = {}
        self._block_ids_to_device_tier: Dict[int, int] = {}
        for device, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator
                self._block_ids_to_device[block_id] = device
                self._block_ids_to_device_tier[block_id] = (
                    self._device_tier.index(device))

    def allocate_or_get_null_block(self) -> Block:
        if self._null_block is None:
            self._null_block = NullBlock(
                self.allocate_mutable_block(None, Device.GPU).block)
        return self._null_block

    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        device: Device,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> MTAllocationOutput:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (Device): The device on which to allocate the new block.

        Returns:
            AllocationOutput: Output of successful allocation.
        """
        return self._allocators[device].allocate_mutable_block(
            prev_block, block_ids_in_use=block_ids_in_use)

    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: Device,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> List[MTAllocationOutput]:
        """Allocates a new group of immutable blocks with the provided block
        token IDs on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            block_token_ids (List[int]): The list of block token IDs to be
                stored in the new blocks.
            device (Device): The device on which to allocate the new block.

        Returns:
            List[Block]: The newly allocated list of allocation outputs.
        """
        return list(self._allocators[device].allocate_immutable_blocks(
            prev_block, block_token_ids, block_ids_in_use=block_ids_in_use))

    def allocate_immutable_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: Device,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> MTAllocationOutput:
        """Allocates a new immutable block with the provided token IDs on the
        specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            token_ids (List[int]): The list of token IDs to be stored in the new
                block.
            device (Device): The device on which to allocate the new block.

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """

        return self._allocators[device].allocate_immutable_block(
            prev_block, token_ids, block_ids_in_use=block_ids_in_use)

    def allocate_cached_block(self, block: Block) -> MTAllocationOutput:
        return self._allocators[self.get_device(block)].allocate_cached_block(
            block)

    def allocate_placeholder_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        content_hash: Optional[int] = None,
    ) -> Block:
        # Use the top-tier device's allocator to allocate placeholder blocks.
        return (
            self._allocators[self._device_tier[0]].allocate_placeholder_block(
                prev_block, token_ids, content_hash=content_hash))

    def promote_placeholder_block(
        self,
        block: Block,
        device: Device,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> MTAllocationOutput:
        # Placeholder blocks are allocated by the top-tier device's allocator.
        assert device == self._device_tier[0]
        return (
            self._allocators[self._device_tier[0]].promote_placeholder_block(
                block, block_ids_in_use))

    def free(self, block: Block) -> None:
        """Frees the memory occupied by the given block.

        Args:
            block (Block): The block to be freed.
        """
        # Null block should never be freed
        if isinstance(block, NullBlock):
            return
        block_id = block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        allocator.free(block)

    def fork(self, last_block: Block) -> List[MTAllocationOutput]:
        """Creates a new sequence of blocks that shares the same underlying
            memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: A new list of blocks that shares the same memory as the
                original sequence.
        """
        # do not attempt to fork the null block
        raise NotImplementedError("Forking is not supported.")
        assert not isinstance(last_block, NullBlock)
        block_id = last_block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        return list(allocator.fork(last_block))

    def get_num_free_blocks(
            self,
            device: Device,
            block_ids_in_use: Optional[Set[int]] = None) -> int:
        """Returns the number of free blocks available on the specified device.

        Args:
            device (Device): The device for which to query the number of free
                blocks. AssertionError is raised if None is passed.

        Returns:
            int: The number of free blocks available on the specified device.
        """
        return self._allocators[device].get_num_free_blocks(block_ids_in_use)

    def get_num_total_blocks(self, device: Device) -> int:
        return self._allocators[device].get_num_total_blocks()

    def get_physical_block_id(self, device: Device, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain device given the
        absolute block id.

        Args:
            device (Device): The device for which to query relative block id.
                absolute_id (int): The absolute block id for the block in
                whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return self._allocators[device].get_physical_block_id(absolute_id)

    def _move(self,
              block: Block,
              src_device: Device,
              dst_device: Optional[Device],
              hit_count: int = 0,
              block_ids_in_use: Optional[Set[int]] = None,
              evictable: bool = False) -> Optional[EvictedBlockMetaData]:
        """
        Returns:
            Optional[Block]: The evicted block if any.
        """
        src_block_id = block.block_id
        assert src_block_id is not None
        assert src_device != dst_device

        if dst_device is None:
            self._allocators[src_device].destroy(block)
            self._block_mover.move((src_device, src_block_id), None)
            return None

        self._allocators[src_device].move_out(block, cache_hit=not evictable)
        alloc = self._allocators[dst_device].move_in(
            block,
            hit_count=hit_count,
            block_ids_in_use=block_ids_in_use,
            evictable=evictable)

        dst_block_id = alloc.block.block_id
        assert block_ids_in_use is None or dst_block_id not in block_ids_in_use
        assert dst_block_id is not None
        src_block_id = self._allocators[src_device].get_physical_block_id(
            src_block_id)
        dst_block_id = self._allocators[dst_device].get_physical_block_id(
            dst_block_id)

        # Move out first.
        if alloc.evicted_meta is not None:
            evicted_block_id = alloc.evicted_meta.block.block_id
            assert evicted_block_id is not None
            self._block_mover.move((dst_device, evicted_block_id),
                                   BlockMover.PLACEHOLDER)
        self._block_mover.move((src_device, src_block_id),
                               (dst_device, dst_block_id))

        return alloc.evicted_meta

    def move_in(self,
                blocks: List[Block],
                block_ids_in_use: Optional[Set[int]] = None) -> None:
        """Move in the given blocks to the top-tier device.

        Args:
            blocks: List of blocks to move in.
        """
        blocks_to_move_out: List[EvictedBlockMetaData] = []
        dst_device = self._device_tier[0]

        for block in blocks:
            assert block.content_hash is not None
            assert block.content_hash in self._prefix_cache

            src_device = self.get_device(block)
            if src_device == dst_device:
                # The block has already been moved in by other sequences.
                # Must be done to increment the ref count.
                self._allocators[dst_device].allocate_cached_block(block)
                continue
            # Reset the hit count when moving in the top-tier device.
            evicted_meta = self._move(block,
                                      src_device,
                                      dst_device,
                                      hit_count=0,
                                      block_ids_in_use=block_ids_in_use,
                                      evictable=False)

            if evicted_meta is not None:
                blocks_to_move_out.append(evicted_meta)

        # Move out the evicted blocks down the device tier.
        self.move_out(blocks_to_move_out, block_ids_in_use=block_ids_in_use)

    def move_out(self,
                 blocks: List[EvictedBlockMetaData],
                 block_ids_in_use: Optional[Set[int]] = None) -> None:
        """Move the given blocks down the device tier.
        
        Args:
            blocks: List of blocks to move out.
        """
        block_metas = sorted(blocks,
                             key=lambda meta: self.get_device_tier(meta.block),
                             reverse=True)
        assert all(
            self.get_device(meta.block) == self._device_tier[0]
            for meta in blocks)

        next_device_tier = self._device_tier[1:]
        for meta in block_metas:
            cur_meta: Optional[EvictedBlockMetaData] = meta
            for cur_device, next_device in zip(self._device_tier,
                                               next_device_tier):
                assert cur_meta is not None
                cur_meta = self._move(cur_meta.block,
                                      cur_device,
                                      next_device,
                                      hit_count=cur_meta.hit_count,
                                      block_ids_in_use=block_ids_in_use,
                                      evictable=True)
                if cur_meta is None:
                    break

            if cur_meta is not None:
                # Move out of the last device
                evicted_block = self._move(cur_meta.block,
                                           self._device_tier[-1], None)
                assert evicted_block is None

    def swap(self, blocks: List[Block], src_device: Device,
             dst_device: Device) -> Dict[int, int]:
        """Execute the swap for the given blocks from source_device
        on to dest_device, save the current swap mapping and append
        them to the accumulated `self._swap_mapping` for each
        scheduling move.

        Args:
            blocks: List of blocks to be swapped.
            src_device (Device): Device to swap the 'blocks' from.
            dst_device (Device): Device to swap the 'blocks' to.

        Returns:
            Dict[int, int]: Swap mapping from source_device
                on to dest_device.
        """
        raise NotImplementedError("Swap is not yet supported.")

    def get_device(self, block: Block) -> Device:
        assert block.block_id is not None
        return self._block_ids_to_device[block.block_id]

    def get_device_from_id(self, block_id: int) -> Device:
        return self._block_ids_to_device[block_id]

    def get_device_tier(self, block: Block) -> int:
        assert block.block_id is not None
        return self._block_ids_to_device_tier[block.block_id]

    def get_device_tier_from_id(self, block_id: int) -> int:
        return self._block_ids_to_device_tier[block_id]

    def get_num_full_blocks_touched(self, blocks: List[Block],
                                    device: Device) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out the given blocks on to the 'device'.

        Args:
            blocks: List of blocks to be swapped.
            device (Device): Device to swap the 'blocks' on.

        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks on to the 'device'.
                Non full blocks are ignored when deciding the number
                of blocks to touch.
        """
        return self._allocators[device].get_num_full_blocks_touched(blocks)

    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        """Clears the copy-on-write (CoW) state and returns the mapping of
            source to destination block IDs.

        Returns:
            List[Tuple[int, int]]: A list mapping source block IDs to
                destination block IDs.
        """
        # CoW only supported on GPU
        device = Device.GPU
        return self._allocators[device].clear_copy_on_writes()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, only use for prefix caching."""
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].mark_blocks_as_accessed(block_ids, now)

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as accessed, only use for prefix caching."""
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].mark_blocks_as_computed(block_ids)

    def get_computed_block_ids(
        self,
        prev_computed_block_ids: List[int],
        block_ids: List[int],
        skip_last_block_id: bool,
    ) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_computed_block_ids(
            prev_computed_block_ids, block_ids, skip_last_block_id)

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        # Prefix caching only supported on GPU.
        device = Device.GPU
        return self._allocators[device].get_common_computed_block_ids(
            computed_seq_block_ids)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        assert device in self._allocators
        return self._allocators[device].get_prefix_cache_hit_rate()

    def get_and_reset_swaps(self) -> List[Tuple[int, int]]:
        """Returns and clears the mapping of source to destination block IDs.
        Will be called after every swapping operations for now, and after every
        schedule when BlockManagerV2 become default. Currently not useful.

        Returns:
            List[Tuple[int, int]]: A mapping of source to destination block IDs.
        """
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return list(mapping.items())

    def get_and_reset_block_moving_record(
        self, ) -> Dict[Tuple[Device, int], Tuple[Device, int]]:
        return self._block_mover.get_and_reset_record()

    def get_cached_block(self, content_hash: int) -> Optional[Block]:
        return self._prefix_cache.get(content_hash)

    def destroy(self, block: Block) -> None:
        device = (self._device_tier[0] if block.state == BlockState.PLACEHOLDER
                  else self.get_device(block))
        self._allocators[device].destroy(block)

    def print_content(self, logger: Logger):
        # TODO(noppanat): Remove this.
        for device, allocator in self._allocators.items():
            logger.info("[noppanat] Device: %s, Cached blocks: %s", device,
                        allocator)  # type: ignore


class NullBlock(Block):
    """
    Null blocks are used as a placeholders for KV cache blocks that have
    been dropped due to sliding window.
    This implementation just wraps an ordinary block and prevents it from
    being modified. It also allows for testing if a block is NullBlock
    via isinstance().
    """

    def __init__(self, proxy: Block):
        super().__init__()
        self._proxy = proxy

    def append_token_ids(self, token_ids: List[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def block_id(self):
        return self._proxy.block_id

    @block_id.setter
    def block_id(self, value: Optional[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def token_ids(self) -> List[BlockId]:
        return self._proxy.token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError(
            "num_tokens_total is not used for null block")

    @property
    def num_empty_slots(self) -> BlockId:
        return self._proxy.num_empty_slots

    @property
    def is_full(self):
        return self._proxy.is_full

    @property
    def prev_block(self):
        return self._proxy.prev_block

    @property
    def computed(self):
        return self._proxy.computed

    @computed.setter
    def computed(self, value):
        self._proxy.computed = value

    @property
    def last_accessed(self) -> float:
        return self._proxy.last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        self._proxy.last_accessed = last_accessed_ts

    @property
    def content_hash(self):
        return self._proxy.content_hash
