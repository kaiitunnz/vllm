from abc import ABC, abstractmethod
from typing import Optional, OrderedDict, Set

from vllm.core.block.interfaces import Block, BlockState, EvictedBlockMetaData
from vllm.core.evictor_v2 import EvictionPolicy


class MTEvictionError(Exception):
    pass


class MTEvictor(ABC):
    """The MTEvictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(
            self,
            blocks_in_use: Optional[Set[int]] = None) -> EvictedBlockMetaData:
        """Runs the eviction algorithm and returns the evicted block
        """
        pass

    @abstractmethod
    def add(self, block: Block, last_accessed: float, hit_count: int):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int) -> EvictedBlockMetaData:
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_blocks(self,
                            blocks_id_in_use: Optional[Set[int]] = None
                            ) -> int:
        pass


class LRUMTEvictor(MTEvictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: OrderedDict[int, EvictedBlockMetaData] = OrderedDict()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(
            self,
            block_ids_in_use: Optional[Set[int]] = None
    ) -> EvictedBlockMetaData:
        if len(self.free_table) == 0:
            raise MTEvictionError("No usable cache memory left")

        block_ids_in_use = set(
        ) if block_ids_in_use is None else block_ids_in_use
        evicted_meta, evicted_id = None, None
        # The blocks with the lowest timestamps should be placed consecutively
        # at the start of OrderedDict. Loop through all these blocks to
        # find the one with maximum number of hashed tokens.
        for block_id, block_meta in self.free_table.items():
            if block_id in block_ids_in_use:
                continue
            if evicted_meta is None:
                evicted_meta, evicted_id = block_meta, block_id
                continue
            if evicted_meta.last_accessed < block_meta.last_accessed:
                break
            if evicted_meta.num_hashed_tokens < block_meta.num_hashed_tokens:
                evicted_meta, evicted_id = block_meta, block_id

        assert evicted_meta is not None
        assert evicted_id is not None
        evicted_block = evicted_meta.block
        self.free_table.pop(evicted_id)

        evicted_block.block_id = evicted_id
        assert evicted_block.state == BlockState.FREED
        evicted_block.set_state(BlockState.EVICTED)
        return evicted_meta

    def add(self, block: Block, last_accessed: float, hit_count: int):
        assert block.state == BlockState.ALLOCATED
        assert block.block_id is not None
        block.set_state(BlockState.FREED)
        self.free_table[block.block_id] = EvictedBlockMetaData(
            block, last_accessed, hit_count)

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int) -> EvictedBlockMetaData:
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block_meta = self.free_table.pop(block_id)
        assert block_meta.block.state == BlockState.FREED
        block_meta.block.set_state(BlockState.ALLOCATED)
        return block_meta

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def get_num_free_blocks(self,
                            blocks_id_in_use: Optional[Set[int]] = None
                            ) -> int:
        if blocks_id_in_use is None:
            return len(self.free_table)

        count = 0
        for block_id in self.free_table:
            if block_id not in blocks_id_in_use:
                count += 1
        return count


def make_mt_evictor(eviction_policy: EvictionPolicy) -> MTEvictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUMTEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
