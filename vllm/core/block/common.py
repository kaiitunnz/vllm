from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Tuple

from vllm.core.block.interfaces import (AllocationOutput, Block,
                                        BlockAllocator, BlockState)
from vllm.core.block.mt_interfaces import MTAllocationOutput, MTBlockAllocator

BlockId = int
RefCount = int


class RefCounterProtocol(Protocol):

    def incr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def decr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def get(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError


class RefCounter(RefCounterProtocol):
    """A class for managing reference counts for a set of block indices.

    The RefCounter class maintains a dictionary that maps block indices to their
    corresponding reference counts. It provides methods to increment, decrement,
    and retrieve the reference count for a given block index.

    Args:
        all_block_indices (Iterable[BlockId]): An iterable of block indices
            to initialize the reference counter with.
    """

    def __init__(self, all_block_indices: Iterable[BlockId]):
        deduped = set(all_block_indices)
        self._refcounts: Dict[BlockId,
                              RefCount] = {index: 0
                                           for index in deduped}

    def incr(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        pre_incr_refcount = self._refcounts[block_id]

        assert pre_incr_refcount >= 0

        post_incr_refcount = pre_incr_refcount + 1
        self._refcounts[block_id] = post_incr_refcount
        return post_incr_refcount

    def decr(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        refcount = self._refcounts[block_id]

        assert refcount > 0
        refcount -= 1

        self._refcounts[block_id] = refcount

        return refcount

    def get(self, block_id: BlockId) -> RefCount:
        assert block_id in self._refcounts
        return self._refcounts[block_id]

    def as_readonly(self) -> "ReadOnlyRefCounter":
        return ReadOnlyRefCounter(self)


class ReadOnlyRefCounter(RefCounterProtocol):
    """A read-only view of the RefCounter class.

    The ReadOnlyRefCounter class provides a read-only interface to access the
    reference counts maintained by a RefCounter instance. It does not allow
    modifications to the reference counts.

    Args:
        refcounter (RefCounter): The RefCounter instance to create a read-only
            view for.
    """

    def __init__(self, refcounter: RefCounter):
        self._refcounter = refcounter

    def incr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Incr not allowed")

    def decr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Decr not allowed")

    def get(self, block_id: BlockId) -> RefCount:
        return self._refcounter.get(block_id)


class CopyOnWriteTracker:
    """A class for tracking and managing copy-on-write operations for blocks.

    The CopyOnWriteTracker class maintains a mapping of source block indices to
        their corresponding copy-on-write destination block indices. It works in
        conjunction with a RefCounter.

    Args:
        refcounter (RefCounter): The reference counter used to track block
            reference counts.
    """

    def __init__(self, refcounter: RefCounterProtocol):
        self._copy_on_writes: List[Tuple[BlockId, BlockId]] = []
        self._refcounter = refcounter

    def is_appendable(self, block: Block) -> bool:
        """Checks if the block is shared or not. If shared, then it cannot
        be appended and needs to be duplicated via copy-on-write
        """
        block_id = block.block_id
        if block_id is None:
            return True

        refcount = self._refcounter.get(block_id)
        return refcount <= 1

    def record_cow(self, src_block_id: Optional[BlockId],
                   trg_block_id: Optional[BlockId]) -> None:
        """Records a copy-on-write operation from source to target block id
        Args:
            src_block_id (BlockId): The source block id from which to copy
                the data
            trg_block_id (BlockId): The target block id to which the data
                is copied
        """
        assert src_block_id is not None
        assert trg_block_id is not None
        self._copy_on_writes.append((src_block_id, trg_block_id))

    def clear_cows(self) -> List[Tuple[BlockId, BlockId]]:
        """Clears the copy-on-write tracking information and returns the current
        state.

        This method returns a list mapping source block indices to
         destination block indices for the current copy-on-write operations.
        It then clears the internal tracking information.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices for the
                current copy-on-write operations.
        """
        cows = self._copy_on_writes
        self._copy_on_writes = []
        return cows


class BlockPool:
    """Used to pre-allocate block objects, in order to avoid excessive python
    object allocations/deallocations.
    The pool starts from "pool_size" objects and will increase to more objects
    if necessary

    Note that multiple block objects may point to the same physical block id,
    which is why this pool is needed, so that it will be easier to support
    prefix caching and more complicated sharing of physical blocks.
    """

    def __init__(self, block_size: int, create_block: Block.Factory,
                 allocator: BlockAllocator, pool_size: int):
        self._block_size = block_size
        self._create_block = create_block
        self._allocator = allocator
        self._pool_size = pool_size
        assert self._pool_size >= 0

        self._free_ids: Deque[int] = deque(range(self._pool_size))
        self._pool: List[Block] = []
        for _ in range(self._pool_size):
            block = self._create_block(
                prev_block=None,
                token_ids=[],
                block_size=self._block_size,
                allocator=self._allocator,
                block_id=None,
            )
            block.init_block_state()
            self._pool.append(block)

    def increase_pool(self):
        """Doubles the internal pool size
        """
        cur_pool_size = self._pool_size
        new_pool_size = cur_pool_size * 2
        self._pool_size = new_pool_size

        self._free_ids += deque(range(cur_pool_size, new_pool_size))

        for _ in range(cur_pool_size, new_pool_size):
            block = self._create_block(
                prev_block=None,
                token_ids=[],
                block_size=self._block_size,
                allocator=self._allocator,
                block_id=None,
            )
            block.init_block_state()
            self._pool.append(block)

    def _allocate_pool_id(self) -> int:
        if len(self._free_ids) == 0:
            self.increase_pool()
            assert len(self._free_ids) > 0

        pool_id = self._free_ids.popleft()
        return pool_id

    def init_block(self, prev_block: Optional[Block], token_ids: List[int],
                   block_size: int, physical_block_id: Optional[int]) -> Block:
        pool_id = self._allocate_pool_id()

        block = self._pool[pool_id]
        assert block.state == BlockState.UNINIT
        block.__init__(  # type: ignore[misc]
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            allocator=block._allocator,  # type: ignore[attr-defined]
            block_id=physical_block_id)
        block.pool_id = pool_id  # type: ignore[attr-defined]
        block.set_state(BlockState.ALLOCATED)
        return block

    def copy_block(self, block: Block) -> Block:
        pool_id = self._allocate_pool_id()

        new_block = self._pool[pool_id]
        assert new_block.state == BlockState.UNINIT
        new_block.__init__(  # type: ignore[misc]
            prev_block=block.prev_block,
            token_ids=block.token_ids,
            block_size=block.block_size,  # type: ignore[attr-defined]
            allocator=block._allocator,  # type: ignore[attr-defined]
            block_id=block.block_id,
            computed=block.computed,
        )
        new_block.pool_id = pool_id  # type: ignore[attr-defined]
        new_block.set_state(BlockState.ALLOCATED)
        return new_block

    def free_block(self, block: Block) -> None:
        block.set_state(BlockState.UNINIT)
        self._free_ids.appendleft(block.pool_id)  # type: ignore[attr-defined]


class AllocationOutputPool:

    def __init__(self, block_pool: BlockPool):
        self._block_pool = block_pool
        self._alloc_pool = self._create_alloc_pool()

    @classmethod
    def create(
        cls,
        block_size: int,
        create_block: Block.Factory,
        allocator: BlockAllocator,
        pool_size: int,
    ) -> "AllocationOutputPool":
        block_pool = BlockPool(
            block_size=block_size,
            create_block=create_block,
            allocator=allocator,
            pool_size=pool_size,
        )
        return cls(block_pool)

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    def _create_alloc_pool(
            self,
            block_list: Optional[List[Block]] = None
    ) -> List[AllocationOutput]:
        if block_list is None:
            block_list = self._block_pool._pool
        return [AllocationOutput(block) for block in block_list]

    def increase_pool(self):
        """Increases pool to match the block pool size"""
        cur_pool_size = self._block_pool._pool_size
        self._alloc_pool.extend([
            AllocationOutput(block)
            for block in self._block_pool._pool[cur_pool_size:]
        ])

    def init_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        physical_block_id: Optional[int],
    ) -> Block:
        block = self._block_pool.init_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            physical_block_id=physical_block_id,
        )
        if self._block_pool._pool_size > len(self._alloc_pool):
            self.increase_pool()
        return block

    def copy_block(self, block: Block) -> Block:
        block = self._block_pool.copy_block(block)
        if self._block_pool._pool_size > len(self._alloc_pool):
            self.increase_pool()
        return block

    def wrap_block(self, block: Block) -> AllocationOutput:
        assert hasattr(block, "pool_id"), "Block must be from the pool"
        alloc = self._alloc_pool[block.pool_id]  # type: ignore[attr-defined]
        alloc.__init__(block=alloc.block,
                       evicted_block=None)  # type: ignore[misc]
        return alloc

    def init_alloc_output(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        physical_block_id: Optional[int],
    ) -> AllocationOutput:
        block = self.init_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            physical_block_id=physical_block_id,
        )
        alloc = self.wrap_block(block)
        return alloc

    def free_block(self, block: Block) -> None:
        self._block_pool.free_block(block)

    def free_alloc_output(self, alloc: AllocationOutput) -> None:
        self.free_block(alloc.block)


class MTAllocationOutputPool:

    def __init__(self, block_pool: BlockPool):
        self._block_pool = block_pool
        self._alloc_pool = self._create_alloc_pool()

    @classmethod
    def create(
        cls,
        block_size: int,
        create_block: Block.Factory,
        allocator: MTBlockAllocator,
        pool_size: int,
    ) -> "MTAllocationOutputPool":
        block_pool = BlockPool(
            block_size=block_size,
            create_block=create_block,
            allocator=allocator,
            pool_size=pool_size,
        )
        return cls(block_pool)

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    def _create_alloc_pool(
            self,
            block_list: Optional[List[Block]] = None
    ) -> List[MTAllocationOutput]:
        if block_list is None:
            block_list = self._block_pool._pool
        return [MTAllocationOutput(block) for block in block_list]

    def increase_pool(self):
        """Increases pool to match the block pool size"""
        cur_pool_size = self._block_pool._pool_size
        self._alloc_pool.extend([
            MTAllocationOutput(block)
            for block in self._block_pool._pool[cur_pool_size:]
        ])

    def init_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        physical_block_id: Optional[int],
    ) -> Block:
        block = self._block_pool.init_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            physical_block_id=physical_block_id,
        )
        if self._block_pool._pool_size > len(self._alloc_pool):
            self.increase_pool()
        return block

    def copy_block(self, block: Block) -> Block:
        block = self._block_pool.copy_block(block)
        if self._block_pool._pool_size > len(self._alloc_pool):
            self.increase_pool()
        return block

    def wrap_block(
        self,
        block: Block,
        evicted_block: Optional[Block] = None,
    ) -> MTAllocationOutput:
        assert hasattr(block, "pool_id"), "Block must be from the pool"
        alloc = self._alloc_pool[block.pool_id]  # type: ignore[attr-defined]
        alloc.__init__(block=alloc.block,
                       evicted_block=evicted_block)  # type: ignore[misc]
        return alloc

    def init_alloc_output(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        physical_block_id: Optional[int],
        evicted_block: Optional[Block] = None,
    ) -> MTAllocationOutput:
        block = self.init_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            physical_block_id=physical_block_id,
        )
        alloc = self.wrap_block(block, evicted_block)
        return alloc

    def free_block(self, block: Block) -> None:
        self._block_pool.free_block(block)

    def free_alloc_output(self, alloc: MTAllocationOutput) -> None:
        self.free_block(alloc.block)


class BlockList:
    """This class is an optimization to allow fast-access to physical 
    block ids. It maintains a block id list that is updated with the 
    block list and this avoids the need to reconstruct the block id 
    list on every iteration of the block manager
    """

    def __init__(self, blocks: List[Block]):
        self._blocks: List[Block] = []
        self._block_ids: List[int] = []

        self.update(blocks)

    def _add_block_id(self, block_id: Optional[BlockId]) -> None:
        assert block_id is not None
        self._block_ids.append(block_id)

    def _update_block_id(self, block_index: int,
                         new_block_id: Optional[BlockId]) -> None:
        assert new_block_id is not None
        self._block_ids[block_index] = new_block_id

    def update(self, blocks: List[Block]):
        self._blocks = blocks

        # Cache block ids for fast query
        self._block_ids = []
        for block in self._blocks:
            self._add_block_id(block.block_id)

    def append_token_ids(self, block_index: int, token_ids: List[int]) -> None:
        block = self._blocks[block_index]
        prev_block_id = block.block_id

        block.append_token_ids(token_ids)

        # CoW or promotion may update the internal block_id
        if prev_block_id != block.block_id:
            self._update_block_id(block_index, block.block_id)

    def append(self, new_block: Block):
        self._blocks.append(new_block)
        self._add_block_id(new_block.block_id)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, block_index: int) -> Block:
        return self._blocks[block_index]

    def __setitem__(self, block_index: int, new_block: Block) -> None:
        self._blocks[block_index] = new_block
        self._update_block_id(block_index, new_block.block_id)

    def reset(self):
        self._blocks = []
        self._block_ids = []

    def list(self) -> List[Block]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids


@dataclass
class CacheMetricData:
    """A utility dataclass to maintain cache metric.
    To avoid overflow, we maintain the hit rate in block granularity, so that
    we can maintain a single hit rate for n_completed_block x block_size,
    and calculate the real time hit rate by the following:
    BS = The number of queries per block.
    nB = The number of completed blocks.
    HR = hit rate of (nB x BS) queries.
    Q = current number of queries (< BS).
    H = current number of hits (< BS).
    hit rate = ((HR x nB) + (H / Q) x (Q / BS)) / (nB + Q / BS)
    """
    num_completed_blocks: int = 0
    completed_block_cache_hit_rate: float = 0.0
    num_incompleted_block_queries: int = 0
    num_incompleted_block_hit: int = 0
    block_size: int = 1000

    def query(self, hit: bool):
        self.num_incompleted_block_queries += 1
        self.num_incompleted_block_hit += 1 if hit else 0

        # When a block is completed, update the cache hit rate
        # and reset the incomplete numbers.
        if self.num_incompleted_block_queries == self.block_size:
            hit_rate = (self.num_incompleted_block_hit /
                        self.num_incompleted_block_queries)
            self.completed_block_cache_hit_rate = (
                self.completed_block_cache_hit_rate * self.num_completed_blocks
                + hit_rate) / (self.num_completed_blocks + 1)
            self.num_incompleted_block_queries = 0
            self.num_incompleted_block_hit = 0
            self.num_completed_blocks += 1

    def get_hit_rate(self):
        incomplete_ratio = self.num_incompleted_block_queries / self.block_size
        total_blocks = self.num_completed_blocks + incomplete_ratio
        if total_blocks == 0:
            return 0.0

        completed_block_hit, incompleted_block_hit = 0.0, 0.0
        if self.num_completed_blocks > 0:
            completed_block_hit = (self.completed_block_cache_hit_rate *
                                   self.num_completed_blocks)
        if self.num_incompleted_block_queries > 0:
            incompleted_hit_rate = (self.num_incompleted_block_hit /
                                    self.num_incompleted_block_queries)
            incompleted_block_hit = (incompleted_hit_rate * incomplete_ratio)
        return (completed_block_hit + incompleted_block_hit) / total_blocks


def get_all_blocks_recursively(last_block: Block) -> List[Block]:
    """Retrieves all the blocks in a sequence starting from the last block.

    This function recursively traverses the sequence of blocks in reverse order,
    starting from the given last block, and returns a list of all the blocks in
    the sequence.

    Args:
        last_block (Block): The last block in the sequence.

    Returns:
        List[Block]: A list of all the blocks in the sequence, in the order they
            appear.
    """

    def recurse(block: Block, lst: List[Block]) -> None:
        if block.prev_block is not None:
            recurse(block.prev_block, lst)
        lst.append(block)

    all_blocks: List[Block] = []
    recurse(last_block, all_blocks)
    return all_blocks
