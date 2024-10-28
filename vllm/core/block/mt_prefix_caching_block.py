from os.path import commonprefix
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from vllm.core.block.common import (BlockPool, CacheMetricData,
                                    CopyOnWriteTracker,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import (Block, BlockAllocator, BlockId,
                                        BlockState, Device,
                                        EvictedBlockMetaData)
from vllm.core.block.mt_interfaces import MTAllocationOutput, MTBlockAllocator
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixHash
from vllm.core.mt_evictor import EvictionPolicy, MTEvictor, make_mt_evictor

_DEFAULT_LAST_ACCESSED_TIME = -1


class MTBlockTracker:
    """Used to track the status of a block inside the prefix caching allocator
    """
    __slots__ = ("active", "last_accessed", "computed", "hit_count")

    def reset(self):
        self.last_accessed: float = _DEFAULT_LAST_ACCESSED_TIME
        self.computed: bool = False
        self.hit_count: int = 0

    def __init__(self):
        self.active: bool = False
        self.reset()

    def enable(self):
        assert not self.active
        self.active = True
        self.reset()

    def disable(self):
        assert self.active
        self.active = False
        self.reset()

    def hit(self):
        self.hit_count += 1


class PrefixCache:

    def __init__(
        self,
        prefix_cache: Optional[Dict[PrefixHash, Block]] = None,
        block_ids: Optional[FrozenSet[int]] = None,
    ):
        self._prefix_cache = {} if prefix_cache is None else prefix_cache
        self._block_ids = block_ids

    def subset(self,
               block_ids: Optional[FrozenSet[int]] = None) -> "PrefixCache":
        return PrefixCache(self._prefix_cache, block_ids)

    def get(self, content_hash: PrefixHash) -> Optional[Block]:
        block = self._prefix_cache.get(content_hash, None)
        if ((block is None) or (self._block_ids is None)
                # or (block.block_id is None)   # TODO(noppanat): Check this.
                or (block.block_id in self._block_ids)):
            return block
        return None

    def add(self, content_hash: PrefixHash, block: Block) -> None:
        if self._block_ids is not None:
            assert block.block_id in self._block_ids
        self._prefix_cache[content_hash] = block

    def pop(self, content_hash: PrefixHash) -> Optional[Block]:
        block = self._prefix_cache.pop(content_hash, None)
        if block is None:
            return None
        if self._block_ids is not None:
            assert block.block_id in self._block_ids
        return block

    def __contains__(self, key: PrefixHash) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return repr({k: v for k, v in self._prefix_cache.items()})


class MTPrefixCachingBlockAllocator(MTBlockAllocator):
    """A block allocator that implements prefix caching.

    The MTPrefixCachingBlockAllocator maintains a cache of blocks based on their
    content hash. It reuses blocks with the same content hash to avoid redundant
    memory allocation. The allocator also supports copy-on-write operations.

    Args:
        num_blocks (int): The total number of blocks to manage.
        block_size (int): The size of each block in tokens.
        block_ids(Optional[Iterable[int]], optional): An optional iterable of
            block IDs. If not provided, block IDs will be assigned sequentially
            from 0 to num_blocks - 1.
    """

    def __repr__(self) -> str:
        ret = []
        for block_id, tracker in self._block_tracker.items():
            ret.append(("active" if tracker.active else "inactive",
                        "computed" if tracker.computed
                        or block_id in self.evictor else "not computed"))
        return repr(ret)

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
        prefix_cache: Optional[PrefixCache] = None,
        block_pool: Optional[BlockPool] = None,
        hit_count_threshold: int = 1,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)

        self._block_size = block_size
        self._hit_count_threshold = hit_count_threshold

        # A list of immutable block IDs that have been touched by scheduler
        # and should be marked as computed after an entire batch of sequences
        # are scheduled.
        self._touched_blocks: Set[BlockId] = set()

        # Used to track status of each physical block id
        self._block_tracker: Dict[BlockId, MTBlockTracker] = {}
        for block_id in block_ids:
            self._block_tracker[block_id] = MTBlockTracker()

        if block_pool is None:
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks
            extra_factor = 4
            self._block_pool = BlockPool(self._block_size, self._create_block,
                                         self, num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool

        # An allocator for blocks that do not have prefix hashes.
        self._hashless_allocator = NaiveBlockAllocator(
            create_block=self._create_block,  # type: ignore
            num_blocks=num_blocks,
            block_size=block_size,
            block_ids=block_ids,
            block_pool=self._block_pool,  # Share alloc pool here
        )

        # A mapping of prefix hash to block index. All blocks which have a
        # prefix hash will be in this dict, even if they have refcount 0.
        self._prefix_cache = (
            PrefixCache(block_ids=self.all_block_ids) if prefix_cache is None
            else prefix_cache.subset(self.all_block_ids)
        )  # TODO(noppanat): may not be necessary.

        # Evitor used to maintain how we want to handle those computed blocks
        # if we find memory pressure is high.
        self.evictor: MTEvictor = make_mt_evictor(eviction_policy)

        # We share the refcounter between allocators. This allows us to promote
        # blocks originally allocated in the hashless allocator to immutable
        # blocks.
        self._refcounter = self._hashless_allocator.refcounter

        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly())

        self.metric_data = CacheMetricData()

    # Implements Block.Factory.
    def _create_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        allocator: BlockAllocator,
        block_id: Optional[int] = None,
        computed: bool = False,
    ) -> Block:
        # Bind block to self.
        allocator = self

        return MTPrefixCachingBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            block_id=block_id,
            allocator=allocator,
            computed=computed,
        )

    def allocate_immutable_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: Optional[Device] = None,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> MTAllocationOutput:
        """Allocates an immutable block with the given token IDs

        Avoid calling this function if you know there is a previously cached 
        block. Instead, use `allocate_cached_block`. In general, you should use 
        `allocate_placeholder_block` and process the block to determine whether 
        it is cached or not and use `promote_placeholder_block` to promote it 
        to an immutable block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
            token_ids (List[int]): The token IDs to be stored in the block.

        Returns:
            Block: The allocated immutable block.
        """
        assert device is None
        assert_prefix_caching_block_or_none(prev_block)

        self.metric_data.query(hit=False)

        # No cached block => Allocate a new block
        alloc = self.allocate_mutable_block(prev_block,
                                            block_ids_in_use=block_ids_in_use)
        alloc.block.append_token_ids(token_ids)
        return alloc

    def allocate_cached_block(self, block: Block) -> MTAllocationOutput:
        """Allocates a block that is already cached."""
        assert block.content_hash is not None
        assert block.content_hash in self._prefix_cache
        assert block.block_id is not None
        self.metric_data.query(hit=True)
        self._incr_refcount_cached_block(block)
        # Hit after _incr_refcount_cached_block as it resets the hit count.
        self._block_tracker[block.block_id].hit()

        alloc = MTAllocationOutput(block)
        return alloc

    def allocate_placeholder_block(self,
                                   prev_block: Optional[Block],
                                   token_ids: List[int],
                                   content_hash: Optional[int] = None,
                                   device: Optional[Device] = None) -> Block:
        assert device is None

        block = self._block_pool.init_block(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=self._block_size,
            physical_block_id=None,
        )
        block.set_state(BlockState.PLACEHOLDER)
        assert isinstance(block, MTPrefixCachingBlock)
        if content_hash is not None:
            block.set_content_hash(content_hash)
        assert block.content_hash is not None

        return block

    def promote_placeholder_block(
            self,
            block: Block,
            block_ids_in_use: Optional[Set[BlockId]] = None
    ) -> MTAllocationOutput:
        """Promotes a placeholder block to an immutable block."""
        assert block.state == BlockState.PLACEHOLDER
        assert block.content_hash is not None
        assert block.block_id is None
        assert block.content_hash not in self._prefix_cache

        self.metric_data.query(hit=False)

        block_id, evicted_block = self._allocate_block_id(block_ids_in_use)

        # Modify the block in place as it has been preallocated.
        block.block_id = block_id
        block.set_state(BlockState.ALLOCATED)

        # Promote to immutable block.
        self._prefix_cache.add(block.content_hash, block)
        self._touched_blocks.add(block_id)

        alloc = MTAllocationOutput(block, evicted_block)
        return alloc

    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: Optional[Device] = None,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> List[MTAllocationOutput]:
        allocs: List[MTAllocationOutput] = []
        for token_ids in block_token_ids:
            prev_alloc = self.allocate_immutable_block(
                prev_block=prev_block,
                token_ids=token_ids,
                device=device,
                block_ids_in_use=block_ids_in_use,
            )
            prev_block = prev_alloc.block
            allocs.append(prev_alloc)
        return allocs

    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        device: Optional[Device] = None,
        block_ids_in_use: Optional[Set[BlockId]] = None,
    ) -> MTAllocationOutput:
        """Allocates a mutable block. If there are no free blocks, this will
        evict unused cached blocks.

        Args:
            prev_block (Block): The previous block in the sequence.
                None is not allowed unlike it is super class.

        Returns:
            Block: The allocated mutable block.
        """
        assert device is None
        assert_prefix_caching_block_or_none(prev_block)

        block_id, evicted_meta = self._allocate_block_id(block_ids_in_use)
        block = self._block_pool.init_block(
            prev_block=prev_block,
            token_ids=[],
            block_size=self._block_size,
            physical_block_id=block_id,
        )
        assert not block.computed
        assert block.content_hash is None
        alloc = MTAllocationOutput(block, evicted_meta)
        return alloc

    def _incr_refcount_cached_block(self, block: Block) -> None:
        # Set this block to be "computed" since it is pointing to a
        # cached block id (which was already computed)
        block.computed = True

        block_id = block.block_id
        assert block_id is not None

        refcount = self._refcounter.incr(block_id)
        if refcount == 1:
            # In case a cached block was evicted, restore its tracking
            hit_count = 0
            if block_id in self.evictor:
                block_meta = self.evictor.remove(block_id)
                hit_count = block_meta.hit_count

            self._track_block_id(block_id, computed=True, hit_count=hit_count)

    def _decr_refcount_cached_block(self, block: Block) -> None:
        # Ensure this is immutable/cached block
        assert block.content_hash is not None

        block_id = block.block_id
        assert block_id is not None

        refcount = self._refcounter.decr(block_id)
        if refcount > 0:
            # block.block_id = None # TODO(noppanat): Check this.
            return
        else:
            assert refcount == 0

        # No longer used
        assert block.content_hash in self._prefix_cache

        # Add the cached block to the evictor
        # (This keeps the cached block around so it can be reused)
        self.evictor.add(
            block,
            self._block_tracker[block_id].last_accessed,
            self._block_tracker[block_id].hit_count,
        )

        # Stop tracking the block
        self._untrack_block_id(block_id)

        # block.block_id = None # TODO(noppanat): Check this.

    def _decr_refcount_hashless_block(self, block: Block) -> None:
        block_id = block.block_id
        assert block_id is not None

        # We may have a fork case where block is shared,
        # in which case, we cannot remove it from tracking
        refcount = self._refcounter.get(block_id)
        if refcount == 1:
            self._untrack_block_id(block_id)

        # Decrement refcount of the block_id, but do not free the block object
        # itself (will be handled by the caller)
        self._hashless_allocator.free(block, keep_block_object=True)

    def _allocate_block_id(
        self,
        block_ids_in_use: Optional[Set[BlockId]] = None
    ) -> Tuple[BlockId, Optional[EvictedBlockMetaData]]:
        """First tries to allocate a block id from the hashless allocator,
        and if there are no blocks, then tries to evict an unused cached block.

        Returns
        -------
        block_id: BlockId
            The allocated block id.
        to_swap_in: List[BlockId]
            The list of block ids that need to be swapped in.
        to_swap_out: List[BlockId]
            The list of block ids that need to be swapped out.
        """
        hashless_block_id = self._maybe_allocate_hashless_block_id()
        if hashless_block_id is not None:
            return hashless_block_id, None

        alloc = self._maybe_allocate_evicted_block_id(block_ids_in_use)
        if alloc is not None:
            return alloc

        # No block available in hashless allocator, nor in unused cache blocks.
        raise MTBlockAllocator.NoFreeBlocksError()

    def _maybe_allocate_hashless_block_id(self) -> Optional[BlockId]:
        try:
            # Allocate mutable block and extract its block_id
            alloc = self._hashless_allocator.allocate_mutable_block(
                prev_block=None)
            block_id = alloc.block.block_id
            self._hashless_allocator._block_pool.free_block(alloc.block)

            self._track_block_id(block_id, computed=False, hit_count=0)
            return block_id
        except MTBlockAllocator.NoFreeBlocksError:
            return None

    def _maybe_allocate_evicted_block_id(
        self,
        block_ids_in_use: Optional[Set[BlockId]] = None
    ) -> Optional[Tuple[BlockId, Optional[EvictedBlockMetaData]]]:
        if self.evictor.num_blocks == 0:
            return None

        # Here we get an evicted block, which is only added
        # into evictor if its ref counter is 0
        # and since its content would be changed, we need
        # to remove it from _cached_blocks's tracking list
        evicted_meta: Optional[EvictedBlockMetaData] = self.evictor.evict(
            block_ids_in_use)
        assert evicted_meta is not None
        evicted_block = evicted_meta.block
        assert evicted_block is not None
        assert evicted_block.block_id is not None
        assert evicted_block.content_hash is not None
        assert evicted_block.state == BlockState.EVICTED
        block_id = evicted_block.block_id
        content_hash_to_evict = evicted_block.content_hash

        # Sanity checks
        cached_block = self._prefix_cache.get(content_hash_to_evict)
        if cached_block is None:
            # NOTE(noppanat): The cached block was somehow destroyed.
            # Free the evicted block directly.
            self._block_pool.free_block(evicted_block)
            evicted_meta = None
        else:
            _block_id = cached_block.block_id
            assert _block_id is not None
            if block_id != _block_id:
                # TODO(noppanat): Remove this.
                # In this case, the block has been recomputed by the running
                # sequences and is already in the highest-tier device.
                self.destroy(evicted_block, keep_prefix_cache=True)
                evicted_meta = None
            elif evicted_meta.hit_count < self._hit_count_threshold:
                # The block does not have enough hits to be evicted.
                self.destroy(evicted_block, keep_prefix_cache=False)
                evicted_meta = None

        # assert _block_id is not None
        assert self._refcounter.get(block_id) == 0

        # self._cached_blocks.pop(content_hash_to_evict)

        self._refcounter.incr(block_id)
        self._track_block_id(block_id, computed=False, hit_count=0)

        return block_id, evicted_meta

    def _free_block_id(self, block: Block) -> None:
        """Decrements the refcount of the block. The block may be in two
        possible states: (1) immutable/cached or (2) mutable/hashless.
        In the first case, the refcount is decremented directly and the block
        may be possibly added to the evictor. In other case, hashless
        allocator free(..) with keep_block_object=True is called to only free
        the block id (since the block object may be reused by the caller)
        """
        assert block.state == BlockState.ALLOCATED
        block_id = block.block_id
        assert block_id is not None, "Freeing unallocated block is undefined"

        if block.content_hash is not None:
            # Immutable: This type of block is always cached, and we want to
            # keep it in the evictor for future reuse
            self._decr_refcount_cached_block(block)
        else:
            # Mutable: This type of block is not cached, so we release it
            # directly to the hashless allocator
            self._decr_refcount_hashless_block(block)

        # assert block.block_id is None # TODO(noppanat): Check this.

    def free(self, block: Block, keep_block_object: bool = False) -> None:
        """Release the block (look at free_block_id(..) docs)"""
        # Release the physical block index
        self._free_block_id(block)

        # Release the block object to the pool
        # if not keep_block_object:
        #     self._alloc_pool.free_block(block)

    def free_block_id(self, block_id: BlockId) -> None:
        raise NotImplementedError(
            "free_block_id is not supported in MTPrefixCachingBlockAllocator")

    def destroy(self, block: Block, keep_prefix_cache: bool = False) -> None:
        if block.state == BlockState.PLACEHOLDER:
            assert block.block_id is None
            self._block_pool.free_block(block)
            return

        assert block.state == BlockState.EVICTED

        if not keep_prefix_cache:
            assert block.content_hash is not None
            cached_block = self._prefix_cache.get(block.content_hash)
            assert cached_block is not None
            if block.block_id == cached_block.block_id:
                # Otherwise, the block has been recomputed by the running
                # sequences and is already in the highest-tier device.
                self._prefix_cache.pop(block.content_hash)
        self._block_pool.free_block(block)

    def fork(self, last_block: Block) -> List[MTAllocationOutput]:
        """Creates a new sequence of blocks that shares the same underlying
        memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: The new sequence of blocks that shares the same memory
                as the original sequence.
        """
        raise NotImplementedError("Forking is not yet supported.")
        source_blocks = get_all_blocks_recursively(last_block)

        forked_allocs: List[MTAllocationOutput] = []
        prev_block = None
        for block in source_blocks:
            block_id = block.block_id
            assert block_id is not None

            refcount = self._refcounter.incr(block_id)
            assert refcount != 1, "can't fork free'd block_id = {}".format(
                block_id)

            forked_alloc = self._alloc_pool.init_alloc_output(
                prev_block=prev_block,
                token_ids=block.token_ids,
                block_size=self._block_size,
                physical_block_id=block_id,
            )

            forked_allocs.append(
                forked_alloc)  # TODO(noppanat): check swapping logic
            prev_block = forked_allocs[-1]

        return forked_allocs

    def get_num_free_blocks(self,
                            block_ids_in_use: Optional[Set[int]] = None,
                            device: Optional[Device] = None) -> int:
        assert device is None
        # The number of free blocks is the number of hashless free blocks
        # plus the number of blocks evictor could free from its list.
        return self._hashless_allocator.get_num_free_blocks(
        ) + self.evictor.get_num_free_blocks(block_ids_in_use)

    def get_num_total_blocks(self) -> int:
        return self._hashless_allocator.get_num_total_blocks()

    def get_physical_block_id(self, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain block allocator
        given the absolute block id.

        Args:
            absolute_id (int): The absolute block id for the block
                in whole allocator.

        Returns:
            int: The rzero-offset block id on certain device.
        """
        return sorted(self.all_block_ids).index(absolute_id)

    @property
    def block_pool(self) -> BlockPool:
        return self._block_pool

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return self._hashless_allocator.all_block_ids

    def get_prefix_cache_hit_rate(self) -> float:
        return self.metric_data.get_hit_rate()

    def is_block_cached(self, block: Block) -> bool:
        assert block.content_hash is not None
        return block.content_hash in self._prefix_cache

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        """Once a mutable block is full, it can be promoted to an immutable
        block. This means that its content can be referenced by future blocks
        having the same prefix.

        Note that if we already have a cached block with the same content, we
        will replace the newly-promoted block's mapping with the existing cached
        block id.

        Args:
            block: The mutable block to be promoted.

        Returns:
            BlockId: Either the original block index, or the block index of
                the previously cached block matching the same content.
        """
        # Ensure block can be promoted
        assert block.state == BlockState.ALLOCATED
        assert block.content_hash is not None
        assert block.block_id is not None
        assert self._refcounter.get(block.block_id) > 0

        cached_block = self._prefix_cache.get(block.content_hash)

        if cached_block is None:
            # No cached content hash => Set this block as cached.
            # Note that this block cannot be marked as computed yet
            # because other sequences in the same batch cannot reuse
            # this block.
            self._prefix_cache.add(block.content_hash, block)
            # Mark this block as touched so that it can be marked as
            # computed after the entire batch of sequences are scheduled.
            self._touched_blocks.add(block.block_id)
            return block.block_id

        # Reuse the cached content hash
        self._decr_refcount_hashless_block(block)

        # If the cached block is freed, allocate it.
        cached_block_id = cached_block.block_id
        assert cached_block_id is not None
        # NOTE(noppanat): This is handled by _incr_refcount_cached_block.
        # if cached_block_id in self.evictor:
        # self.evictor.remove(cached_block_id)
        block.block_id = cached_block_id

        # Increment refcount of the cached block and (possibly) restore
        # it from the evictor.
        # Note that in this case, the block is marked as computed
        self._incr_refcount_cached_block(block)

        return block.block_id

    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            BlockId: The block index of the new block if a copy-on-write
                operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        # NOTE(noppanat): Copy-on-write is not yet supported.
        block_id = block.block_id
        assert block_id is not None
        return block_id
        src_block_id = block.block_id
        assert src_block_id is not None

        if self._cow_tracker.is_appendable(block):
            return src_block_id

        self._free_block_id(block)
        trg_block_id = self._allocate_block_id()

        self._cow_tracker.record_cow(src_block_id, trg_block_id)

        return trg_block_id

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, used in prefix caching.

        If the block is added into evictor, we need to update corresponding
        info in evictor's metadata.
        """

        for block_id in block_ids:
            if self._block_tracker[block_id].active:
                self._block_tracker[block_id].last_accessed = now
            elif block_id in self.evictor:
                self.evictor.update(block_id, now)
            else:
                raise ValueError(
                    "Mark block as accessed which is not belonged to GPU")

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        # Mark all touched blocks as computed.
        for block_id in self._touched_blocks:
            self._block_tracker[block_id].computed = True
        self._touched_blocks.clear()

    def _track_block_id(self, block_id: Optional[BlockId], computed: bool,
                        hit_count: int) -> None:
        assert block_id is not None
        self._block_tracker[block_id].enable()
        self._block_tracker[block_id].computed = computed
        self._block_tracker[block_id].hit_count = hit_count

    def _untrack_block_id(self, block_id: Optional[BlockId]) -> None:
        assert block_id is not None
        self._block_tracker[block_id].disable()

    def block_is_computed(self, block_id: int) -> bool:
        if self._block_tracker[block_id].active:
            return self._block_tracker[block_id].computed
        else:
            return block_id in self.evictor

    def get_computed_block_ids(
        self,
        prev_computed_block_ids: List[int],
        block_ids: List[int],
        skip_last_block_id: bool = True,
    ) -> List[int]:
        prev_prefix_size = len(prev_computed_block_ids)
        cur_size = len(block_ids)
        if skip_last_block_id:
            cur_size -= 1

        # Sanity checks
        assert cur_size >= 0
        assert prev_prefix_size <= cur_size

        ret = prev_computed_block_ids
        for i in range(prev_prefix_size, cur_size):
            block_id = block_ids[i]
            if self.block_is_computed(block_id):
                ret.append(block_id)
        return ret

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        """Return the block ids that are common for a given sequence group.

        Only those blocks that are immutable and already be marked
        compyted would be taken consideration.
        """

        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.

        # It returns a list of int although type annotation says list of string.
        if len(computed_seq_block_ids) == 1:
            return computed_seq_block_ids[0]

        return commonprefix([ids for ids in computed_seq_block_ids
                             if ids]  # type: ignore
                            )

    def get_num_full_blocks_touched(self, blocks: List[Block]) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out.

        Args:
            blocks: List of blocks to be swapped.
        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks. Non full blocks are ignored
                when deciding the number of blocks to touch.
        """
        num_touched_blocks: int = 0
        for block in blocks:
            # If the block has a match in the cache and the cached
            # block is not referenced, then we still count it as a
            # touched block
            assert block.content_hash is not None
            cached_block = self._prefix_cache.get(block.content_hash)
            if block.is_full and (cached_block is None or
                                  (block.content_hash is not None and
                                   ((cached_block.block_id is not None) and
                                    (cached_block.block_id in self.evictor)))):
                num_touched_blocks += 1
        return num_touched_blocks

    def swap_out(self, blocks: List[Block]) -> None:
        raise NotImplementedError("Swap out is not supported.")

    def swap_in(self, blocks: List[Block]) -> None:
        raise NotImplementedError("Swap in is not supported.")

    def move_out(self, block: Block) -> None:
        assert ((block.state == BlockState.FREED)
                or (block.state == BlockState.EVICTED))
        assert block.content_hash is not None
        assert block.content_hash in self._prefix_cache

        block_id = block.block_id
        assert block_id is not None

        if block_id in self.evictor:
            # In case of moving out of lower-tier devices where all blocks are
            # in the evictor.
            self.evictor.remove(block_id)
            self._hashless_allocator.free_block_id(block_id)

        block.block_id = None
        block.set_state(BlockState.EVICTED)

    def move_in(
            self,
            block: Block,
            hit_count: int = 0,
            evictable: bool = False,
            block_ids_in_use: Optional[Set[int]] = None) -> MTAllocationOutput:
        assert block.state == BlockState.EVICTED
        assert block.content_hash is not None

        alloc = self.allocate_mutable_block(prev_block=None,
                                            block_ids_in_use=block_ids_in_use)
        # Must modify the block in place as it is held by the prefix cache.
        block.block_id = alloc.block.block_id
        assert block.block_id is not None
        self._block_pool.free_block(alloc.block)
        block.set_state(BlockState.ALLOCATED)

        # Set the hit count inherited from the source device.
        self._block_tracker[block.block_id].hit_count = hit_count

        if evictable:
            self._decr_refcount_cached_block(block)
            assert block.state == BlockState.FREED
        else:
            # The block must have already been computed if moved in.
            self._block_tracker[block.block_id].computed = True

        alloc = MTAllocationOutput(block, alloc.evicted_meta)
        return alloc


class MTPrefixCachingBlock(Block):
    """A block implementation that supports prefix caching.

    The MTPrefixCachingBlock class represents a block of token IDs with prefix
    caching capabilities. It wraps a NaiveBlock internally and provides
    additional functionality for content hashing and promoting immutable blocks
    with the prefix caching allocator.

    Args:
        prev_block (Optional[MTPrefixCachingBlock]): The previous block in the
            sequence.
        token_ids (List[int]): The initial token IDs to be stored in the block.
        block_size (int): The maximum number of token IDs that can be stored in
            the block.
        allocator (BlockAllocator): The prefix
            caching block allocator associated with this block.
        block_id (Optional[int], optional): The physical block index
            of this block. Defaults to None.
    """

    def __init__(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        block_size: int,
        allocator: MTBlockAllocator,
        block_id: Optional[int] = None,
        computed: bool = False,
    ):
        assert isinstance(allocator, MTPrefixCachingBlockAllocator), (
            "Currently this class is only tested with "
            "MTPrefixCachingBlockAllocator. Got instead allocator = {}".format(
                allocator))
        assert_prefix_caching_block_or_none(prev_block)

        self._prev_block = prev_block
        self._cached_content_hash: Optional[int] = None
        self._cached_num_tokens_total: int = 0
        self._allocator = allocator
        self._last_accessed: float = _DEFAULT_LAST_ACCESSED_TIME
        self._computed = computed

        # On the first time, we create the block object, and next we only
        # reinitialize it
        if hasattr(self, "_block"):
            self._block.__init__(  # type: ignore[has-type]
                prev_block=prev_block,
                token_ids=token_ids,
                block_size=block_size,
                block_id=block_id,
                allocator=self._allocator,
            )
        else:
            self._block = NaiveBlock(
                prev_block=prev_block,
                token_ids=token_ids,
                block_size=block_size,
                block_id=block_id,
                allocator=self._allocator,
            )

        self._update_num_tokens_total()

    def _update_num_tokens_total(self):
        """Incrementally computes the number of tokens that there is
        till the current block (included)
        """
        res = 0

        # Add all previous blocks
        if self._prev_block is not None:
            res += self._prev_block.num_tokens_total

        # Add current block
        res += len(self.token_ids)

        self._cached_num_tokens_total = res

    @property
    def computed(self) -> bool:
        return self._computed

    @computed.setter
    def computed(self, value) -> None:
        self._computed = value

    @property
    def last_accessed(self) -> float:
        return self._last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        self._last_accessed = last_accessed_ts

    def append_token_ids(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block and registers the block as
        immutable if the block becomes full.

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        # Ensure this is mutable block (not promoted)
        assert self.content_hash is None
        assert not self.computed

        if len(token_ids) == 0:
            return

        # Ensure there are input tokens
        assert token_ids, "Got token_ids = {}".format(token_ids)

        # Naive block handles CoW.
        self._block.append_token_ids(token_ids)
        self._update_num_tokens_total()

        # If the content hash is present, then the block can be made immutable.
        # Register ourselves with the allocator, potentially replacing the
        # physical block index.
        if self.content_hash is not None:
            self.block_id = self._allocator.promote_to_immutable_block(self)

    def set_content_hash(self, content_hash: int) -> None:
        """Sets the content hash of the current block
        
        The block must be a placeholder block."""
        assert self._cached_content_hash is None
        assert self.state == BlockState.PLACEHOLDER
        self._cached_content_hash = content_hash

    @property
    def block_id(self) -> Optional[int]:
        return self._block.block_id

    @block_id.setter
    def block_id(self, value) -> None:
        self._block.block_id = value

    @property
    def is_full(self) -> bool:
        return self._block.is_full

    @property
    def num_empty_slots(self) -> int:
        return self._block.num_empty_slots

    @property
    def num_tokens_total(self) -> int:
        return self._cached_num_tokens_total

    @property
    def block_size(self) -> int:
        return self._block.block_size

    @property
    def token_ids(self) -> List[int]:
        return self._block.token_ids

    @property
    def prev_block(self) -> Optional[Block]:
        return self._prev_block

    @property
    def content_hash(self) -> Optional[int]:
        """Return the content-based hash of the current block, or None if it is
        not yet defined.

        For the content-based hash to be defined, the current block must be
        full.
        """
        # If the hash is already computed, return it.
        if self._cached_content_hash is not None:
            return self._cached_content_hash

        # We cannot compute a hash for the current block because it is not full.
        if not self.is_full:
            return None

        is_first_block = self._prev_block is None
        prev_block_hash = (
            None if is_first_block else
            self._prev_block.content_hash  # type: ignore
        )

        # Previous block exists but does not yet have a hash.
        # Return no hash in this case.
        if prev_block_hash is None and not is_first_block:
            return None

        self._cached_content_hash = MTPrefixCachingBlock.hash_block_tokens(
            is_first_block,
            prev_block_hash,
            cur_block_token_ids=self.token_ids)
        return self._cached_content_hash

    @staticmethod
    def hash_block_tokens(
        is_first_block: bool,
        prev_block_hash: Optional[int],
        cur_block_token_ids: List[int],
    ) -> int:
        """Computes a hash value corresponding to the contents of a block and
        the contents of the preceding block(s). The hash value is used for
        prefix caching.

        NOTE: Content-based hashing does not yet support LoRA.

        Parameters:
        - is_first_block (bool): A flag indicating if the block is the first in
            the sequence.
        - prev_block_hash (Optional[int]): The hash of the previous block. None
            if this is the first block.
        - cur_block_token_ids (List[int]): A list of token ids in the current
            block. The current block is assumed to be full.

        Returns:
        - int: The computed hash value for the block.
        """
        assert (prev_block_hash is None) == is_first_block
        return hash((is_first_block, prev_block_hash, *cur_block_token_ids))


def assert_prefix_caching_block_or_none(block: Optional[Block]):
    if block is None:
        return
    assert isinstance(block,
                      MTPrefixCachingBlock), "Got block = {}".format(block)
