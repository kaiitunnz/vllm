"""A block manager that manages token blocks."""
import itertools
from typing import Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple

from vllm.core.block.interfaces import Block, BlockState
from vllm.core.block.mt_block_allocator import MTPrefixAwareBlockAllocator
from vllm.core.block.mt_block_table import MTBlockTable
from vllm.core.block.mt_interfaces import MTDeviceAwareBlockAllocator
from vllm.core.block.mt_prefix_caching_block import MTPrefixCachingBlock
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.interfaces import AllocStatus
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device, cdiv, chunk_list

logger = init_logger(__name__)

SeqId = int
EncoderSeqId = str


class SequenceMeta:

    class CacheManager:

        def __init__(self):
            self.cached_blocks: List[Block] = []
            self.cached_blocks_token_ids: List[List[int]] = []
            self.cached_blocks_hashes: List[int] = []

            self.cached_blocks_to_move_in: List[Block] = []
            self.cached_blocks_to_move_in_token_ids: List[List[int]] = []
            self.cached_blocks_to_move_in_hashes: List[int] = []

            self.full_blocks: List[Block] = []

        def add_cached_block(self, block: Block, token_ids: List[int],
                             content_hash: int) -> None:
            self.cached_blocks.append(block)
            self.cached_blocks_token_ids.append(token_ids)
            self.cached_blocks_hashes.append(content_hash)

        def add_cached_block_to_move_in(self, block: Block,
                                        token_ids: List[int],
                                        content_hash: int) -> None:
            self.cached_blocks_to_move_in.append(block)
            self.cached_blocks_to_move_in_token_ids.append(token_ids)
            self.cached_blocks_to_move_in_hashes.append(content_hash)

        def add_full_block(self, block: Block) -> None:
            assert block.content_hash is not None
            self.full_blocks.append(block)

        def iter_cached_blocks(self) -> Iterable[Tuple[List[int], int]]:
            return zip(self.cached_blocks_token_ids, self.cached_blocks_hashes)

        def iter_cached_blocks_to_move_in(
                self) -> Iterable[Tuple[List[int], int]]:
            return zip(self.cached_blocks_to_move_in_token_ids,
                       self.cached_blocks_to_move_in_hashes)

        def iter_all_cached_blocks(self) -> Iterable[Tuple[List[int], int]]:
            return itertools.chain(self.iter_cached_blocks(),
                                   self.iter_cached_blocks_to_move_in())

        @property
        def all_cached_blocks(self) -> Iterable[Block]:
            return itertools.chain(self.cached_blocks,
                                   self.cached_blocks_to_move_in)

        def reset_cached_blocks(self) -> None:
            self.cached_blocks = []
            self.cached_blocks_token_ids = []
            self.cached_blocks_hashes = []
            self.cached_blocks_to_move_in = []
            self.cached_blocks_to_move_in_token_ids = []
            self.cached_blocks_to_move_in_hashes = []

    def __init__(self, seq: Sequence, block_size: int,
                 allocator: MTDeviceAwareBlockAllocator):
        self._sequence = seq
        self._allocator = allocator
        self._manager = SequenceMeta.CacheManager()
        self._tail_block_token_ids = self._init_cache_manager(
            seq.get_token_ids(), block_size)
        self._num_required_blocks: Optional[int] = None

    @property
    def sequence(self):
        return self._sequence

    @property
    def seq_id(self):
        return self._sequence.seq_id

    @property
    def cached_blocks(self):
        return self._manager.cached_blocks

    @property
    def cached_blocks_to_move_in(self):
        return self._manager.cached_blocks_to_move_in

    @property
    def full_blocks(self):
        return self._manager.full_blocks

    @property
    def tail_block_token_ids(self):
        return self._tail_block_token_ids

    @property
    def block_ids_in_use(self) -> Set[int]:
        return {
            block.block_id
            for block in self._manager.all_cached_blocks
            if block.block_id is not None
        }

    @property
    def num_required_blocks(self) -> int:
        assert (
            self._num_required_blocks
            is not None), "Number of required blocks has not been computed."
        return self._num_required_blocks

    def get_num_required_blocks(
            self,
            block_size: int,
            num_lookahead_slots: int,
            allocated_evicted_blocks: Optional[List[int]] = None) -> int:
        if (allocated_evicted_blocks is None
                and self._num_required_blocks is not None):
            return self._num_required_blocks

        cached_blocks_to_allocate: List[int] = []
        if allocated_evicted_blocks is None:
            for block in self.cached_blocks:
                assert block.block_id is not None
                if block.state == BlockState.FREED:
                    cached_blocks_to_allocate.append(block.block_id)
        else:
            for block in self.cached_blocks:
                assert block.block_id is not None
                if ((block.state == BlockState.FREED)
                        and (block.block_id not in allocated_evicted_blocks)):
                    cached_blocks_to_allocate.append(block.block_id)
            allocated_evicted_blocks.extend(cached_blocks_to_allocate)

        num_tail_blocks = cdiv(
            len(self._tail_block_token_ids) + num_lookahead_slots, block_size)

        self._num_required_blocks = (len(cached_blocks_to_allocate) +
                                     len(self.cached_blocks_to_move_in) +
                                     len(self.full_blocks) + num_tail_blocks)
        return self._num_required_blocks

    def _init_cache_manager(self, token_ids: List[int],
                            block_size: int) -> List[int]:
        """
        Returns:
            List[int]: The tail block.
        """
        block_token_ids: List[List[int]] = []
        tail_block_token_ids: List[int] = []
        for cur_token_ids in chunk_list(token_ids, block_size):
            if len(cur_token_ids) == block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_block_token_ids = cur_token_ids

        content_hash: Optional[int] = None
        cached_block: Optional[Block] = None
        prev_block: Optional[Block] = None
        for cur_token_ids in block_token_ids:
            content_hash, cached_block = self._get_cached_block(
                content_hash, cur_token_ids)
            if cached_block is None:
                cached_block = self._allocator.allocate_placeholder_block(
                    prev_block=prev_block,
                    token_ids=cur_token_ids,
                    content_hash=content_hash)
                self._manager.add_full_block(cached_block)
            elif self._allocator.get_device_tier(cached_block) == 0:
                self._manager.add_cached_block(cached_block, cur_token_ids,
                                               content_hash)
            else:
                self._manager.add_cached_block_to_move_in(
                    cached_block, cur_token_ids, content_hash)
            prev_block = cached_block

        return tail_block_token_ids

    def recompute(self) -> None:
        new_manager = SequenceMeta.CacheManager()

        prev_block: Optional[Block] = None
        for cur_token_ids, content_hash in (
                self._manager.iter_all_cached_blocks()):
            cached_block = self._allocator.get_cached_block(content_hash)
            if cached_block is None:
                cached_block = self._allocator.allocate_placeholder_block(
                    prev_block=prev_block,
                    token_ids=cur_token_ids,
                    content_hash=content_hash)
                new_manager.add_full_block(cached_block)
            elif self._allocator.get_device_tier(cached_block) == 0:
                new_manager.add_cached_block(cached_block, cur_token_ids,
                                             content_hash)
            else:
                new_manager.add_cached_block_to_move_in(
                    cached_block, cur_token_ids, content_hash)
            prev_block = cached_block

        for placeholder_block in self.full_blocks:
            assert placeholder_block.state == BlockState.PLACEHOLDER
            assert placeholder_block.content_hash is not None
            cached_block = self._allocator.get_cached_block(
                placeholder_block.content_hash)
            if cached_block is None:
                new_manager.add_full_block(placeholder_block)
                continue
            assert cached_block.content_hash is not None
            if self._allocator.get_device_tier(cached_block) == 0:
                new_manager.add_cached_block(cached_block,
                                             cached_block.token_ids,
                                             cached_block.content_hash)
            else:
                new_manager.add_cached_block_to_move_in(
                    cached_block, cached_block.token_ids,
                    cached_block.content_hash)
            self._allocator.destroy(placeholder_block)

        self._manager = new_manager

    def refresh_cached_blocks(self) -> None:
        self._num_required_blocks = None
        cache_iter = self._manager.iter_all_cached_blocks()
        self._manager.reset_cached_blocks()

        prev_block: Optional[Block] = None
        for cur_token_ids, content_hash in cache_iter:
            cached_block = self._allocator.get_cached_block(content_hash)
            if cached_block is None:
                cached_block = self._allocator.allocate_placeholder_block(
                    prev_block=prev_block,
                    token_ids=cur_token_ids,
                    content_hash=content_hash)
                self._manager.add_full_block(cached_block)
            elif self._allocator.get_device_tier(cached_block) == 0:
                self._manager.add_cached_block(cached_block, cur_token_ids,
                                               content_hash)
            else:
                self._manager.add_cached_block_to_move_in(
                    cached_block, cur_token_ids, content_hash)
            prev_block = cached_block

    def _get_cached_block(
            self, prev_block_hash: Optional[int],
            cur_block_token_ids: List[int]) -> Tuple[int, Optional[Block]]:
        content_hash = MTPrefixCachingBlock.hash_block_tokens(
            is_first_block=prev_block_hash is None,
            prev_block_hash=prev_block_hash,
            cur_block_token_ids=cur_block_token_ids,
        )
        cached_block = self._allocator.get_cached_block(content_hash)
        return content_hash, cached_block

    def deallocate(self) -> None:
        # Destroy placeholder blocks.
        for block in self.full_blocks:
            if block.state == BlockState.PLACEHOLDER:
                self._allocator.destroy(block)
                assert block.state == BlockState.UNINIT


# class MTBlockSpaceManager(BlockSpaceManager):
class MTBlockSpaceManager:
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    The current implementation is partial; in particular prefix caching and
    sliding-window are not feature complete. This class implements the design
    described in https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching
        assert enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = MTPrefixAwareBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, MTBlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, MTBlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)

    def process_sequence_group(self, seq_group: SequenceGroup,
                               status: SequenceStatus) -> List[SequenceMeta]:
        return [
            SequenceMeta(seq, self.block_size, self.block_allocator)
            for seq in seq_group.get_seqs(status=status)
        ]

    def get_block_ids_in_use(self, seq_metas: List[SequenceMeta]) -> Set[int]:
        return {
            block_id
            for seq_meta in seq_metas for block_id in seq_meta.block_ids_in_use
        }

    def can_allocate(
            self,
            seq_group: SequenceGroup,
            seq_metas: List[SequenceMeta],
            num_allocated_blocks: int = 0,
            num_lookahead_slots: int = 0,
            allocated_evicted_blocks: Optional[List[int]] = None
    ) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # NOTE(noppanat): This is necessary for determining the number of
        # required blocks.
        assert allocated_evicted_blocks is not None

        num_required_blocks = self.get_num_required_blocks(
            seq_group,
            seq_metas[0],
            num_lookahead_slots=num_lookahead_slots,
            allocated_evicted_blocks=allocated_evicted_blocks)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)
        # NOTE(noppanat): In this scheduler, we try allocating without actually
        # allocating the blocks.
        num_free_gpu_blocks -= num_allocated_blocks

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_num_required_blocks(
        self,
        seq_group: SequenceGroup,
        seq_meta: SequenceMeta,
        num_lookahead_slots: int = 0,
        allocated_evicted_blocks: Optional[List[int]] = None,
    ) -> int:
        num_required_blocks = seq_meta.get_num_required_blocks(
            self.block_size, num_lookahead_slots, allocated_evicted_blocks)

        if seq_group.is_encoder_decoder():
            raise NotImplementedError("Cross-attention is not yet supported.")
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += MTBlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        return num_required_blocks

    def _allocate_sequence(self, seq_meta: SequenceMeta,
                           block_ids_in_use: Set[int]) -> MTBlockTable:
        block_table = MTBlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        block_table.allocate(
            token_ids=seq_meta.sequence.get_token_ids(),
            cached_blocks=seq_meta.cached_blocks,
            cached_blocks_to_move_in=seq_meta.cached_blocks_to_move_in,
            full_blocks=seq_meta.full_blocks,
            tail_block_token_ids=seq_meta.tail_block_token_ids,
            block_ids_in_use=block_ids_in_use,
        )

        return block_table

    def allocate(
        self,
        seq_group: SequenceGroup,
        seq_metas: List[SequenceMeta],
        block_ids_in_use: Set[int],
    ) -> None:
        assert not (set(seq_meta.seq_id for seq_meta in seq_metas)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq_meta = seq_metas[0]
        assert seq_meta.sequence.status == SequenceStatus.WAITING
        block_table = self._allocate_sequence(seq_meta, block_ids_in_use)
        self.block_tables[seq_meta.seq_id] = block_table

        # Track seq
        self._computed_blocks_tracker.add_seq(seq_meta.seq_id)
        self._last_access_blocks_tracker.add_seq(seq_meta.seq_id)

        # Assign the block table for each sequence.
        for seq_meta in seq_metas[1:]:
            seq = seq_meta.sequence
            assert seq.status == SequenceStatus.WAITING
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._computed_blocks_tracker.add_seq(seq.seq_id)
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
                "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            raise NotImplementedError("Cross-attention is not yet supported.")
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            computed_seq_block_ids.append(
                self._computed_blocks_tracker.
                get_cached_computed_blocks_and_update(
                    seq.seq_id,
                    self.block_tables[seq.seq_id].physical_block_ids))

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._computed_blocks_tracker.add_seq(child_seq.seq_id)
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        raise NotImplementedError("Swap in is not yet supported.")
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap in is not supported.")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError("Swap out is not supported.")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swap out is not supported.")

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given
                sequence_group on to the 'device'.
        """
        raise NotImplementedError("Swapping is not yet supported.")
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_and_reset_block_moving_record(
            self) -> Dict[Tuple[Device, int], Tuple[Device, int]]:
        return self.block_allocator.get_and_reset_block_moving_record()

    def prefetch(self,
                 seq_group: SequenceGroup,
                 seq_metas: List[SequenceMeta],
                 moved_in_blocks: List[Block],
                 block_ids_in_use: Optional[Set[int]] = None,
                 num_usable_blocks: Optional[int] = None,
                 device: Device = Device.GPU) -> int:
        # NOTE(noppanat): Assume exactly one waiting sequence in each sequence
        # group as we do not support CoW.
        assert len(seq_metas) == 1
        assert len(seq_group.get_seqs(status=SequenceStatus.WAITING)) == 1

        cached_blocks_to_move_in = seq_metas[0].cached_blocks_to_move_in
        if num_usable_blocks is None:
            # num_usable_blocks = self.block_allocator.get_num_free_blocks(
            #     device, block_ids_in_use)
            num_usable_blocks = (self.block_allocator.get_num_free_blocks(
                device, block_ids_in_use) - self.watermark_blocks)
        if num_usable_blocks >= 0:
            to_move_in = cached_blocks_to_move_in[:num_usable_blocks]
            self.block_allocator.move_in(to_move_in, block_ids_in_use)
            moved_in_blocks.extend(to_move_in)
            return num_usable_blocks - len(to_move_in)
        return num_usable_blocks

    def free_blocks(self, blocks: List[Block]) -> None:
        for block in blocks:
            self.block_allocator.free(block)

    def print_content(self):
        # TODO(noppanat): Remove this.
        self.block_allocator.print_content(logger)  # type: ignore
