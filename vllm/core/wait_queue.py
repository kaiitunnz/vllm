import itertools
from abc import abstractmethod
from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from vllm.core.mt_block_manager import MTBlockSpaceManager, SequenceMeta
from vllm.sequence import SequenceGroup, SequenceStatus


class WaitQueueBase:

    def __init__(self, waiting: Optional[Deque[SequenceGroup]] = None):
        self._waiting: Deque[SequenceGroup] = (deque()
                                               if waiting is None else waiting)

    @abstractmethod
    def append(self, seq_group: SequenceGroup) -> None:
        pass

    @abstractmethod
    def remove(self, seq_group: SequenceGroup) -> None:
        pass

    @abstractmethod
    def popleft(self) -> SequenceGroup:
        pass

    @abstractmethod
    def appendleft(self, seq_group: SequenceGroup) -> None:
        pass

    @abstractmethod
    def extendleft(self, seq_groups: Iterable[SequenceGroup]) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> SequenceGroup:
        pass

    def __bool__(self) -> bool:
        return bool(len(self))

    @abstractmethod
    def __contains__(self, seq_group: SequenceGroup) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[SequenceGroup]:
        pass


class WaitQueue(WaitQueueBase):

    def append(self, seq_group: SequenceGroup) -> None:
        self._waiting.append(seq_group)

    def remove(self, seq_group: SequenceGroup) -> None:
        self._waiting.remove(seq_group)

    def popleft(self) -> SequenceGroup:
        return self._waiting.popleft()

    def appendleft(self, seq_group: SequenceGroup) -> None:
        self._waiting.appendleft(seq_group)

    def extendleft(self, seq_groups: Iterable[SequenceGroup]) -> None:
        self._waiting.extendleft(seq_groups)

    def __len__(self) -> int:
        return len(self._waiting)

    def __getitem__(self, index: int) -> SequenceGroup:
        return self._waiting[index]

    def __contains__(self, seq_group: SequenceGroup) -> bool:
        return seq_group in self._waiting

    def __iter__(self) -> Iterator[SequenceGroup]:
        return iter(self._waiting)


class MTWaitQueueBase(WaitQueueBase):
    STATUS: SequenceStatus = SequenceStatus.WAITING
    Item = Tuple[SequenceGroup, List[SequenceMeta]]

    class ContextManager:

        @abstractmethod
        def peekleft(self) -> "MTWaitQueueBase.Item":
            pass

        @abstractmethod
        def popleft(self) -> SequenceGroup:
            pass

        @abstractmethod
        def __bool__(self) -> bool:
            pass

        @abstractmethod
        def get_prefetchable(self) -> List["MTWaitQueueBase.Item"]:
            pass

    @abstractmethod
    def __enter__(self) -> "MTWaitQueueBase.ContextManager":
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback) -> None:
        pass


class MTWaitQueue(MTWaitQueueBase):

    class ContextManager(MTWaitQueueBase.ContextManager):

        def __init__(self, block_manager: MTBlockSpaceManager,
                     waiting: Deque[SequenceGroup]):
            self._block_manager: MTBlockSpaceManager = block_manager
            self._waiting: Deque[SequenceGroup] = waiting
            self._seq_meta_cache: Dict[SequenceGroup, List[SequenceMeta]] = {}

        def peekleft(self) -> "MTWaitQueue.Item":
            seq_group = self._waiting[0]
            if seq_group not in self._seq_meta_cache:
                seq_metas = self._block_manager.process_sequence_group(
                    seq_group, status=MTWaitQueue.STATUS)
                self._seq_meta_cache[seq_group] = seq_metas
            else:
                seq_metas = self._seq_meta_cache[seq_group]
            return seq_group, seq_metas

        def popleft(self) -> SequenceGroup:
            return self._waiting.popleft()

        def _reset(self) -> None:
            self._seq_meta_cache.clear()

        def __bool__(self) -> bool:
            return bool(self._waiting)

        def get_prefetchable(self) -> List["MTWaitQueue.Item"]:
            # Default MTWaitQueue does not support prefetching.
            return []

    def __init__(self,
                 block_manager: MTBlockSpaceManager,
                 waiting: Optional[Deque[SequenceGroup]] = None):
        super().__init__(waiting)
        self._context_manager = MTWaitQueue.ContextManager(
            block_manager, self._waiting)

    def append(self, seq_group: SequenceGroup) -> None:
        self._waiting.append(seq_group)

    def remove(self, seq_group: SequenceGroup) -> None:
        self._waiting.remove(seq_group)

    def popleft(self) -> SequenceGroup:
        return self._waiting.popleft()

    def appendleft(self, seq_group: SequenceGroup) -> None:
        self._waiting.appendleft(seq_group)

    def extendleft(self, seq_groups: Iterable[SequenceGroup]) -> None:
        self._waiting.extendleft(seq_groups)

    def __len__(self) -> int:
        return len(self._waiting)

    def __getitem__(self, index: int) -> SequenceGroup:
        return self._waiting[index]

    def __bool__(self) -> bool:
        return bool(self._waiting)

    def __contains__(self, seq_group: SequenceGroup) -> bool:
        return seq_group in self._waiting

    def __iter__(self) -> Iterator[SequenceGroup]:
        return iter(self._waiting)

    def __enter__(self):
        self._context_manager._reset()
        return self._context_manager

    def __exit__(self, type, value, traceback) -> None:
        self._context_manager._reset()


class PrefixAwareWaitQueue(MTWaitQueueBase):

    class ContextManager(MTWaitQueueBase.ContextManager):

        def __init__(self,
                     block_manager: MTBlockSpaceManager,
                     waiting: Deque[SequenceGroup],
                     dispenser: Deque[SequenceGroup],
                     window_size: int,
                     reorder_on_reset: bool = False):
            self._block_manager: MTBlockSpaceManager = block_manager
            self._waiting: Deque[SequenceGroup] = waiting
            self._dispenser: Deque[SequenceGroup] = dispenser
            self._window_size: int = window_size
            self._reorder_on_reset: bool = reorder_on_reset
            self._seq_meta_cache: Dict[SequenceGroup, List[SequenceMeta]] = {}

        def peekleft(self) -> "PrefixAwareWaitQueue.Item":
            seq_group = self.get(0)
            if seq_group not in self._seq_meta_cache:
                if self._reorder_on_reset:
                    next_seq_groups = self._process_and_sort(self._dispenser)
                    self._seq_meta_cache.update(next_seq_groups)
                    self._dispenser.clear()
                    self._dispenser.extend(seq_group
                                           for seq_group, _ in next_seq_groups)
                    seq_metas = self._seq_meta_cache[seq_group]
                else:
                    seq_metas = self._block_manager.process_sequence_group(
                        seq_group, status=MTWaitQueue.STATUS)
                    self._seq_meta_cache[seq_group] = seq_metas
            else:
                seq_metas = self._seq_meta_cache[seq_group]
            return seq_group, seq_metas

        def popleft(self) -> SequenceGroup:
            return self._dispenser.popleft()

        def _process_and_sort(
            self, seq_groups: Iterable[SequenceGroup]
        ) -> List["PrefixAwareWaitQueue.Item"]:

            def process_seq_group(
                    seq_group: SequenceGroup) -> "PrefixAwareWaitQueue.Item":
                assert seq_group not in self._seq_meta_cache
                seq_metas = self._block_manager.process_sequence_group(
                    seq_group, status=PrefixAwareWaitQueue.STATUS)
                return seq_group, seq_metas

            def sort_key(
                    item: "PrefixAwareWaitQueue.Item"
            ) -> Tuple[int, int, float]:
                seq_group, seq_metas = item
                seq_meta = seq_metas[0]
                num_cached_blocks_in_gpu = len(seq_meta.cached_blocks)
                num_total_cached_blocks = num_cached_blocks_in_gpu + len(
                    seq_meta.cached_blocks_to_move_in)
                return (-num_cached_blocks_in_gpu, -num_total_cached_blocks,
                        seq_group.arrival_time)

            return sorted(
                [process_seq_group(seq_group) for seq_group in seq_groups],
                key=sort_key)

        def _grow_dispenser(self) -> None:
            num_to_process = min(self._window_size, len(self._waiting))
            next_seq_groups = self._process_and_sort(
                [self._waiting.popleft() for _ in range(num_to_process)])
            for seq_group, seq_metas in next_seq_groups:
                self._seq_meta_cache[seq_group] = seq_metas
                self._dispenser.append(seq_group)

        def _reset(self) -> None:
            self._seq_meta_cache.clear()

        def get(self, index: int) -> SequenceGroup:
            while index >= len(self._dispenser):
                self._grow_dispenser()
            return self._dispenser[index]

        def __bool__(self) -> bool:
            return bool(self._dispenser) or bool(self._waiting)

        def get_prefetchable(self) -> List["PrefixAwareWaitQueue.Item"]:
            prefetchable: List["PrefixAwareWaitQueue.Item"] = []
            for seq_group in self._dispenser:
                seq_metas = self._seq_meta_cache.get(seq_group, None)
                if seq_metas is None:
                    seq_metas = self._block_manager.process_sequence_group(
                        seq_group, status=PrefixAwareWaitQueue.STATUS)
                    self._seq_meta_cache[seq_group] = seq_metas
                prefetchable.append((seq_group, seq_metas))
            return prefetchable

    def __init__(self,
                 block_manager: MTBlockSpaceManager,
                 waiting: Optional[Deque[SequenceGroup]] = None,
                 window_size: int = 10):
        super().__init__(waiting)
        self._block_manager: MTBlockSpaceManager = block_manager

        self._dispenser: Deque[SequenceGroup] = deque()
        self._context_manager = PrefixAwareWaitQueue.ContextManager(
            block_manager, self._waiting, self._dispenser, window_size)

    def append(self, seq_group: SequenceGroup) -> None:
        self._waiting.append(seq_group)

    def remove(self, seq_group: SequenceGroup) -> None:
        try:
            self._dispenser.remove(seq_group)
        except ValueError:
            self._waiting.remove(seq_group)

    def popleft(self) -> SequenceGroup:
        return self._dispenser.popleft()

    def appendleft(self, seq_group: SequenceGroup) -> None:
        self._dispenser.appendleft(seq_group)

    def extendleft(self, seq_groups: Iterable[SequenceGroup]) -> None:
        self._dispenser.extendleft(seq_groups)

    def peekleft(self) -> "PrefixAwareWaitQueue.Item":
        seq_group = self[0]
        seq_metas = self._block_manager.process_sequence_group(
            seq_group, status=MTWaitQueue.STATUS)
        return seq_group, seq_metas

    def __len__(self) -> int:
        return len(self._dispenser) + len(self._waiting)

    def __getitem__(self, index: int) -> SequenceGroup:
        if index < len(self._dispenser):
            return self._dispenser[index]
        return self._waiting[index - len(self._dispenser)]

    def __bool__(self) -> bool:
        return bool(self._dispenser) or bool(self._waiting)

    def __contains__(self, seq_group: SequenceGroup) -> bool:
        return seq_group in self._dispenser or seq_group in self._waiting

    def __iter__(self) -> Iterator[SequenceGroup]:
        return itertools.chain(self._dispenser, self._waiting)

    def __enter__(self):
        self._context_manager._reset()
        return self._context_manager

    def __exit__(self, type, value, traceback) -> None:
        self._context_manager._reset()
