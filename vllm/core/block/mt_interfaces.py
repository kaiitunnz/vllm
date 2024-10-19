from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from vllm.core.block.interfaces import (AllocationOutput, Block,
                                        BlockAllocator, BlockState,
                                        DeviceAwareBlockAllocator)
from vllm.utils import Device


@dataclass
class MTAllocationOutput(AllocationOutput):

    def __post_init__(self):
        assert (self.evicted_block is None
                or self.evicted_block.state == BlockState.EVICTED)


class MTBlockAllocator(BlockAllocator):

    @abstractmethod
    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        device: Optional[Device] = None,
        block_ids_in_use: Optional[Set[int]] = None,
    ) -> MTAllocationOutput:
        pass

    @abstractmethod
    def allocate_immutable_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: Optional[Device] = None,
        block_ids_in_use: Optional[Set[int]] = None,
    ) -> MTAllocationOutput:
        pass

    @abstractmethod
    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: Optional[Device] = None,
        block_ids_in_use: Optional[Set[int]] = None,
    ) -> Sequence[MTAllocationOutput]:
        pass

    @abstractmethod
    def allocate_cached_block(self, block: Block) -> MTAllocationOutput:
        pass

    @abstractmethod
    def fork(self, last_block: Block) -> Sequence[MTAllocationOutput]:
        pass

    @abstractmethod
    def swap_in(self, blocks: List[Block]) -> None:
        pass

    @abstractmethod
    def move_out(self, block: Block) -> None:
        pass

    @abstractmethod
    def move_in(self,
                block: Block,
                evictable: bool = False) -> MTAllocationOutput:
        pass

    @abstractmethod
    def destroy(self, block: Block) -> None:
        pass


class MTDeviceAwareBlockAllocator(DeviceAwareBlockAllocator):

    @abstractmethod
    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        device: Device,
        block_ids_in_use: Optional[Set[int]] = None,
    ) -> MTAllocationOutput:
        pass

    @abstractmethod
    def allocate_immutable_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: Device,
        block_ids_in_use: Optional[Set[int]] = None,
    ) -> MTAllocationOutput:
        pass

    @abstractmethod
    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: Device,
        block_ids_in_use: Optional[Set[int]] = None,
    ) -> Sequence[MTAllocationOutput]:
        pass

    @abstractmethod
    def allocate_cached_block(self, block: Block) -> MTAllocationOutput:
        pass

    @abstractmethod
    def fork(self, last_block: Block) -> Sequence[MTAllocationOutput]:
        pass

    @abstractmethod
    def move_in(self, blocks: List[Block]) -> None:
        pass

    @abstractmethod
    def move_out(self, blocks: List[Block]) -> None:
        pass

    @abstractmethod
    def destroy(self, block: Block) -> None:
        pass

    @abstractmethod
    def get_and_reset_block_moving_record(
        self, ) -> Dict[Tuple[Device, int], Tuple[Device, int]]:
        pass

    @abstractmethod
    def get_cached_block(self, content_hash: int) -> Optional[Block]:
        pass

    @abstractmethod
    def get_device(self, block: Block) -> Device:
        pass

    @abstractmethod
    def get_device_from_id(self, block_id: int) -> Device:
        pass

    @abstractmethod
    def get_device_tier(self, block: Block) -> int:
        pass

    @abstractmethod
    def get_device_tier_from_id(self, block_id: int) -> int:
        pass
