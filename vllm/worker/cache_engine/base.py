from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.worker.cache_engine.cache_engine import CacheEngine
from vllm.worker.cache_engine.mt_cache_engine import MTCacheEngine
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


def create_cache_engine(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    device_config: DeviceConfig,
) -> CacheEngine:
    if cache_config.enable_async_swapping:
        return MTCacheEngine(cache_config, model_config, parallel_config,
                                  device_config)
    return CacheEngine(cache_config, model_config, parallel_config,
                       device_config)


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
