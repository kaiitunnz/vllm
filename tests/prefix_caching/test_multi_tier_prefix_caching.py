"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""

import pytest

from tests.kernels.utils import override_backend_env_variable

from ..models.utils import check_outputs_equal

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["FLASH_ATTN"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("cached_position", [0, 1])
@pytest.mark.parametrize(",".join([
    "enable_prefix_caching",
    "enable_multi_tier_prefix_caching",
    "enable_async_swapping",
    "enable_prefix_aware_scheduling",
    "enable_async_prefetching",
]), [
    (False, False, False, False, False),
    (True, False, False, False, False),
    (True, True, True, True, True),
    (True, True, False, False, False),
    (True, True, True, False, False),
    (True, True, False, True, False),
    (True, True, True, True, False),
    (True, True, True, False, True),
])
def test_mixed_requests(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    backend: str,
    dtype: str,
    max_tokens: int,
    cached_position: int,
    enable_prefix_caching: bool,
    enable_multi_tier_prefix_caching: bool,
    enable_async_swapping: bool,
    enable_prefix_aware_scheduling: bool,
    enable_async_prefetching: bool,
    monkeypatch,
) -> None:
    """
    Test the case when some sequences have the prefix cache hit
    and the others don't. The cached position determines where 
    the sequence is at among the batch of prefills.
    """
    override_backend_env_variable(monkeypatch, backend)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    cached_prompt = example_prompts[cached_position]
    with vllm_runner(
            model,
            dtype=dtype,
            preemption_mode="recompute",
            num_gpu_blocks_override=512,
            num_cpu_blocks_override=512,
            max_model_len=8192,
            block_size=16,
            enable_prefix_caching=enable_prefix_caching,
            enable_multi_tier_prefix_caching=enable_multi_tier_prefix_caching,
            enable_async_swapping=enable_async_swapping,
            enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
            enable_async_prefetching=enable_async_prefetching,
    ) as vllm_model:
        # Run the first prompt so the cache is populated
        vllm_outputs = vllm_model.generate_greedy([cached_prompt], max_tokens)

        # Run all the promopts
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
