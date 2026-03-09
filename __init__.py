"""Compatibility shim for source-tree imports.

Helium adds `src/` to `sys.path`, which makes Python discover the vLLM git
submodule directory (`src/vllm/`) before the actual package directory
(`src/vllm/vllm/`). Load the inner package under the top-level `vllm` name so
imports like `vllm.envs` and `from vllm import SamplingParams` resolve exactly
as they do in a normal install.
"""

from importlib.util import spec_from_file_location
from pathlib import Path
import sys

_pkg_dir = Path(__file__).resolve().parent / "vllm"
_pkg_init = _pkg_dir / "__init__.py"
_module = sys.modules[__name__]
_module.__file__ = str(_pkg_init)
_module.__path__ = [str(_pkg_dir)]  # type: ignore[attr-defined]
_module.__package__ = __name__
_module.__spec__ = spec_from_file_location(  # type: ignore[attr-defined]
    __name__,
    _pkg_init,
    submodule_search_locations=[str(_pkg_dir)],
)

assert _module.__spec__ is not None and _module.__spec__.loader is not None
_module.__spec__.loader.exec_module(_module)
sys.modules[f"{__name__}.vllm"] = _module
