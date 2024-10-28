# enable python only development
# copy compiled files to the current directory directly

import os
import shutil
import subprocess
import sys

# cannot directly `import vllm` , because it will try to
# import from the current directory
output = subprocess.run([sys.executable, "-m", "pip", "show", "vllm"],
                        capture_output=True)

assert output.returncode == 0, "vllm is not installed"

text = output.stdout.decode("utf-8")

package_path = None
for line in text.split("\n"):
    if line.startswith("Location: "):
        package_path = line.split(": ")[1]
        break

assert package_path is not None, "could not find package path"

cwd = os.getcwd()

assert cwd != package_path, "should not import from the current directory"

files_to_copy = [
    "vllm/_C.abi3.so",
    # "vllm/_core_C.abi3.so",
    "vllm/_moe_C.abi3.so",
    "vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so",
    "vllm/vllm_flash_attn/flash_attn_interface.py",
    "vllm/vllm_flash_attn/__init__.py",
    # "vllm/_version.py", # not available in nightly wheels yet
]

for file in files_to_copy:
    src = os.path.join(package_path, file)
    dst = file
    print(f"Copying {src} to {dst}")
    shutil.copyfile(src, dst)

pre_built_vllm_path = os.path.join(package_path, "vllm")
tmp_path = os.path.join(package_path, "vllm_pre_built")
current_vllm_path = os.path.join(cwd, "vllm")

print(f"Renaming {pre_built_vllm_path} to {tmp_path}")
os.rename(pre_built_vllm_path, tmp_path)

print(f"linking {current_vllm_path} to {pre_built_vllm_path}")
os.symlink(current_vllm_path, pre_built_vllm_path)
