# %%
"""
Demonstrate how to profile Triton kernels generated by Inductor.
We will also learn how to inspect the performance metrics to identify
any interesting optimization opportunites.
"""
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch

torch.__version__


# %%
def run(command: str) -> None:
    output, errors = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=-1,
        close_fds=True,
        env=os.environ,
    ).communicate()
    print(output.decode("utf-8"))


run(f"{sys.executable} --version")
# %%
"""
TORCHINDUCTOR_BENCHMARK_KERNEL: enables Inductor to save standalone benchmark Triton kernels
TORCHINDUCTOR_CACHE_DIR: specifies where to save standalone benchmark Triton kernels
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES: generates a unique and informative name for each Triton kernel
TORCHINDUCTOR_PROFILE: enables the builtin bandwidth profiler
TORCHINDUCTOR_PROFILE_OUTPUT: specifies where to save the profiling result summary
TORCHINDUCTOR_ENABLED_METRIC_TABLES: enables Inductor to collect a number of performance-related metrics
ref: https://github.com/pytorch/pytorch/blob/24cecf06d7c3d2f38870cdb01f1e322678abea9c/torch/_inductor/metrics.py#L229
"""
compilation_targets_dir = "./targets"

env = os.environ.copy()
env["TORCHINDUCTOR_CACHE_DIR"] = compilation_targets_dir
env["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
env["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
env["TORCHINDUCTOR_PROFILE"] = "1"
env["TORCHINDUCTOR_PROFILE_OUTPUT"] = "./profiler_output.txt"
env['TORCHINDUCTOR_ENABLED_METRIC_TABLES'] = "kernel_metadata"

wd: str = Path(__file__).parent.absolute()
benchmark_script: str = os.path.join(wd, "benchmark_flux.py")
model_id: str = "black-forest-labs/FLUX.1-dev"
height: int = 1024
width: int = 1024
diffusion_steps: int = 28
max_sequence_length: int = 512
use_torch_compile: bool = True
fullgraph: bool = True
use_nexfort: bool = False
debug: bool = True
iters: int = 10
seed: int = 42

command: str = (
    f"{sys.executable} {benchmark_script} "
    f"--model-id {model_id} "
    f"--height {height} "
    f"--width {width} "
    f"--diffusion-steps {diffusion_steps} "
    f"--max-sequence-length {max_sequence_length} "
    "--use-torch-compile" if use_torch_compile else ""
)
print(command)
print(shlex.split(command))
# %%
subprocess.run(shlex.split(command), env=env)
# %%
command = f"find {compilation_targets_dir} -name '*.py'"
subprocess.run(shlex.split(command), env=env)
# %%
command = f"cat ./profiler-outputs/l2/cl2zcpw6lel54uhriisprrv62t5nzrm3q4zcq6puyyejb6xxe32k.py"