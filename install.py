import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from tools.git_utils import checkout_submodules
from tools.python_utils import pip_install_requirements

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_PATH = Path(os.path.abspath(__file__)).parent

def install_fa2(compile=False):
    if compile:
        # compile from source (slow)
        FA2_PATH = REPO_PATH.joinpath("submodules", "flash-attention")
        cmd = [sys.executable, "setup.py", "install"]
        subprocess.check_call(cmd, cwd=str(FA2_PATH.resolve()))
    else:
        # Install the pre-built binary
        cmd = ["pip", "install", "flash-attn", "--no-build-isolation"]
        subprocess.check_call(cmd)


def install_fa3():
    FA3_PATH = REPO_PATH.joinpath("submodules", "flash-attention", "hopper")
    cmd = [sys.executable, "setup.py", "install"]
    subprocess.check_call(cmd, cwd=str(FA3_PATH.resolve()))

def install_xformers():
    os_env = os.environ.copy()
    os_env["TORCH_CUDA_ARCH_LIST"] = "9.0;9.0a"
    XFORMERS_PATH = REPO_PATH.joinpath("submodules", "xformers")
    cmd = ["pip", "install", "-e", XFORMERS_PATH]
    subprocess.check_call(cmd, env=os_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--fa2", action="store_true", help="Install optional flash_attention 2 kernels"
    )
    parser.add_argument(
        "--fa2-compile",
        action="store_true",
        help="Install optional flash_attention 2 kernels from source.",
    )
    parser.add_argument(
        "--fa3", action="store_true", help="Install optional flash_attention 3 kernels"
    )
    parser.add_argument("--xformers", action="store_true", help="Install xformers")
    parser.add_argument(
        "--all", action="store_true", help="Install all custom kernel repos"
    )
    args = parser.parse_args()
    # install framework dependencies
    pip_install_requirements("requirements.txt")
    # checkout submodules
    checkout_submodules(REPO_PATH)
    # install submodules
    if args.fa2 or args.all:
        logger.info("[test-attn] installing fa2 from source...")
        install_fa2(compile=True)
    if args.fa3 or args.all:
        logger.info("[test-attn] installing fa3...")
        install_fa3()
    if args.xformers or args.all:
        logger.info("[test-attn] installing xformers...")
        install_xformers()
    logger.info("[test-attn] installation complete!")

