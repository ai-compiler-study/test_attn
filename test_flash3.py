import subprocess
import os
import sys
from typing import Literal, Tuple

import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    import xformers.ops
    import xformers.ops.fmha as fmha

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

ABS_TOL = 5e-3
REL_TOL = 1e-1

VERBOSE = os.getenv("VERBOSE", "0") == "1"

if VERBOSE:
    XFORMERS_INFO = f"{sys.executable} -m xformers.info"
    subprocess.run(XFORMERS_INFO.split())
torch.set_default_device("cuda")

op = xformers.ops.fmha.flash3.FwOp
if op is fmha.flash3.FwOp and op.is_available():
    print(f"xformers_ops_fmha_flash3 supported: {HAS_FLASH}")


def compiled_xformers_flash_hopper(q, k, v):
    xformers_flash3 = torch.compile(
        xformers.ops.fmha.flash3.FwOp,
        fullgraph=True,
        backend="inductor",
        mode="max-autotune",
    )
    softmax_scale = q.size(-1) ** -0.5

    return fmha.memory_efficient_attention_forward(  # noqa: E731
        q,
        k,
        v,
        scale=softmax_scale,
        op=xformers_flash3,
    )

if __name__ == "__main__":
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 1
    print(dtype)
    torch.random.manual_seed(0)

    head_dim = 256
    seqlen_q = 512
    seqlen_k = 512
    n_heads = 6

    q = torch.randn(
        batch_size,
        seqlen_q,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    ) # B, S, H, D
    k = torch.randn(
        batch_size,
        seqlen_k,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    ) # B, S, H, D
    v = torch.randn(
        batch_size,
        seqlen_k,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    ) # B, S, H, D

    attn_output_xformers_flash_hopper = compiled_xformers_flash_hopper(q, k, v)
    attn_torch_ref = scaled_dot_product_attention(q, k, v)
    print(attn_output_xformers_flash_hopper.shape)
    print(attn_torch_ref.shape)
    torch.allclose(attn_output_xformers_flash_hopper, attn_torch_ref, atol=ABS_TOL, rtol=REL_TOL)
