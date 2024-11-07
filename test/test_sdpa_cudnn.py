from collections import namedtuple
from functools import partial

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

import triton.profiler as proton

from hopper.flash_attn_interface import flash_attn_func as flash_attn_func_hopper


# Set random seed for reproducibility
torch.manual_seed(42)
torch.set_default_device("cuda")

SdpaShape = namedtuple("Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"])
Tolerances = namedtuple("Tolerances", ["atol", "rtol"])

def compiled_sdpa():
    return torch.compile(
        scaled_dot_product_attention,
        fullgraph=True,
        backend="inductor",
        mode="max-autotune",
    )

def warmup(iters: int, backend, f, *args, **kwargs) -> None:
    for _ in range(iters):
        with sdpa_kernel(backends=[backend]):
            _ = f(*args, **kwargs)


"""
Flux q, k, v shapes:
    q = [1, 24, 4608, 128]
    k = [1, 24, 4608, 128]
    v = [1, 24, 4608, 128]


FA -> (batch, seqlen, nheads, headdim)
Torch sdpa expects -> (batch, nheads, seqlen, headdim)

ref: 
    torch.functional: https://github.com/pytorch/pytorch/blob/8231180147a096a703d8891756068c89365292e0/torch/nn/functional.py#L5617
"""

device = torch.device("cuda")
dtype = torch.bfloat16
is_causal = False
make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=False)
batch = 1
num_heads = 24
head_dim = 128
seq_len = 4608

flops = 4 * batch * seq_len**2 * num_heads * head_dim // (2 if is_causal else 1)
warmup_iter = 10

q_shape = SdpaShape(batch, num_heads, seq_len, head_dim)
k_shape = SdpaShape(batch, num_heads, seq_len, head_dim)
v_shape = SdpaShape(batch, num_heads, seq_len, head_dim)

query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

assert query.shape == q_shape
assert key.shape == k_shape
assert value.shape == v_shape

# warmup for sdpa_kernel
warmup(
    warmup_iter,
    SDPBackend.CUDNN_ATTENTION,
    torch.nn.functional.scaled_dot_product_attention,
    query,
    key,
    value,
    is_causal=is_causal,
)

proton.start()
# cuDNN attention
with proton.scope("torch_scaled_dot_product_cudnn_attention", metrics={"flops": flops}):
    with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
        attn_out_sdpa_cudnn = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal
        )
# torch sdpa native
for _ in range(warmup_iter):
    _ = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )
with proton.scope("torch_scaled_dot_product_attention", metrics={"flops": flops}):
    attn_out_sdpa = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )
# torch aten explict cuDNN op call
for _ in range(warmup_iter):
    _ = torch.ops.aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=None,
        compute_log_sumexp=False,
        dropout_p=0.0,
        is_causal=is_causal,
    )
with proton.scope(
    "torch.ops.aten._scaled_dot_product_cudnn_attention", metrics={"flops": flops}
):
    attn_out_aten_sdpa_cudnn = torch.ops.aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=None,
        compute_log_sumexp=False,
        dropout_p=0.0,
        is_causal=is_causal,
    )
# torch compiled sdpa
for _ in range(warmup_iter):
    _ = compiled_sdpa()(query, key, value, is_causal=is_causal)
with proton.scope("torch_scaled_dot_product_attention_compiled", metrics={"flops": flops}):
    flash_attention_compiled_op = compiled_sdpa()
    attn_out_sdpa_compiled = flash_attention_compiled_op(query, key, value, is_causal=is_causal)
# FlashAttention-3 Hopper
for _ in range(warmup_iter):
    _, _ = flash_attn_func_hopper(query, key, value)
with proton.scope("flash_attention_hopper", metrics={"flops": flops}):
    attn_output_fa_hopper, _ = flash_attn_func_hopper(query, key, value, causal=is_causal)

proton.finalize()
