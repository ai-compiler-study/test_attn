from functools import partial

import torch
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.benchmarks.utils import benchmark_main_helper, create_argparser

from torch.nn.functional import scaled_dot_product_attention
from torch.utils import benchmark

ABS_TOL = 5e-3
REL_TOL = 1e-1


xformers_flash3 = torch.compile(
    xformers.ops.fmha.flash3.FwOp,
    fullgraph=True,
    backend="inductor",
    mode="max-autotune",
)

dtype = torch.bfloat16
device = "cuda"
batch_size = 1
print(dtype)
torch.random.manual_seed(0)

batch_size = 1
head_dim = 256
seqlen_q = 512
seqlen_k = 512
n_heads = 6
p = 0.0

q = torch.randn(
    batch_size,
    seqlen_q,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)  # B, S, H, D
k = torch.randn(
    batch_size,
    seqlen_k,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)  # B, S, H, D
v = torch.randn(
    batch_size,
    seqlen_k,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)  # B, S, H, D

dtype_str = {
    torch.bfloat16: "bf16",
    torch.half: "fp16",
}[dtype]
sub_label = f"{dtype_str}-{batch_size}-{seqlen_k}-{n_heads}-{head_dim}, p={p}"

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
measurement = benchmark.Timer(
    stmt="fn(q, k, v, attn_bias, p)",
    globals={
        "q": q,
        "k": k,
        "v": v,
        "attn_bias": None,
        "p": p,
        "fn": partial(xformers.ops.memory_efficient_attention, op=[xformers_flash3]),
    },
    label=f"attention (attn_bias={None})",
    description=xformers.ops.fmha.flash3.FwOp.NAME,
    sub_label=sub_label,
).blocked_autorange(min_run_time=0.5)

torch.cuda.synchronize()
measurement_name = measurement.task_spec.description
print(f"Mesurement op name: {measurement_name}")
print(f"Measurement median: {measurement.median * 1e6:.5f} ms")
print(f"Measurement mean: {measurement.mean * 1e6:.5f} ms")
print(measurement.task_spec.sub_label)
print(measurement.task_spec.label)

