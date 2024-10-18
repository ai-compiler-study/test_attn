import math
from contextlib import contextmanager
from typing import Literal, Tuple

import torch
from flash_attn.flash_attn_interface import flash_attn_func
from hopper.flash_attn_interface import flash_attn_func as flash_attn_func_hopper
from torch.nn.attention import SDPBackend
from torch.nn.attention._flex_attention import _flex_attention
from transformer_engine.pytorch.attention import DotProductAttention
import transformer_engine.pytorch as te
from triton.ops import attention as attention_triton

try:
    import xformers.ops
    import xformers.ops.fmha as fmha

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

compiled_flex_attention = torch.compile(_flex_attention, dynamic=False)

def compiled_flash_attention_v2(q, k, v):
    #kv_len_max = 1024 * 12
    #torch._dynamo.mark_dynamic(k, 1, min=512, max=kv_len_max)

    flash_attention_v2 = torch.compile(
        flash_attn_func,
        fullgraph=True,
        backend="inductor",
        mode="max-autotune-no-cudagraphs",
    )
    return flash_attention_v2(q, k, v)

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

@contextmanager
def time_with_cuda_event(name, flops):
    """
    Context manager to time CUDA operations and compute MFU.

    Args:
        name (str): Name of the attention implementation.
        flops (float): Number of floating-point operations.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(name)
    start.record()
    yield
    end.record()
    torch.cuda.nvtx.range_pop()
    end.synchronize()

    elapsed_time = start.elapsed_time(end)
    mfu = flops / (elapsed_time * 0.989 * 1e12)
    print(f"{name} took {elapsed_time:.4f} ms, mfu: {mfu:.2f}")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    query_len, key_len = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(query_len, key_len, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(
            query_len, key_len, dtype=torch.bool, device=query.device
        ).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def get_qkv(
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    mqa: bool,
    layout: Literal["bhsd", "sbhd"] = "bhsd"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert layout in ["bhsd", "sbhd"]

    if mqa:
        q = torch.randn(
            batch_size, num_heads, q_len, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            batch_size, 1, kv_len, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            batch_size, 1, kv_len, head_dim, device="cuda", dtype=torch.float16
        )
    else:
        q = torch.randn(
            batch_size, num_heads, q_len, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            batch_size, num_heads, kv_len, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            batch_size, num_heads, kv_len, head_dim, device="cuda", dtype=torch.float16
        )

    if layout == "sbhd":
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    return q, k, v


if __name__ == "__main__":
    batch_size = 1
    num_heads = 8
    head_dim = 128
    q_len = 1024 * 12
    kv_lens = [512, q_len]
    warmup_iter = 10
    test_iter = 100
    mqa = False

    for kv_len in kv_lens:
        torch.cuda.empty_cache()

        q, k, v = get_qkv(
            batch_size, num_heads, q_len, kv_len, head_dim, mqa, layout="bhsd"
        )
        softmax_scale = q.size(-1) ** -0.5

        flops = 4 * q_len * kv_len * num_heads * head_dim * test_iter
        flops_bwd = flops * 2

        torch.cuda.profiler.start()
        for _ in range(warmup_iter):
            scaled_dot_product_attention(q, k, v, is_causal=False)
        with time_with_cuda_event(
            f"scaled_dot_product_attention_torch_fwd, kv_len={kv_len}", flops
        ):
            for _ in range(test_iter):
                attn_output_torch = scaled_dot_product_attention(
                    q, k, v, is_causal=False
                )

        for _ in range(warmup_iter):
            with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=False
                )
        with time_with_cuda_event(
            f"scaled_dot_product_attention_torch_math_fwd, kv_len={kv_len}", flops
        ):
            with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
                for _ in range(test_iter):
                    attn_output_torch_math = (
                        torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, is_causal=False
                        )
                    )

        for _ in range(warmup_iter):
            _ = compiled_flex_attention(q, k, v)
        with time_with_cuda_event("flex_attention_fwd", flops):
            for _ in range(test_iter):
                flex_attn_output = compiled_flex_attention(q, k, v)

        q, k, v = get_qkv(
            batch_size, num_heads, q_len, kv_len, head_dim, mqa, layout="sbhd"
        )
        te_fused_attn = DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            qkv_format="bshd",
            attn_mask_type="no_mask",
            num_gqa_groups=1 if mqa else None,
        )

        for _ in range(warmup_iter):
            _ = te_fused_attn(q, k, v)
        with time_with_cuda_event(f"cudnn_attention_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_te = te_fused_attn(q, k, v)

        for _ in range(warmup_iter):
            _ = flash_attn_func(q, k, v)
        with time_with_cuda_event(f"flash_attn_func_v2_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_fa_v2 = flash_attn_func(q, k, v)

        for _ in range(warmup_iter):
            _ = compiled_flash_attention_v2(q, k, v)
        with time_with_cuda_event(f"flash_attn_func_v2_fwd_compiled, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                with torch.no_grad():
                    attn_output_fa_v2_compiled = compiled_flash_attention_v2(q, k, v)

        for _ in range(warmup_iter):
            _ = fmha.memory_efficient_attention_forward(  # noqa: E731
                    q,
                    k,
                    v,
                    scale=softmax_scale,
                    op=xformers.ops.fmha.cutlass.FwOp,
                )
        with time_with_cuda_event(
            f"xformers_flash_attn_cutlass_fwd, kv_len={kv_len}", flops
        ):
            for _ in range(test_iter):
                _ = fmha.memory_efficient_attention_forward(  # noqa: E731
                    q,
                    k,
                    v,
                    scale=softmax_scale,
                    op=xformers.ops.fmha.cutlass.FwOp,
                )

        for _ in range(warmup_iter):
            _ = compiled_xformers_flash_hopper(q, k, v)
        with time_with_cuda_event(f"xformers_flash_hopper_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_xformers_flash_hopper = compiled_xformers_flash_hopper(q, k, v)

        for _ in range(warmup_iter):
            _ = attention_triton(q, k, v, True, softmax_scale)
        with time_with_cuda_event(f"triton_attention_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_triton = attention_triton(q, k, v, True, softmax_scale)

        for _ in range(warmup_iter):
            _, _ = flash_attn_func_hopper(q, k, v)
        with time_with_cuda_event(
            f"flash_attn_func_hopper_fwd, kv_len={kv_len}", flops
        ):
            for _ in range(test_iter):
                attn_output_fa_hopper, softmax_lse = flash_attn_func_hopper(q, k, v)
        torch.cuda.profiler.stop()