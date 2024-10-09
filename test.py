import torch
from contextlib import contextmanager
from torch.nn.attention import SDPBackend
from transformer_engine.pytorch.attention import FusedAttention, DotProductAttention
import transformer_engine_torch as tex
import math 
from flash_attn.flash_attn_interface import flash_attn_func
from hopper.flash_attn_interface import flash_attn_func as flash_attn_func_hopper
from torch.nn.attention.flex_attention import flex_attention

@torch.compile
def compiled_flex_attention(q, k, v):
    return flex_attention(q, k, v)


from typing import Literal

@contextmanager
def time_with_cuda_event(name, flops):
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
    print(f"{name} took {elapsed_time} ms, mfu: {mfu:.2f}")

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
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

def get_qkv(batch_size, num_heads, q_len, kv_len, head_dim, mqa, layout=Literal["bshd, sbhd"]):
    assert layout in ["bhsd", "sbhd"]

    if mqa:
        q = torch.randn(batch_size, num_heads, q_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, 1, kv_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, 1, kv_len, head_dim, device='cuda', dtype=torch.float16)
    else:
        q = torch.randn(batch_size, num_heads, q_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, kv_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, kv_len, head_dim, device='cuda', dtype=torch.float16)

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
        
        q, k, v = get_qkv(batch_size, num_heads, q_len, kv_len, head_dim, mqa, layout="bhsd")

        flops = 4 * q_len * kv_len * num_heads * head_dim * test_iter
        flops_bwd = flops * 2

        torch.cuda.profiler.start()
        for _ in range(warmup_iter):
            scaled_dot_product_attention(q, k, v, is_causal=False)
        with time_with_cuda_event(f"scaled_dot_product_attention_torch_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_torch1 = scaled_dot_product_attention(q, k, v, is_causal=False)

        for _ in range(warmup_iter):
            attn_output_torch1.backward(attn_output_torch1.detach(), retain_graph=True)
        with time_with_cuda_event(f"scaled_dot_product_attention_torch_bwd, kv_len={kv_len}", flops_bwd):
            for _ in range(test_iter):
                attn_output_torch1.backward(attn_output_torch1.detach(), retain_graph=True)
        attn_output_torch1.backward(attn_output_torch1.detach())

        for _ in range(warmup_iter):
            with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
                attn_output_torch2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        with time_with_cuda_event(f"scaled_dot_product_attention_torch_math_fwd, kv_len={kv_len}", flops):
            with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
                for _ in range(test_iter):
                    attn_output_torch2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        for _ in range(warmup_iter):
            attn_output_torch2.backward(attn_output_torch2.detach(), retain_graph=True)
        with time_with_cuda_event(f"scaled_dot_product_attention_torch_math_bwd, kv_len={kv_len}", flops_bwd):
            for _ in range(test_iter):
                attn_output_torch2.backward(attn_output_torch2.detach(), retain_graph=True)
        attn_output_torch2.backward(attn_output_torch2.detach())

        for _ in range(warmup_iter):
            attn_output_torch3 = compiled_flex_attention(q, k, v)
        with time_with_cuda_event("flex_attention_fwd", flops):
            for _ in range(test_iter):
                attn_output_torch3 = compiled_flex_attention(q, k, v)

        for _ in range(warmup_iter):
            attn_output_torch3.backward(attn_output_torch3.detach(), retain_graph=True)
        with time_with_cuda_event("flex_attention_bwd", flops_bwd):
            for _ in range(test_iter):
                attn_output_torch3.backward(attn_output_torch3.detach(), retain_graph=True)
        attn_output_torch3.backward(attn_output_torch3.detach())

        q, k, v = get_qkv(batch_size, num_heads, q_len, kv_len, head_dim, mqa, layout="sbhd")
        te_fused_attn = DotProductAttention(num_attention_heads=num_heads, kv_channels=head_dim, qkv_format="bshd", attn_mask_type="no_mask", num_gqa_groups=1 if mqa else None)

        for _ in range(warmup_iter):
            attn_output_te = te_fused_attn(q, k, v)
        with time_with_cuda_event(f"cudnn_attention_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_te = te_fused_attn(q, k, v)

        for _ in range(warmup_iter):
            attn_output_te.backward(attn_output_te.detach(), retain_graph=True)
        with time_with_cuda_event(f"cudnn_attention_bwd, kv_len={kv_len}", flops_bwd):
            for _ in range(test_iter):
                attn_output_te.backward(attn_output_te.detach(), retain_graph=True)
        attn_output_te.backward(attn_output_te.detach())

        for _ in range(warmup_iter):
            attn_output_fa = flash_attn_func(q, k, v)
        with time_with_cuda_event(f"flash_attn_func_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_fa = flash_attn_func(q, k, v)

        for _ in range(warmup_iter):
            attn_output_fa.backward(attn_output_fa.detach(), retain_graph=True)
        with time_with_cuda_event(f"flash_attn_func_bwd, kv_len={kv_len}", flops_bwd):
            for _ in range(test_iter):
                attn_output_fa.backward(attn_output_fa.detach(), retain_graph=True)
        attn_output_fa.backward(attn_output_fa.detach())

        for _ in range(warmup_iter):
            attn_output_fa_hopper, softmax_lse = flash_attn_func_hopper(q, k, v)    
        with time_with_cuda_event(f"flash_attn_func_hopper_fwd, kv_len={kv_len}", flops):
            for _ in range(test_iter):
                attn_output_fa_hopper, softmax_lse = flash_attn_func_hopper(q, k, v)

        for _ in range(warmup_iter):
            attn_output_fa_hopper.backward(attn_output_fa_hopper.detach(), retain_graph=True)
        with time_with_cuda_event(f"flash_attn_func_hopper_bwd, kv_len={kv_len}", flops_bwd):
            for _ in range(test_iter):
                attn_output_fa_hopper.backward(attn_output_fa_hopper.detach(), retain_graph=True)
        attn_output_fa_hopper.backward(attn_output_fa_hopper.detach())
        torch.cuda.profiler.stop()
