import math
import platform
import os
import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import torch
from torch import nn
import torch._inductor.config as inductor_config

import typer
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from torch.utils.flop_counter import FlopCounterMode
# from torchao.profiler.performance_counter import PerformanceCounterMode


torch.set_default_device("cuda")
app = typer.Typer()

TORCH_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


@dataclass
class DeviceLimit:
    name: str = "default"  # pattern to match from `torch.cuda.get_device_name()`
    source: str = ""
    sm: Tuple[int, int] = (0, 0)
    # bytes/s
    gmem_bandwidth: float = math.inf
    # dtype -> TFlop/s
    gemm_tflops: Mapping[torch.dtype, float] = field(default_factory=dict)


# For f32, we assume we can use tf32
DEVICE_LIMITS: Tuple[DeviceLimit, ...] = (
    DeviceLimit(
        "H100",
        "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet",  # noqa: E501
        sm=(9, 0),
        gmem_bandwidth=3.35 * (1024**4),  # NOTE: PCIe is 2 TB/s
        gemm_tflops={
            torch.float64: 67,
            # NOTE: NVIDIA gives all numbers "with 2:4 sparsity"
            # but we want the full GEMM numbers
            torch.float32: 989 // 2,
            torch.float16: 1979 // 2,
            torch.bfloat16: 1979 // 2,
            torch.int8: 3958 // 2,
        },
    ),
    DeviceLimit(
        "A100",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",  # noqa: E501
        sm=(8, 0),
        gmem_bandwidth=2 * (1024**4),  # NOTE: PCIe is 1.5 TB/s
        gemm_tflops={
            torch.float64: 19.5,
            torch.float32: 156,
            torch.float16: 312,
            torch.bfloat16: 312,
            torch.int8: 624,
        },
    ),
)


@dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float
    dtype: str
    device: str
    arch: str  # GPU architecture
    is_model: bool = False
    is_compiled: bool = False


class CudaTimer:
    """
    A static context manager class for measuring execution time of PyTorch code
    using CUDA events. It synchronizes GPU operations to ensure accurate time measurements.
    """

    def __init__(self, name="", precision=5, display=False):
        self.name = name
        self.precision = precision
        self.display = display

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *exc):
        self.end_event.record()
        torch.cuda.synchronize()
        # Convert from ms to s
        self.elapsed_time = self.start_event.elapsed_time(self.end_event) * 1e-3

        if self.display:
            print(f"{self.name}: {self.elapsed_time:.{self.precision}f} s")

    def get_elapsed_time(self):
        """Returns the elapsed time in microseconds."""
        return self.elapsed_time


def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


def print_experiment_summary(experiment: Experiment) -> str:
    return (
        f"Experiment Summary:\n"
        f"  Name: {experiment.name}\n"
        f"  Metric: {experiment.metric}\n"
        f"  Target: {experiment.target:.2f}\n"
        f"  Actual: {experiment.actual:.2f}\n"
        f"  Performance: {(experiment.actual / experiment.target) * 100:.2f}% of target\n"
        f"  Data Type: {experiment.dtype}\n"
        f"  Device: {experiment.device}\n"
        f"  Architecture: {experiment.arch}\n"
        f"  Is Model: {'Yes' if experiment.is_model else 'No'}"
    )


def _compile_transformer_backbone(
    transformer: nn.Module,
    enable_torch_compile: bool,
    fullgraph: bool = True,
    verbose: bool = True,
):
    if verbose:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_LOGS"] = "inductor,dynamo"
    # torch._inductor.list_options()
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    inductor_config.benchmark_kernel = True
    inductor_config.cuda.compile_opt_level = "-O3"  # default: "-O1"
    inductor_config.cuda.use_fast_math = True

    if enable_torch_compile:
        if getattr(transformer, "forward") is not None:
            if enable_torch_compile:
                optimized_transformer_forward = torch.compile(
                    getattr(transformer, "forward"),
                    fullgraph=fullgraph,
                    backend="inductor",
                    mode="max-autotune",
                )
            setattr(transformer, "forward", optimized_transformer_forward)
        else:
            raise AttributeError(
                f"Transformer backbone type: {transformer.__class__.__name__} has no attribute 'forward'"
            )
    return transformer


@app.command()
def main(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    dtype: str = "fp16",
    diffusion_steps: int = 30,
    max_sequence_length: int = 512,
    compile: bool = True,
    warmup: int = 3,
    fullgraph: bool = True,
    verbose: bool = False,
):
    dtype = TORCH_DTYPES[dtype]
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipeline.to("cuda")
    prompt = "A dart tree in the forest"

    flop_counter = FlopCounterMode(display=verbose)
    with flop_counter:
        _ = pipeline(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=diffusion_steps,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0),
            )
    device_flops_per_s = DEVICE_LIMITS[0].gemm_tflops[dtype]
    expected_tflops = flop_counter.get_total_flops() * 1e-12

    transformer = getattr(pipeline, "transformer", None)
    vae = getattr(pipeline, "vae", None)
    scheduler = getattr(pipeline, "scheduler", None)

    if verbose:
        transformer_params = (
            sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
            / 1e9
        )
        print(f"Total transformer number of parameters: {transformer_params:.2f}B")

    if transformer is not None and compile:
        pipeline.transformer = _compile_transformer_backbone(
            transformer,
            enable_torch_compile=compile,
            fullgraph=fullgraph,
            verbose=verbose,
        )
    # warmup
    logging.info("Warmup")
    for _ in range(warmup):
        _ = pipeline(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=diffusion_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(0),
        )

    with CudaTimer(display=False) as timer:
        _ = pipeline(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=diffusion_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(0),
        )
    tflops_per_s = expected_tflops / timer.get_elapsed_time()
    flops_utilization = tflops_per_s / device_flops_per_s
    print(f"{tflops_per_s:.5f} TFLOP/s")
    print(f"flops utilization {flops_utilization:.2f}")

    experiment = Experiment(
        f"flux_fwd_{dtype}_seqlen:{max_sequence_length}_steps:{diffusion_steps}",
        "tflop/s",
        device_flops_per_s,
        tflops_per_s,
        dtype,
        "cuda",
        get_arch_name(),
        is_model=True,
        is_compiled=False,
    )
    print(experiment)
    if verbose:
        print(print_experiment_summary(experiment))


if __name__ == "__main__":
    typer.run(main)
