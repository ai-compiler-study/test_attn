import logging
import os
from typing import Callable, List, Tuple

import typer
import torch
import torch.nn as nn
import torch._inductor.config as inductor_config

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

torch.set_default_device("cuda")
app = typer.Typer()


def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    return start_event.elapsed_time(end_event) * 1000 / iters, *f(*args, **kwargs)


class IterationProfiler:
    def __init__(self, steps=None):
        self.start = None
        self.end = None
        self.num_iterations = 0
        self.steps = steps

    def get_iter_per_sec(self):
        if self.start is None or self.end is None:
            return None
        self.end.synchronize()
        et = self.start.elapsed_time(self.end)
        return self.num_iterations / et * 1000.0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs):
        if self.start is None:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.start = start
        else:
            if self.steps is None or i == self.steps - 1:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.end = event
            self.num_iterations += 1
        return callback_kwargs


def _compile_transformer_backbone(
    transformer: nn.Module,
    enable_torch_compile: bool,
    enable_nexfort: bool,
    fullgraph: bool = True,
    debug: bool = False,
):
    if debug:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_LOGS"] = "+inductor,dynamo"
        inductor_config.debug = True
        # Whether to disable a progress bar for autotuning
        inductor_config.disable_progress = False
        # Whether to enable printing the source code for each future
        inductor_config.verbose_progress = True
        inductor_config.trace.enabled = True
        inductor_config.trace.debug_log = True
        inductor_config.trace.info_log = True
        # inductor_config.trace.graph_diagram = True  # INDUCTOR_POST_FUSION_SVG=1
        # inductor_config.trace.draw_orig_fx_graph = True  # INDUCTOR_ORIG_FX_SVG=1

    # torch._inductor.list_options()
    inductor_config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"

    # TORCHINDUCTOR_BENCHMARK_KERNEL: inductor_config.benchmark_kernel
    # inductor_config.triton.cudagraphs = False
    # Tune the generated Triton kernels at compile time instead of first time they run
    # Setting to None means uninitialized
    # inductor_config.triton.autotune_at_compile_time = True

    inductor_config.cuda.compile_opt_level = "-O3"  # default: "-O1"
    inductor_config.cuda.use_fast_math = True

    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
    inductor_config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    if enable_torch_compile and enable_nexfort:
        logging.warning(
            f"apply --use_torch_compile {enable_torch_compile} and --use_nexfort {enable_nexfort} together. we use torch compile only"
        )

    if enable_torch_compile:
        if getattr(transformer, "forward") is not None:
            if enable_torch_compile:
                optimized_transformer_forward = torch.compile(
                    getattr(transformer, "forward"),
                    fullgraph=fullgraph,
                    backend="inductor",
                    mode="max-autotune-no-cudagraphs",
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
    height: int = 1024,
    width: int = 1024,
    diffusion_steps: int = 28,
    max_sequence_length: int = 512,
    use_torch_compile: bool = True,
    fullgraph: bool = True,
    use_nexfort: bool = False,
    debug: bool = True,
    iters: int = 10,
    seed: int = 42,
):
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")

    transformer = getattr(pipeline, "transformer", None)
    vae = getattr(pipeline, "vae", None)

    transformer.to(memory_format=torch.channels_last)
    vae.to(memory_format=torch.channels_last)

    if debug:
        transformer_params = (
            sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
            / 1e9
        )
        print(f"Total transformer number of parameters: {transformer_params:.2f}B")

    if transformer is not None and use_torch_compile:
        pipeline.transformer = _compile_transformer_backbone(
            transformer,
            enable_torch_compile=use_torch_compile,
            enable_nexfort=use_nexfort,
            fullgraph=fullgraph,
            debug=debug,
        )
    if vae is not None and use_torch_compile:
        pipeline.vae = torch.compile(
            vae, mode="max-autotune-no-cudagraphs", fullgraph=True
        )

    forward_time, _ = benchmark_torch_function(
        iters,
        pipeline,
        height=height,
        width=width,
        prompt="A tree in the forest",
        num_inference_steps=diffusion_steps,
        max_sequence_length=max_sequence_length,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    print(f"avg fwd time: {forward_time / 1e6} s")


if __name__ == "__main__":
    typer.run(main)
