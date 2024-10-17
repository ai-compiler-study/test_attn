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

    if enable_torch_compile and enable_nexfort:
        logging.warning(
            f"apply --use_torch_compile and --use_nexfort together. we use torch compile only"
        )

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
    diffusion_steps: int = 30,
    max_sequence_length: int = 512,
    use_torch_compile: bool = True,
    fullgraph: bool = True,
    use_nexfort: bool = False,
    verbose: bool = True,
    ):  
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")

    transformer = getattr(pipeline, "transformer", None)
    vae = getattr(pipeline, "vae", None)
    scheduler = getattr(pipeline, "scheduler", None)

    if verbose:
        transformer_params = sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad) / 1e9
        print(f"Total transformer number of parameters: {transformer_params:.2f}B")

    if transformer is not None and use_torch_compile:
        pipeline.transformer = _compile_transformer_backbone(
            transformer,
            enable_torch_compile=use_torch_compile,
            enable_nexfort=use_nexfort,
            fullgraph=fullgraph,
            verbose=verbose,
        )
            
    forward_time, _ = benchmark_torch_function(
    10,
    pipeline,
    "A tree in the forest",
    guidance_scale=0.0,
    num_inference_steps=diffusion_steps,
    max_sequence_length=max_sequence_length,
    )
    print(f"avg fwd time: {forward_time / 1e6} s")


if __name__ == "__main__":
    typer.run(main)