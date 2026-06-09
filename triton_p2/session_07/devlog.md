# Session 07 Devlog — Autotuning

## What autotuning does
- Kernel performance depends on BLOCK_SIZE, num_warps, num_stages
- Optimal values depend on GPU, input size, operation — can't know in advance
- @triton.autotune: give a list of configs, Triton benchmarks all, picks fastest
- Result cached — subsequent runs use best config without re-searching

## key parameter
- key=['N_cols'] → re-tune when N_cols changes
- Different input sizes need different optimal configs
- Separate cache entry per unique key value

## triton.Config parameters
- BLOCK_SIZE          → elements per block, injected as tl.constexpr into kernel
- BLOCK_SIZE_M/N/K   → for 2D/3D kernels like GEMM
- num_warps          → warps per block, hardware-level compiler directive
- num_stages         → software pipelining depth, prefetch next tile while computing

## num_warps vs BLOCK_SIZE
- BLOCK_SIZE: declared in kernel as tl.constexpr, autotune injects value
- num_warps: never declared in kernel, compiler directive passed to PTX compiler
- Two different mechanisms, both recognized by Triton automatically

## Best practice for configs
- Decouple BLOCK_SIZE and num_warps — give each their own combinations
- Otherwise autotune can't find optimal combination independently
  triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
  triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),

## Without autotuning
- Must manually calculate BLOCK_SIZE = triton.next_power_of_2(N_cols)
- With autotuning: configs already define power-of-2 values, injected automatically

## Decorator order
@triton.autotune  ← outer decorator
@triton.jit       ← inner decorator
def kernel():
