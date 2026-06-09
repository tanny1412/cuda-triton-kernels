# Session 04 Devlog — Fused Elementwise Ops

## Why fusion matters
- Each kernel launch = one read + one write to HBM
- Unfused ReLU + scale + bias = 3 kernel launches = 3 reads + 3 writes HBM
- Fused = 1 read + 1 write HBM — same math, 3x less memory traffic
- All ops happen in registers between load and store — HBM never touched in between

## ReLU in Triton
- tl.maximum(x, 0.0) — element-wise max between vector and 0
- Negatives → 0, positives → unchanged
- x is a whole block vector, operation applied to all elements simultaneously

## Realistic fusion pattern
- In transformer MLPs: linear(x) → relu → next layer
- linear does scale + bias, relu is the activation — always appear together
- Fusing them eliminates intermediate HBM reads/writes

## Benchmarking pattern
1. Warmup (10 runs) — JIT compiles kernel, warms GPU caches
2. Time 100 reps with wall clock (time.perf_counter)
3. torch.cuda.synchronize() after all reps — wait for GPU to finish before stopping timer
4. Divide by REPS, multiply by 1000 → ms per run

## Wall clock time
- Measures real elapsed time including everything: Python, kernel launch, GPU, overhead
- vs cudaEvent which measures pure GPU time only
- For relative benchmarks (fused vs unfused) wall clock is fine — overhead cancels out
- For precise GPU-only timing use triton.testing.do_bench (later sessions)

## torch.cuda.synchronize()
- Kernel launches are async — CPU fires and moves on immediately
- Without synchronize, timer stops before GPU finishes → measures near-zero time
- synchronize() forces CPU to wait for GPU — same as cudaEventSynchronize in CUDA

## tl.* functions reference (90% of what you'll ever use)
# Memory
tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(ptr + offsets, val, mask=mask)

# Indexing
tl.program_id(axis=0)        # which block am I
tl.arange(0, BLOCK_SIZE)     # vector of indices 0 to N-1

# Reductions
tl.sum(x, axis=0)
tl.max(x, axis=0)
tl.min(x, axis=0)

# Math
tl.maximum(x, 0.0)           # elementwise max — ReLU
tl.exp(x)                    # elementwise exp — needed for softmax
tl.log(x)
tl.sqrt(x)
tl.dot(a, b)                 # matrix multiply — GEMM

# Utilities
tl.constexpr                 # compile-time constant
triton.cdiv(n, d)            # ceiling division
triton.next_power_of_2(n)    # round up to nearest power of 2

# Inside kernel: use tl.* for special ops, regular Python +*-/ for arithmetic
# Outside kernel (host side): use PyTorch as normal

## Warmup
- First Triton run: JIT compiles the kernel — artificially slow
- GPU caches cold on first run
- Always warmup before benchmarking — 10 runs is standard
