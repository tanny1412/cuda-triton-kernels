# Session 07 Devlog — Profiling with Nsight

## Two tools
- Nsight Systems  → full timeline, CPU + GPU, kernel launches, memory transfers
- Nsight Compute  → per-kernel deep dive, the one that matters for optimization

## Key metrics to check every kernel
- Memory bandwidth utilization → how close to HBM peak (3.35 TB/s on H100, ~900 GB/s on 4090)
- Compute throughput           → how close to peak FLOPS
- L1/L2 hit rate               → are you reusing data or constantly hitting global memory
- Roofline position            → memory-bound or compute-bound

## Commands
ncu ./my_kernel                        # quick summary in terminal
ncu --set full -o profile ./my_kernel  # full report saved to file

## The workflow for every kernel going forward
1. Write kernel
2. Verify correctness
3. Run ncu — find the bottleneck
4. Optimize
5. Run ncu again — did it improve?

## What the roofline tells you
- If memory bandwidth is the bottleneck → reduce global memory reads (tiling, fusion)
- If compute is the bottleneck → use tensor cores, FP8, better math
- You can't fix what you can't measure — this is why profiling matters

## TODO
- Run ncu on vector_add and tiled_matmul on RunPod
- Note bandwidth utilization for each
- Compare roofline position of naive vs tiled matmul
