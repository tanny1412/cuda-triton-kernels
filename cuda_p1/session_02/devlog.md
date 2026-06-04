# Session 02 Devlog — Memory Hierarchy

## The four levels (fastest to slowest)
- Registers     → per thread,  ~1 cycle,    fastest
- Shared Memory → per block,   ~5 cycles,   48KB, visible only within a block
- L1/L2 Cache  → automatic,   ~30 cycles
- Global Memory → all threads, ~300 cycles, 80GB HBM3

## Why naive matmul is slow
- Computing C[r][c] requires reading row r of A once per column of B
- Row r of A gets read N times from global memory → N × 300 cycle penalties
- You're paying for the same data over and over

## The fix: shared memory tiling
- Load a chunk of A and B into shared memory once (5 cycles)
- Reuse it N times from shared memory instead of global memory
- 300 cycles → 5 cycles = 60x improvement on memory access
- This is the core idea behind tiled matmul, FlashAttention, and every
  serious production CUDA kernel

## Key numbers (512x512 float32)
- Naive matmul reads same data 512x from global memory → 0.16ms memory read time
- Tiled matmul reads A once + B once → 0.0006ms memory read time
- 256x reduction in global memory traffic
- Actual speedup won't be exactly 256x (other bottlenecks) but this is the principle

## Why FlashAttention is "memory efficient"
- Same idea — tiles Q, K, V so they're read from global memory as few times as possible
- Every "memory efficient" kernel you'll see is just tiling applied to a different operation
