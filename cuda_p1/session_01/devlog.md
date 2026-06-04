# Session 01 Devlog — GPU Architecture

## GPU vs CPU
- CPU: 8-16 powerful cores, flexible, handles complex logic and branching
- GPU: 16,896 CUDA cores on H100, simple cores, best when all doing the same op
- GPU philosophy: not faster per core, but massively parallel
- Matrix multiply is perfect: every C[r][c] is independent, compute all simultaneously

## Why GPU beats CPU for matmul
- CPU: computes each output element one by one → 501ms for 512x512
- GPU: computes thousands of output elements simultaneously → ~1ms
- Not because each operation is faster, but because 16,000 run at the same time

## Thread hierarchy
- Grid → Blocks → Threads → Warps (32 threads, automatic)
- You define grid and blocks, hardware groups threads into warps
- All 32 threads in a warp execute the same instruction simultaneously
- For matmul: launch one thread per output element C[r][c]
  instead of a loop, thousands of threads each compute their own element in parallel

## Streaming Multiprocessors (SMs)
- H100 has 132 SMs — each is a mini-processor
- Each SM has: CUDA cores, shared memory (~48KB), registers
- CUDA distributes blocks across SMs automatically — you don't control this
- One or more blocks run per SM

## Memory hierarchy (fastest to slowest)
- Registers     → per thread,  ~1 cycle,    fastest
- Shared Memory → per block,   ~5 cycles,   48KB, only visible within a block
- L1/L2 Cache  → automatic,   ~30 cycles
- Global Memory → all threads, ~300 cycles, 80GB HBM3 on H100, slowest

## SIMT — Single Instruction Multiple Threads
- All 32 threads in a warp execute the same instruction simultaneously, on different data
- CPU: one worker doing tasks one by one
- GPU: 32 workers doing the exact same task simultaneously, each on different data

## Warp divergence
- When threads in a warp take different paths (if/else), performance tanks
- Warp can't split — runs branch A with other threads idle, then branch B with others idle
- Half throughput if 16 threads go one way and 16 go another
- Rule: keep all threads in a warp doing the same thing

## Key insight
Most CUDA optimization = keep data in registers/shared memory, avoid global memory.
Every time you read from global memory (HBM) you pay 300 cycles.
Every time you read from shared memory you pay 5 cycles.
That 60x difference is where all the optimization happens.

## Roofline model
- Every kernel is either memory-bound or compute-bound
- Arithmetic intensity = FLOPS / bytes_read
- Low intensity → memory-bound (waiting for data, fix: reduce reads, fuse ops)
- High intensity → compute-bound (math is bottleneck, fix: tensor cores, FP8)
- Matmul N=512: 128 FLOPS/byte → compute-bound
  - reuses data heavily: 2N math ops per element, only reads A and B once
- ReLU: ~0.25 FLOPS/byte → memory-bound
  - read one float, one op, write back — barely any math per byte
- This number tells you HOW to optimize a kernel — different problem, different fix
