# Session 09 Devlog — GEMM in Triton

## Matrix shapes
- A is [M, K], B is [K, N], C is [M, N]
- M = rows of A and C
- N = cols of B and C
- K = shared inner dimension (cols of A = rows of B)

## Tiling C
- C is split into [BLOCK_M, BLOCK_N] tiles
- Each program instance owns exactly one tile of C
- Number of tiles = (M / BLOCK_M) × (N / BLOCK_N)
- For 512×512 with BLOCK_M=64, BLOCK_N=64: (512/64) × (512/64) = 8 × 8 = 64 tiles
- 64 tiles = 64 program instances running in parallel

## BLOCK_K and the K loop
- K is the accumulation dimension — you can't compute a C tile in one shot
- BLOCK_K is the chunk size for the K dimension per iteration
- iterations per tile = K / BLOCK_K = 512 / 32 = 16
- Bigger BLOCK_K → fewer iterations, more work per iteration
- Smaller BLOCK_K → more iterations, less work per iteration

## What each iteration loads
- Tile of A: shape [BLOCK_M, BLOCK_K] = [64, 32]
- Tile of B: shape [BLOCK_K, BLOCK_N] = [32, 64]
- Multiply them → [64, 64], add to acc
- After 16 iterations, acc = complete 64×64 tile of C

## Key insight: what stays fixed vs what moves
- For a fixed C tile, A keeps the same rows and walks across columns (K dimension)
- For a fixed C tile, B keeps the same columns and walks across rows (K dimension)
- Different program instances use different rows of A and different cols of B

## Why different tiles get different values
- pid_m and pid_n are different for each program instance
- offs_m = pid_m * BLOCK_M + arange(0, BLOCK_M) → different rows
- offs_n = pid_n * BLOCK_N + arange(0, BLOCK_N) → different cols
- Tile (0,0): rows 0-63 of A, cols 0-63 of B
- Tile (0,1): rows 0-63 of A, cols 64-127 of B
- Tile (1,0): rows 64-127 of A, cols 0-63 of B

## Strides
- 2D matrix stored as flat 1D array in memory
- To get A[row, col]: index = row * stride_am + col * stride_ak
- stride_am = 512 (skip one full row = 512 elements to move one row down)
- stride_ak = 1 (skip 1 element to move one col right)
- PyTorch computes strides automatically from shape: A.stride(0), A.stride(1)
- For any contiguous [M, K] matrix: stride(0) = K, stride(1) = 1
- A, B, C all [512, 512] → all have stride(0)=512, stride(1)=1

## 2D pointer arithmetic
- A_ptr is the base pointer — memory address of A[0][0]
- To get A[row, col]: A_ptr + row * stride_am + col * stride_ak
- offs_m[:, None] → shape [64, 1] (column vector of row indices)
- offs_k[None, :] → shape [1, 32] (row vector of col indices)
- Broadcasting: [64, 1] + [1, 32] = [64, 32] grid of pointers
- a_ptrs[i, j] = A_ptr + offs_m[i] * 512 + offs_k[j] * 1 = address of A[offs_m[i], offs_k[j]]
- tl.load(a_ptrs) fetches all 64×32 values at once

## acc
- One acc per program instance, shape [BLOCK_M, BLOCK_N] = [64, 64]
- Starts as zeros, accumulates partial dot products across 16 K iterations
- Lives in registers — never written to HBM until the final tl.store

## Grid
- grid = lambda meta: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
- Lambda needed because autotune picks BLOCK_M/BLOCK_N at runtime
- For 512×512 with BLOCK_M=64: grid = (8, 8) = 64 program instances

## Full kernel flow
Own C tile
↓
Fix offs_m, offs_n (actual row/col indices for this tile)
↓
Walk K using offs_k (shifts by BLOCK_K each iteration)
↓
Load A tile [BLOCK_M, BLOCK_K] using 2D pointer arithmetic
↓
Load B tile [BLOCK_K, BLOCK_N] using 2D pointer arithmetic
↓
acc += tl.dot(a, b)
↓
Repeat until K exhausted (16 iterations)
↓
Store acc into C at fixed rows/cols
