# Session 04 Devlog — Shared Memory + Tiled Matmul

## Why tiling
- Naive matmul reads row r of A N times from global memory (300 cycles each)
- Tiling: load a 16x16 chunk into shared memory once (5 cycles), reuse it
- 60x faster memory access for the same data

## New concepts
- __shared__ float tile_A[TILE][TILE] → allocates 16x16 scratchpad in shared memory per block
- Every block gets its own private copy of tile_A and tile_B
- __syncthreads() → barrier, all threads must reach this point before anyone continues

## Thread indexing in 2D
- row = blockIdx.y * TILE + threadIdx.y  → which row of C this thread handles
- col = blockIdx.x * TILE + threadIdx.x  → which col of C this thread handles
- x = columns, y = rows (CUDA convention — opposite of what you'd expect)
- Each thread is responsible for exactly ONE element of C

## The tiling loop
- Loop runs N/TILE times (32 times for N=512, TILE=16)
- Each iteration (tile t):
  1. Each thread loads one element from global memory into its tile slot
  2. __syncthreads() — wait for all 256 threads to finish loading
  3. Compute partial dot product from fast shared memory (k loop, 0 to TILE-1)
  4. __syncthreads() — wait before overwriting tile with next iteration
- After all tiles: sum = complete dot product → write to C[row][col]

## Two __syncthreads() per iteration
- First: after loading, before computing — don't read tile before it's fully loaded
- Second: after computing, before loading next tile — don't overwrite tile while others still read

## Indexing math (concept, not memorization)
- Always row * N + col to convert 2D to 1D
- Tile offset t * TILE shifts which chunk of A/B you're loading this iteration
- Don't memorize exact formulas — know what they're doing: finding the right element

## The cleanest mental model
Each thread computes one C[row][col]. The full row of A and full column of B are too
large to load at once, so the outer loop walks through them 16 elements at a time.
Every tile contributes a partial dot-product, which is added to the same running sum
until the entire row-column dot product is complete.

## Indexing pattern — always row * N + col
- For A: row is fixed (my row), col varies by tile → t * TILE + threadIdx.x
- For B: col is fixed (my col), row varies by tile → t * TILE + threadIdx.y
- t * TILE = where the tile starts, + threadIdx = my offset within the tile
- Everything is just row * N + col with different expressions for row and col

## dim3
- dim3 block(TILE, TILE) → 2D block of 16x16 = 256 threads
- dim3 grid(N/TILE, N/TILE) → enough blocks to cover NxN output matrix
- Use dim3 whenever your problem is 2D
