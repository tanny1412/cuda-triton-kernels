# Session 03 Devlog — First CUDA Kernel

## New syntax vs C++
- __global__  → marks a function as a GPU kernel, called from CPU, runs on GPU
- <<<grid, block>>> → kernel launch syntax, defines how many threads to spawn
- cudaMalloc  → allocate memory on GPU heap (same idea as new, but GPU)
- cudaMemcpy  → copy data between CPU and GPU
- cudaFree    → free GPU memory (same idea as delete[])

## Thread indexing — the most important line in CUDA
int i = blockIdx.x * blockDim.x + threadIdx.x

- blockIdx.x  → which block am I in?
- blockDim.x  → how many threads per block? (256 in our case)
- threadIdx.x → which thread am I within my block? (0-255)
- Result: each thread gets a unique i → its own element to work on
- Block 0 → elements 0-255, Block 1 → elements 256-511, etc.
- This formula is the same in every kernel — memorize it

## Flow of a CUDA program
1. Allocate CPU memory (new)
2. Allocate GPU memory (cudaMalloc)
3. Copy data CPU → GPU (cudaMemcpy HostToDevice)
4. Launch kernel <<<grid, blocks>>>
5. Copy results GPU → CPU (cudaMemcpy DeviceToHost)
6. Verify results
7. Free GPU memory (cudaFree)
8. Free CPU memory (delete[])

## Grid size formula
- grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE
- Ensures enough blocks to cover all N elements even if N is not divisible by BLOCK_SIZE
- if (i < N) guard in kernel — threads beyond N do nothing

## Full CUDA boilerplate pattern (every program follows this)
1. Calculate N (number of elements) and bytes (N * sizeof(float))
2. Allocate CPU heap with pointers (h_A, h_B, h_C) using new
3. Initialize CPU data (fill with values or load from somewhere)
4. Declare GPU pointers (d_A, d_B, d_C) — just pointers, no data yet
5. Allocate GPU VRAM with cudaMalloc(&d_A, bytes) — d_A now holds starting address on VRAM
6. Copy data CPU → GPU with cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice)
7. Calculate grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE
8. Launch kernel: kernel<<<grid, BLOCK_SIZE>>>(d_A, d_B, d_C, N)
9. Copy results GPU → CPU with cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost)
10. Read results from h_C using indexing (pointer + index = element)
11. Free GPU memory with cudaFree, CPU memory with delete[]

## h_ and d_ naming convention
- h_ prefix → host (CPU) memory
- d_ prefix → device (GPU) memory
- Standard convention in all CUDA code

## Result
- Ran on RTX 4090 on RunPod
- C[0] = 3.0, C[1023] = 3.0 — 1024 parallel additions, all correct
- First CUDA kernel working
