# Session 01 Devlog — Triton Mental Model + First Kernel

## CUDA vs Triton mental model
- CUDA: thread-level — one thread handles one element, you manage everything manually
- Triton: block-level — one program handles a whole block of elements as a vector
- Triton compiler handles thread scheduling, shared memory, coalescing automatically
- No threadIdx, no __syncthreads, no cudaMalloc, no cudaMemcpy

## Key Triton concepts
- @triton.jit       → marks function as GPU kernel (like __global__ in CUDA)
- tl.program_id(0)  → which block am I? (like blockIdx.x in CUDA)
- tl.arange(0, N)   → generates [0, 1, 2, ..., N-1] — a vector of indices
- tl.load()         → load a vector of elements from global memory
- tl.store()        → write a vector of elements to global memory
- tl.constexpr      → compile-time constant — required for BLOCK_SIZE so compiler can optimize

## Offsets pattern — always the same
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
- pid * BLOCK_SIZE = where this block starts
- tl.arange = [0..BLOCK_SIZE-1] = my indices within the block
- Result: a vector of all global indices this block handles

## Mask pattern — always the same
mask = offsets < N
- Boolean vector, True for valid elements, False for out of bounds
- Passed to tl.load and tl.store to handle partial last block safely

## PyTorch integration
- Tensors live on GPU via device='cuda' — no cudaMalloc needed
- .numel() → total number of elements
- .item() → extract single scalar from GPU tensor to Python (implicit CPU transfer)
- torch.empty_like(A) → allocate output tensor same shape/dtype/device as A
- torch.full((N,), val, device='cuda') → fill tensor with a value on GPU

## Launch syntax
kernel[grid](args...)           # Triton
kernel<<<grid, block>>>(args)   # CUDA equivalent
- grid = (triton.cdiv(N, BLOCK_SIZE),)  → same ceiling division formula as CUDA
- BLOCK_SIZE passed as regular argument, not in launch syntax

## Comparison to CUDA vector_add
CUDA:  ~50 lines, manual cudaMalloc/cudaMemcpy/cudaFree, thread indexing
Triton: ~20 lines, PyTorch handles memory, block-level vector operations
Same logic, dramatically less code
