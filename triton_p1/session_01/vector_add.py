import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    A_ptr, B_ptr, C_ptr,  # pointers to GPU memory (same as CUDA d_A, d_B, d_C)
    N,                     # total number of elements
    BLOCK_SIZE: tl.constexpr,  # how many elements this block handles
):
    # which block am I? (same as blockIdx.x in CUDA)
    pid = tl.program_id(axis=0)

    # the range of elements this block is responsible for
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask: don't go out of bounds (same as if (i < N) in CUDA)
    mask = offsets < N

    # load a block of elements from global memory
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    # compute — operates on the whole block at once
    c = a + b

    # store result back to global memory
    tl.store(C_ptr + offsets, c, mask=mask)


def vector_add(A, B):
    C = torch.empty_like(A)
    N = A.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE),)  # number of blocks — same formula as CUDA

    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE)
    return C


# --- test ---
A = torch.ones(4096, device='cuda')
B = torch.full((4096,), 2.0, device='cuda')
C = vector_add(A, B)

print(f"C[0] = {C[0].item()}  (expected 3.0)")
print(f"C[4095] = {C[4095].item()}  (expected 3.0)")
print(f"max error vs torch: {(C - (A + B)).abs().max().item()}")
