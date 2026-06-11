import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # how many elements to skip to move 1 row or 1 col in A
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # which tile of C am I computing?
    pid_m = tl.program_id(axis=0)  # row tile index
    pid_n = tl.program_id(axis=1)  # col tile index

    # row and col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator for this tile of C — starts at zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K tiles
    for k in range(0, K, BLOCK_K): # 16 iterations if K=512 and BLOCK_K=32 (512/32 = 16) for one tile of C
        offs_k = k + tl.arange(0, BLOCK_K)

        # load tile of A: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak 
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # load tile of B: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # multiply and accumulate
        acc += tl.dot(a, b)

    # write result tile to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(A, B):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), # stride_am, stride_ak will be 512, 1 for row-major
        B.stride(0), B.stride(1), # stride_bk, stride_bn will be 512, 1 for row-major
        C.stride(0), C.stride(1), # stride_cm, stride_cn will be 512, 1 for row-major
    )
    return C


# test
A = torch.randn(512, 512, device='cuda') # m = 512, k = 512
B = torch.randn(512, 512, device='cuda') # k = 512, n = 512

C = matmul(A, B)
expected = torch.matmul(A, B)

print(f"max error vs torch: {(C - expected).abs().max().item():.4f}")
print(f"C[0][0] = {C[0][0].item():.4f}  torch = {expected[0][0].item():.4f}")
