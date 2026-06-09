import torch
import triton
import triton.language as tl

@triton.jit
def scale_kernel(
    X_ptr, OUT_ptr,
    N,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # other=-1.0 — masked elements load as -1 so we can see them clearly
    x = tl.load(X_ptr + offsets, mask=mask, other=-1.0)

    out = x * scale  # scale every element

    tl.store(OUT_ptr + offsets, out, mask=mask)  # only store valid elements


def scale(X, scale_val):
    OUT = torch.empty_like(X)
    N = X.numel()
    BLOCK_SIZE = 64  # small block so we get a partial last block with N=100

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    scale_kernel[grid](X, OUT, N, scale_val, BLOCK_SIZE)
    return OUT


# N=100, BLOCK_SIZE=64 → 2 blocks
# block 0: elements 0-63  (full)
# block 1: elements 64-99 (partial — only 36 valid, 28 masked)
X = torch.ones(100, device='cuda')
OUT = scale(X, 3.0)

print(f"N=100, BLOCK_SIZE=64")
print(f"OUT[0]  = {OUT[0].item()}   (expected 3.0 — valid)")
print(f"OUT[99] = {OUT[99].item()}  (expected 3.0 — last valid element)")
print(f"max error vs torch: {(OUT - X * 3.0).abs().max().item()}")
