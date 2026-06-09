import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    X_ptr, OUT_ptr,
    N_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # each program handles one row
    row = tl.program_id(axis=0)
    offsets = row * N_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N_cols

    # --- Pass 1: load row, find max, compute exp, sum ---
    x = tl.load(X_ptr + offsets, mask=mask, other=-float('inf'))  # masked = -inf so they don't affect max

    # numerically stable: subtract max before exp
    row_max = tl.max(x, axis=0)
    x = x - row_max         # largest value is now 0, exp(0)=1, no overflow
    x = tl.exp(x)

    row_sum = tl.sum(x, axis=0)

    # --- Pass 2: normalize ---
    out = x / row_sum

    tl.store(OUT_ptr + offsets, out, mask=mask)


def softmax(X):
    N_rows, N_cols = X.shape
    OUT = torch.empty_like(X)
    BLOCK_SIZE = triton.next_power_of_2(N_cols)
    grid = (N_rows,)
    softmax_kernel[grid](X, OUT, N_cols, BLOCK_SIZE)
    return OUT


# test
X = torch.randn(4, 512, device='cuda')
OUT = softmax(X)

# verify against PyTorch
expected = torch.softmax(X, dim=1)
print(f"max error vs torch: {(OUT - expected).abs().max().item():.6f}")
print(f"row 0 sum = {OUT[0].sum().item():.6f}  (expected 1.0)")
print(f"row 1 sum = {OUT[1].sum().item():.6f}  (expected 1.0)")
