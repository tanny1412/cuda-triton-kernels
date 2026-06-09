import torch
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(
    X_ptr, OUT_ptr,
    N_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # each program handles one row
    row = tl.program_id(axis=0)

    # offsets for this row
    offsets = row * N_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N_cols

    # load the row
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # reduce — one line vs 30 lines in CUDA
    row_sum = tl.sum(x, axis=0)

    # store scalar result
    tl.store(OUT_ptr + row, row_sum)


def row_sum(X):
    N_rows, N_cols = X.shape
    OUT = torch.empty(N_rows, device=X.device)

    BLOCK_SIZE = triton.next_power_of_2(N_cols)  # round up to nearest power of 2
    grid = (N_rows,)  # one program per row

    row_sum_kernel[grid](X, OUT, N_cols, BLOCK_SIZE)
    return OUT


# test: 4 rows x 8 cols, all ones → each row sum = 8
X = torch.ones(4, 8, device='cuda')
OUT = row_sum(X)

print(f"row sums: {OUT.tolist()}  (expected [8.0, 8.0, 8.0, 8.0])")
print(f"max error vs torch: {(OUT - X.sum(dim=1)).abs().max().item()}")
