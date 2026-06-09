import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_cols']  # re-tune when number of columns changes
)
@triton.jit
def autotuned_softmax_kernel(
    X_ptr, OUT_ptr,
    N_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = row * N_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N_cols

    x = tl.load(X_ptr + offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    x = tl.exp(x)
    row_sum = tl.sum(x, axis=0)
    out = x / row_sum

    tl.store(OUT_ptr + offsets, out, mask=mask)


def softmax(X):
    N_rows, N_cols = X.shape
    OUT = torch.empty_like(X)
    grid = (N_rows,)
    autotuned_softmax_kernel[grid](X, OUT, N_cols)  # no BLOCK_SIZE — autotune picks it
    return OUT


# test correctness
X = torch.randn(128, 1024, device='cuda')
OUT = softmax(X)
expected = torch.softmax(X, dim=1)
print(f"max error: {(OUT - expected).abs().max().item():.6f}")

# show which config was selected
print(f"\nbest config: {autotuned_softmax_kernel.best_config}")
