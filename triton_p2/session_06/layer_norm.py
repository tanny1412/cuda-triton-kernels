import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X_ptr, OUT_ptr,
    W_ptr, B_ptr,       # weight and bias (learned, one per feature)
    N_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = row * N_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N_cols

    # load row, weight, bias
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    b = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # compute mean
    mean = tl.sum(x, axis=0) / N_cols

    # compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N_cols

    # normalize
    x_norm = x_centered / tl.sqrt(var + eps)

    # scale and shift
    out = x_norm * w + b

    tl.store(OUT_ptr + offsets, out, mask=mask)


def layer_norm(X, weight, bias, eps=1e-5):
    N_rows, N_cols = X.shape
    OUT = torch.empty_like(X)
    BLOCK_SIZE = triton.next_power_of_2(N_cols)
    grid = (N_rows,)
    layer_norm_kernel[grid](X, OUT, weight, bias, N_cols, eps, BLOCK_SIZE)
    return OUT


# test
X = torch.randn(4, 512, device='cuda')
weight = torch.ones(512, device='cuda')   # initialized to 1 (no scaling)
bias = torch.zeros(512, device='cuda')    # initialized to 0 (no shift)

OUT = layer_norm(X, weight, bias)
expected = torch.nn.functional.layer_norm(X, [512], weight, bias)

print(f"max error vs torch: {(OUT - expected).abs().max().item():.6f}")
print(f"row 0 mean ≈ {OUT[0].mean().item():.6f}  (expected ≈ 0.0)")
print(f"row 0 std  ≈ {OUT[0].std().item():.6f}   (expected ≈ 1.0)")
