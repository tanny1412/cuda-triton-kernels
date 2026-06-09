import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_scale_bias_kernel(
    X_ptr, OUT_ptr,
    scale, bias,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # all three ops in registers — no HBM between them
    x = x * scale + bias
    x = tl.maximum(x, 0.0)  # ReLU

    tl.store(OUT_ptr + offsets, x, mask=mask)


def fused_relu_scale_bias(X, scale, bias):
    OUT = torch.empty_like(X)
    N = X.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fused_relu_scale_bias_kernel[grid](X, OUT, scale, bias, N, BLOCK_SIZE)
    return OUT


# --- benchmark: fused vs unfused ---
import time

N = 1024 * 1024 * 16  # 16M elements
X = torch.randn(N, device='cuda')
scale = 2.0
bias = -1.0

# warmup
for _ in range(10):
    _ = fused_relu_scale_bias(X, scale, bias)
    _ = torch.relu(X * scale + bias)
torch.cuda.synchronize()

# fused
REPS = 100
start = time.perf_counter()
for _ in range(REPS):
    out_fused = fused_relu_scale_bias(X, scale, bias)
torch.cuda.synchronize()
t_fused = (time.perf_counter() - start) / REPS * 1000

# unfused
start = time.perf_counter()
for _ in range(REPS):
    out_unfused = torch.relu(X * scale + bias)
torch.cuda.synchronize()
t_unfused = (time.perf_counter() - start) / REPS * 1000

print(f"fused:   {t_fused:.3f} ms")
print(f"unfused: {t_unfused:.3f} ms")
print(f"speedup: {t_unfused / t_fused:.2f}x")
print(f"max error: {(out_fused - out_unfused).abs().max().item()}")
