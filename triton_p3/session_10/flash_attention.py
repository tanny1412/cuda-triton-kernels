import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, D,                          # sequence length, head dimension
    stride_qn, stride_qd,          # strides for Q
    stride_kn, stride_kd,          # strides for K
    stride_vn, stride_vd,          # strides for V
    stride_on, stride_od,          # strides for O
    BLOCK_N: tl.constexpr,         # tile size along sequence dimension
    BLOCK_D: tl.constexpr,         # tile size along head dimension
):
    # each program instance handles one tile of Q (one row tile of the output)
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)   # which rows of Q this instance owns
    offs_d = tl.arange(0, BLOCK_D)                    # head dimension indices

    # load tile of Q: shape [BLOCK_N, BLOCK_D]
    q_ptrs = Q_ptr + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)

    # initialize online softmax state
    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)  # output accumulator

    # loop over K and V tiles (the key/value sequence)
    for start in range(0, N, BLOCK_N):
        offs_kv = start + tl.arange(0, BLOCK_N)

        # load tile of K: shape [BLOCK_N, BLOCK_D]
        k_ptrs = K_ptr + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_kv[:, None] < N) & (offs_d[None, :] < D), other=0.0)

        # compute attention scores: Q @ K^T → shape [BLOCK_N, BLOCK_N]
        scores = tl.dot(q, tl.trans(k))                   # [BLOCK_N, BLOCK_N]
        scores = scores / tl.sqrt(D.to(tl.float32))       # scale by 1/sqrt(d)

        # online softmax: find max of this tile
        tile_max = tl.max(scores, axis=1)                  # [BLOCK_N]
        new_max = tl.maximum(running_max, tile_max)        # update running max

        # rescale previous acc and running_sum when max changes
        scale = tl.exp(running_max - new_max)              # how much to shrink old values
        acc = acc * scale[:, None]                         # rescale output accumulator
        running_sum = running_sum * scale                  # rescale running sum

        # compute exp scores for this tile using new max
        scores = tl.exp(scores - new_max[:, None])         # [BLOCK_N, BLOCK_N]

        # load tile of V: shape [BLOCK_N, BLOCK_D]
        v_ptrs = V_ptr + offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(offs_kv[:, None] < N) & (offs_d[None, :] < D), other=0.0)

        # accumulate: scores @ V
        acc += tl.dot(scores, v)                           # [BLOCK_N, BLOCK_D]
        running_sum += tl.sum(scores, axis=1)              # update running sum
        running_max = new_max                              # update running max

    # normalize: divide by running sum
    acc = acc / running_sum[:, None]

    # write output
    o_ptrs = O_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D))


def flash_attention(Q, K, V):
    N, D = Q.shape
    O = torch.empty_like(Q)
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(N, BLOCK_N),)

    flash_attention_kernel[grid](
        Q, K, V, O,
        N, D,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return O

# test
N, D = 512, 64   # sequence length 512, head dim 64
Q = torch.randn(N, D, device='cuda')
K = torch.randn(N, D, device='cuda')
V = torch.randn(N, D, device='cuda')

O = flash_attention(Q, K, V)

# reference: standard attention
scores = (Q @ K.T) / (D ** 0.5)
expected = torch.softmax(scores, dim=1) @ V

print(f"max error vs standard attention: {(O - expected).abs().max().item():.6f}")
print(f"O[0][0] = {O[0][0].item():.4f}  expected = {expected[0][0].item():.4f}")

# benchmark
def standard_attention(Q, K, V, D):
    scores = (Q @ K.T) / (D ** 0.5)
    return torch.softmax(scores, dim=1) @ V

N, D = 8192, 64
Q = torch.randn(N, D, device='cuda')
K = torch.randn(N, D, device='cuda')
V = torch.randn(N, D, device='cuda')

t_flash = triton.testing.do_bench(lambda: flash_attention(Q, K, V))
t_std   = triton.testing.do_bench(lambda: standard_attention(Q, K, V, D))

print(f"\nN={N}, D={D}")
print(f"flash attention:    {t_flash:.3f} ms")
print(f"standard attention: {t_std:.3f} ms")
print(f"speedup: {t_std / t_flash:.2f}x")
