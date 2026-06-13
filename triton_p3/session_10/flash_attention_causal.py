import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_causal_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, D,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)   # rows of Q this instance owns
    offs_d = tl.arange(0, BLOCK_D)

    # load Q tile — fixed for this instance
    q_ptrs = Q_ptr + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)

    running_max = tl.full([BLOCK_N], value=-float('inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # loop over K and V tiles
    for start in range(0, N, BLOCK_N):
        offs_kv = start + tl.arange(0, BLOCK_N)

        # load K tile
        k_ptrs = K_ptr + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_kv[:, None] < N) & (offs_d[None, :] < D), other=0.0)

        # compute scores
        scores = tl.dot(q, tl.trans(k))
        scores = scores / tl.sqrt(D.to(tl.float32))

        # causal mask: token i can only attend to token j if j <= i
        causal_mask = offs_n[:, None] >= offs_kv[None, :] 
        scores = tl.where(causal_mask, scores, -float('inf'))

        # online softmax
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)

        scale = tl.exp(running_max - new_max)
        acc = acc * scale[:, None]
        running_sum = running_sum * scale

        scores = tl.exp(scores - new_max[:, None])

        # load V tile
        v_ptrs = V_ptr + offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(offs_kv[:, None] < N) & (offs_d[None, :] < D), other=0.0)

        acc += tl.dot(scores, v)
        running_sum += tl.sum(scores, axis=1)
        running_max = new_max

    acc = acc / running_sum[:, None]

    o_ptrs = O_ptr + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D))


def flash_attention_causal(Q, K, V):
    N, D = Q.shape
    O = torch.empty_like(Q)
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(N, BLOCK_N),)

    flash_attention_causal_kernel[grid](
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


def standard_causal_attention(Q, K, V, D):
    scores = (Q @ K.T) / (D ** 0.5)
    # causal mask: upper triangle = -inf
    mask = torch.triu(torch.ones(scores.shape, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    return torch.softmax(scores, dim=1) @ V


# test correctness
N, D = 512, 64
Q = torch.randn(N, D, device='cuda')
K = torch.randn(N, D, device='cuda')
V = torch.randn(N, D, device='cuda')

O = flash_attention_causal(Q, K, V)
expected = standard_causal_attention(Q, K, V, D)

print(f"max error vs causal attention: {(O - expected).abs().max().item():.6f}")
print(f"O[0][0] = {O[0][0].item():.4f}  expected = {expected[0][0].item():.4f}")

# benchmark at multiple sequence lengths
print("\nBenchmark — causal FA vs standard causal attention:")
print(f"{'N':>6}  {'flash (ms)':>10}  {'standard (ms)':>13}  {'speedup':>8}")
for N in [512, 1024, 2048, 4096]:
    Q = torch.randn(N, D, device='cuda')
    K = torch.randn(N, D, device='cuda')
    V = torch.randn(N, D, device='cuda')
    t_flash = triton.testing.do_bench(lambda: flash_attention_causal(Q, K, V))
    t_std   = triton.testing.do_bench(lambda: standard_causal_attention(Q, K, V, D))
    print(f"{N:>6}  {t_flash:>10.3f}  {t_std:>13.3f}  {t_std/t_flash:>7.2f}x")
