# CUDA + Triton Kernel Learning

GPU kernels written in CUDA and Triton, built incrementally from first principles.

## FlashAttention Forward Pass

Implemented FlashAttention forward pass in Triton — tiles Q/K/V in SRAM using online softmax, never materializing the full [N, N] attention matrix.

**Causal (autoregressive) benchmark:**

| Sequence Length | FlashAttention | Standard Attention | Speedup |
|----------------|---------------|-------------------|---------|
| 512            | 0.027 ms      | 0.034 ms          | 1.25x   |
| 1024           | 0.043 ms      | 0.057 ms          | 1.32x   |
| 2048           | 0.082 ms      | 0.179 ms          | 2.19x   |
| 4096           | 0.159 ms      | 1.006 ms          | 6.34x   |

**Non-causal benchmark (N=8192):** 5.54x faster than standard attention.

Speedup grows with sequence length because standard attention allocates an [N, N] score matrix in HBM (268MB at N=8192). FlashAttention stays O(N) in memory.

Files: `triton_p3/session_10/flash_attention.py`, `triton_p3/session_10/flash_attention_causal.py`

---

## Kernels

| Kernel | File | Speedup vs PyTorch |
|--------|------|--------------------|
| FlashAttention (causal) | `triton_p3/session_10/flash_attention_causal.py` | 6.34x at N=4096 |
| FlashAttention (non-causal) | `triton_p3/session_10/flash_attention.py` | 5.54x at N=8192 |
| GEMM | `triton_p3/session_09/matmul.py` | — |
| Autotuned softmax | `triton_p2/session_07/autotuned_kernel.py` | — |
| LayerNorm | `triton_p2/session_06/layer_norm.py` | — |
| Fused ReLU+scale+bias | `triton_p2/session_04/fused_ops.py` | — |

---

## Curriculum

```
p0_cpp_prereqs/   — C++ memory model, pointers, CPU matmul
cuda_p1/          — GPU architecture, first CUDA kernels
cuda_p2/          — tiled matmul, coalescing, warp reductions, streams
triton_p1/        — vector add, masking, reductions
triton_p2/        — fused ops, softmax, LayerNorm, autotuning
triton_p3/        — GEMM, FlashAttention
```
