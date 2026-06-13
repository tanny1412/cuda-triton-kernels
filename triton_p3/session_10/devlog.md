# Session 10 Devlog — FlashAttention Forward Pass

## Why FlashAttention exists
- Standard attention computes S = Q @ K^T, shape [N, N]
- For N=8192 that's 67M floats written to HBM and read back — memory bottleneck
- FlashAttention never materializes the full [N, N] matrix
- Tiles Q, K, V and keeps everything in SRAM using online softmax

## The kernel flow (one program instance)
```
Load Q tile                  # fixed for this instance, never changes
Init running_max = -inf      # one value per row
Init running_sum = 0         # one value per row
Init acc = zeros             # [BLOCK_N, BLOCK_D], the output accumulator

Loop over K/V tiles:
    compute scores           # scores = Q @ K^T / sqrt(D), shape [BLOCK_N, BLOCK_N]
    update running max       # new_max = max(running_max, tile_max)
    rescale acc              # acc = acc * exp(running_max - new_max)
    rescale running_sum      # running_sum = running_sum * exp(running_max - new_max)
    acc += scores @ V        # accumulate weighted V
    update running_sum       # running_sum += sum(scores, axis=1)
    running_max = new_max    # update max

Normalize: acc = acc / running_sum
Store acc to O
```

## Dimensions
- Q, K, V, O all shape [N, D] — N = sequence length, D = head dimension
- N=8192, D=64 in our test
- BLOCK_N=64 — tokens per tile
- BLOCK_D=64 — full head dimension loaded at once (no loop over D)

## Grid
- 1D grid: (N / BLOCK_N,) = (8192 / 64,) = 128 program instances
- Each instance owns 64 rows of Q (64 tokens)
- Each instance loops over all 8192 K/V tokens in chunks of 64

## Why running_max and running_sum are per row
- Softmax is computed per row — each token attends independently
- 64 rows in the Q tile → 64 separate softmax states

## Online softmax rescaling
- When new tile has higher max, old acc was computed with wrong normalization
- Fix: multiply acc and running_sum by exp(old_max - new_max) before adding new tile
- This is the same trick as session 05 softmax, applied across tiles

## Benchmark results (RTX 4090)
- N=2048: flash=0.088ms, standard=0.121ms → 1.38x speedup
- N=8192: flash=0.338ms, standard=1.872ms → 5.54x speedup
- Speedup grows with N because standard attention's [N,N] matrix grows quadratically

## This kernel vs production FlashAttention
- Forward pass only (no backward)
- Single head, no batch dimension
- Non-causal (every token attends to all tokens)
- No causal mask, no warp specialization, no pipelining
- Good enough to understand the algorithm — production adds optimizations on top
