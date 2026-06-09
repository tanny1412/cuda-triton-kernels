# Session 05 Devlog — Softmax

## Naive softmax — the problem
softmax(x)_i = exp(x_i) / sum(exp(x_j))
- exp(1000) = infinity — float32 overflow → NaN
- Any large value in x destroys the entire softmax

## Numerically stable softmax — the fix
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
- Subtract row max before exp
- Mathematically identical — constant cancels in numerator and denominator
- Largest value always becomes exp(0) = 1
- All other values are exp(negative) → between 0 and 1 → no overflow possible
- Every production implementation uses this

## Two-pass softmax
- Pass 1: scan row → find max
- Pass 2: subtract max, compute exp, sum, divide
- Problem: reads row twice from HBM — max loaded but thrown away (too large for registers)

## Online softmax (single pass for max+sum)
- Track running_max and running_sum simultaneously
- When new element > running_max: correct sum = old_sum * exp(old_max - new_max) + exp(0)
- Correction rescales previously computed sum to be relative to new max
- Result: max AND sum in one HBM read instead of two
- Still needs second pass to compute final exp(x_i - max) / sum values

## Why rows can't be kept in registers
- Attention rows can be 4096-8192 elements — too large for registers
- Must load in tiles, process, discard — can't hold full row simultaneously
- This is the constraint that makes online softmax necessary

## FlashAttention connection
- Online softmax is THE core insight behind FlashAttention
- For each Q row: loop over K and V tiles, maintain running max+sum
- Accumulate weighted V sum simultaneously in same loop
- Never materialize full NxN attention matrix → O(N) memory vs O(N²)
- Each Q row is independent — process one at a time, loop over all K and V

## Why loop Q over K and V
- attention(Q_row) = softmax(Q_row · K^T) · V
- One Q row needs all K rows (dot products) and all V rows (weighted sum)
- Each Q row independent → process one Q at a time, never need full matrix stored
