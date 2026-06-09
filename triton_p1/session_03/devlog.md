# Session 03 Devlog — Reductions

## Reductions in Triton vs CUDA
- CUDA: 30 lines, log2 loop, __shfl_down_sync, atomicAdd
- Triton: tl.sum(x, axis=0) — one line, compiler generates warp shuffle automatically
- Same for tl.max, tl.min — all reductions are one line

## Why power of 2 is required
- Triton reductions use warp shuffle underneath: offset halves each step (16→8→4→2→1)
- Only works cleanly with power of 2 sizes
- triton.next_power_of_2(N) → rounds up to nearest power of 2
- This is why knowing CUDA matters — you understand WHY Triton requires it

## Row reduction pattern
- grid = (N_rows,) → one program per row, all rows run in parallel
- Each program: load one row → tl.sum → store one scalar
- offsets = row * N_cols + tl.arange(0, BLOCK_SIZE)  → skip to row start, then column indices
- mask = tl.arange(0, BLOCK_SIZE) < N_cols  → per-block column validity check (not global offset)

## Why mask uses tl.arange not offsets
- offsets includes row skip (large number) — wrong for column bounds check
- We only care about column position within the row: tl.arange(0, BLOCK_SIZE) < N_cols
- Mask is per-block, offsets are global

## other=0.0 is critical for reductions
- Masked elements must be 0 so they don't corrupt the sum
- other=-1.0 would give wrong sum for partial rows
- Always use other=0.0 for sum reductions, other=float('-inf') for max reductions

## axis=0 vs dim=1
- Triton tl.sum(x, axis=0) → sum along block dimension (the only dimension in kernel)
- PyTorch X.sum(dim=1) → sum along columns of 2D tensor
- Both mean "sum each row into one value" — different naming conventions
