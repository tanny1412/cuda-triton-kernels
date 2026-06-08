# Session 05 Devlog — Memory Coalescing

## What is coalescing
When 32 threads in a warp read consecutive addresses → GPU combines into 1 memory transaction
When 32 threads read scattered addresses → GPU needs 32 separate transactions → 32x slower

## Row vs column access
- Matrix is stored row-major: rows are consecutive in memory
- Reading a row: thread 0→[0,0], thread 1→[0,1], thread 2→[0,2]... consecutive → 1 transaction
- Reading a column: thread 0→[0,0], thread 1→[1,0], thread 2→[2,0]... N floats apart → 32 transactions
- Same data, 32x cost difference

## Rule (row-major matrix)
Threads varying column → consecutive addresses → coalesced → fast
Threads varying row → strided addresses (stride = width) → uncoalesced → slow

Counter-intuitive: col_access (threads own columns) is FASTER than row_access
Because at each step, the warp reads A[row][0], A[row][1], A[row][2]... consecutive.
row_access reads A[0][col], A[1][col], A[2][col]... stride of width apart.

## When to check
First thing to check when a kernel is slower than expected.
