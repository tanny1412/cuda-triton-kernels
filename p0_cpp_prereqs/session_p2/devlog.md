# Session P2 Devlog — CPU Matmul Baseline

## Why this session
Every GPU optimization we build later gets compared against this CPU number.
This is our baseline.

## CPU Matrix Multiply — O(N³)
- Three nested loops: r (row of C), c (col of C), k (dot product index)
- r loop: iterate over each row of output matrix C
- c loop: iterate over each column for each row
- k loop: compute dot product for that [r][c] element
  - sum += A[r*N + k] * B[k*N + c]  runs N times
  - A[r*N + k] moves along row r of A
  - B[k*N + c] moves down column c of B
  - when k loop finishes, sum = one complete dot product = one element of C
- Total: N rows × N cols × N dot product steps = N³ operations

## CPU kernel analogy
- matmul() = the kernel (does the actual computation)
- main() = the host (allocates memory, calls the kernel, checks results, frees memory)
- In CUDA the structure is identical, just keywords change:
  - new → cudaMalloc
  - matmul(A, B, C, N) → matmul<<<grid, block>>>(A, B, C, N)
  - delete[] → cudaFree

## Timing with chrono
- #include <chrono> — built-in C++ library for timing
- auto start = std::chrono::high_resolution_clock::now()
- auto end   = std::chrono::high_resolution_clock::now()
- duration<double, std::milli>(end - start).count() → milliseconds
- This is how we measure CPU performance before comparing to GPU
