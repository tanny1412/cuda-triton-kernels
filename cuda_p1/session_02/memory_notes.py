# Memory Hierarchy — bandwidth and latency reference
# The goal: understand why moving data to faster memory matters

memory = {
    "registers":     {"latency_cycles": 1,   "scope": "per thread"},
    "shared_memory": {"latency_cycles": 5,   "scope": "per block", "size_kb": 48},
    "l2_cache":      {"latency_cycles": 30,  "scope": "per SM"},
    "global_memory": {"latency_cycles": 300, "scope": "all threads", "size_gb": 80},
}

# H100 global memory bandwidth
HBM_BANDWIDTH_TB_S = 3.35   # terabytes per second
HBM_BANDWIDTH_GB_S = HBM_BANDWIDTH_TB_S * 1000

# --- How long does it take to read a matrix from global memory? ---
def time_to_read_ms(size_bytes, bandwidth_gb_s):
    size_gb = size_bytes / 1e9
    return (size_gb / bandwidth_gb_s) * 1000  # milliseconds

N = 512
matrix_bytes = N * N * 4  # float32 = 4 bytes

t = time_to_read_ms(matrix_bytes, HBM_BANDWIDTH_GB_S)
print(f"512x512 float32 matrix = {matrix_bytes / 1024:.1f} KB")
print(f"Time to read from global memory: {t:.4f} ms")
print()

# --- Naive matmul reads row r of A once per column of B ---
# Row r is read N times total from global memory
reads_naive = N * matrix_bytes   # row read N times
t_naive = time_to_read_ms(reads_naive, HBM_BANDWIDTH_GB_S)
print(f"Naive matmul: reads {N}x the matrix size from global memory")
print(f"Memory read time (naive): {t_naive:.2f} ms")
print()

# --- Tiled matmul reads each tile once into shared memory ---
# Each element of A and B read exactly once from global memory
reads_tiled = 2 * matrix_bytes  # A once + B once
t_tiled = time_to_read_ms(reads_tiled, HBM_BANDWIDTH_GB_S)
print(f"Tiled matmul: reads A once + B once from global memory")
print(f"Memory read time (tiled): {t_tiled:.4f} ms")
print(f"Reduction in memory reads: {reads_naive / reads_tiled:.0f}x")
