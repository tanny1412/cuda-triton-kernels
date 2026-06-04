# H100 GPU Specs — reference sheet
# Run this to understand the hardware you're programming against

H100 = {
    "SMs": 132,
    "cuda_cores_per_sm": 128,
    "total_cuda_cores": 132 * 128,       # 16,896
    "warp_size": 32,                      # always 32 threads
    "shared_mem_per_sm_kb": 48,
    "hbm3_bandwidth_tb_s": 3.35,         # TB/s — global memory bandwidth
    "vram_gb": 80,
}

# --- Roofline model ---
# A kernel is either:
#   memory-bound:  it spends most time waiting for data from global memory
#   compute-bound: it spends most time doing math
#
# To figure out which: compare how much data you read vs how much math you do
# Arithmetic Intensity = FLOPS / bytes_read
# If intensity is low → memory-bound. If high → compute-bound.

def arithmetic_intensity(flops, bytes_read):
    return flops / bytes_read

# Example: naive matmul N=512
# Each output element does N multiplications + N additions = 2N FLOPS
# Total FLOPS = N^3 * 2
# Bytes read: A (N^2 floats) + B (N^2 floats) = 2 * N^2 * 4 bytes

N = 512
flops = 2 * N**3
bytes_read = 2 * N**2 * 4

intensity = arithmetic_intensity(flops, bytes_read)
print(f"Naive matmul N={N}")
print(f"  FLOPS:           {flops:,}")
print(f"  Bytes read:      {bytes_read:,}")
print(f"  Arithmetic intensity: {intensity:.1f} FLOPS/byte")
print(f"  (higher = more compute-bound, lower = more memory-bound)")
print()
print(f"H100 total CUDA cores: {H100['total_cuda_cores']:,}")
print(f"H100 global memory bandwidth: {H100['hbm3_bandwidth_tb_s']} TB/s")
