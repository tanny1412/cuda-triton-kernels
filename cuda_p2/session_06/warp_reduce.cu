#include <stdio.h>

// Warp reduce: sum all 32 values in a warp into thread 0
// No shared memory — values passed directly between registers
__device__ float warp_reduce_sum(float val) {
    // 0xffffffff = all 32 threads in warp are active
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // only thread 0 has the correct total
}

// Kernel: sum all elements of an array
__global__ void reduce(float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < N) ? input[i] : 0.0f;  // load my element, or 0 if out of bounds

    val = warp_reduce_sum(val);  // reduce within warp → thread 0 of each warp has sum

    // only thread 0 of each warp writes the result
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, val);  // add warp sum to global output
    }
}

int main() {
    int N = 1024;
    int bytes = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;  // sum of 1024 ones = 1024

    float h_output = 0.0f;

    float* d_input; float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    reduce<<<grid, BLOCK>>>(d_input, d_output, N);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %.0f  (expected %d)\n", h_output, N);

    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input;
    return 0;
}
