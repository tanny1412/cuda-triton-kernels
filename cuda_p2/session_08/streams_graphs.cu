#include <stdio.h>

#define N 1024

__global__ void vector_add(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    int bytes = N * sizeof(float);

    // pinned (page-locked) memory on CPU — required for async memcpy
    float *h_A, *h_B, *h_C1, *h_C2;
    cudaMallocHost(&h_A,  bytes);
    cudaMallocHost(&h_B,  bytes);
    cudaMallocHost(&h_C1, bytes);
    cudaMallocHost(&h_C2, bytes);

    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A1, bytes); cudaMalloc(&d_B1, bytes); cudaMalloc(&d_C1, bytes);
    cudaMalloc(&d_A2, bytes); cudaMalloc(&d_B2, bytes); cudaMalloc(&d_C2, bytes);

    // create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    // --- Stream 1 ---
    cudaMemcpyAsync(d_A1, h_A, bytes, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B, bytes, cudaMemcpyHostToDevice, stream1);
    vector_add<<<grid, BLOCK, 0, stream1>>>(d_A1, d_B1, d_C1, N);
    cudaMemcpyAsync(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost, stream1);

    // --- Stream 2 (overlaps with stream 1) ---
    cudaMemcpyAsync(d_A2, h_A, bytes, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B2, h_B, bytes, cudaMemcpyHostToDevice, stream2);
    vector_add<<<grid, BLOCK, 0, stream2>>>(d_A2, d_B2, d_C2, N);
    cudaMemcpyAsync(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost, stream2);

    // wait for both streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("stream1 C[0] = %.1f  (expected 3.0)\n", h_C1[0]);
    printf("stream2 C[0] = %.1f  (expected 3.0)\n", h_C2[0]);

    // cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cudaFreeHost(h_A); cudaFreeHost(h_B);
    cudaFreeHost(h_C1); cudaFreeHost(h_C2);

    return 0;
}
