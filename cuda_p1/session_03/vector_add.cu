#include <stdio.h>

// GPU kernel — each thread adds one element
// __global__ means: this runs on GPU, called from CPU
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which element am I?
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024;
    int bytes = N * sizeof(float);

    // --- CPU memory ---
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // --- GPU memory ---
    float* d_A;  float* d_B;  float* d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // --- Copy CPU → GPU ---
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // --- Launch kernel ---
    int BLOCK_SIZE = 256;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<grid, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // --- Copy GPU → CPU ---
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- Verify ---
    printf("C[0] = %.1f  (expected 3.0)\n", h_C[0]);
    printf("C[1023] = %.1f  (expected 3.0)\n", h_C[1023]);

    // --- Free ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
