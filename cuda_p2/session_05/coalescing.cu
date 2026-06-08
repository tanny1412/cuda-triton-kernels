#include <stdio.h>

#define N 4096

// Uncoalesced: each thread owns a row, reads stridedly at each col step
// warp reads: A[0][0], A[1][0], A[2][0]... → stride of width → slow
__global__ void row_access(float* A, float* out, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width) {
        float sum = 0.0f;
        for (int col = 0; col < width; col++) {
            sum += A[row * width + col];  // consecutive: col increments by 1
        }
        out[row] = sum;
    }
}

// Coalesced: each thread owns a column, reads consecutively at each row step
// warp reads: A[row][0], A[row][1], A[row][2]... → consecutive → fast
__global__ void col_access(float* A, float* out, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        float sum = 0.0f;
        for (int row = 0; row < width; row++) {
            sum += A[row * width + col];  // strided: jumps by width each time
        }
        out[col] = sum;
    }
}

int main() {
    int bytes = N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_out = new float[N];
    for (int i = 0; i < N * N; i++) h_A[i] = 1.0f;

    float* d_A; float* d_out;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    // time row access
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);

    cudaEventRecord(start);
    row_access<<<grid, BLOCK>>>(d_A, d_out, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms_row;
    cudaEventElapsedTime(&ms_row, start, end);

    // time col access
    cudaEventRecord(start);
    col_access<<<grid, BLOCK>>>(d_A, d_out, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms_col;
    cudaEventElapsedTime(&ms_col, start, end);

    printf("row access (uncoalesced): %.2f ms\n", ms_row);
    printf("col access (coalesced):   %.2f ms\n", ms_col);
    printf("speedup from coalescing: %.1fx\n", ms_row / ms_col);

    cudaFree(d_A); cudaFree(d_out);
    delete[] h_A; delete[] h_out;
    return 0;
}
