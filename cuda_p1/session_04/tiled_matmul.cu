#include <stdio.h>

#define TILE 16  // tile size — 16x16 threads per block

// Tiled matrix multiply: C = A * B (all NxN)
__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE][TILE];  // tile of A in shared memory
    __shared__ float tile_B[TILE][TILE];  // tile of B in shared memory

    int row = blockIdx.y * TILE + threadIdx.y;  // which row of C am I?
    int col = blockIdx.x * TILE + threadIdx.x;  // which col of C am I?

    float sum = 0.0f;

    // loop over tiles along the K dimension
    for (int t = 0; t < N / TILE; t++) {

        // each thread loads one element into shared memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();  // wait until ALL threads have loaded their element

        // compute dot product for this tile
        for (int k = 0; k < TILE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();  // wait before loading next tile
    }

    C[row * N + col] = sum;
}

int main() {
    int N = 512;
    int bytes = N * N * sizeof(float);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float* d_A; float* d_B; float* d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);          // 16x16 = 256 threads per block
    dim3 grid(N / TILE, N / TILE);   // enough blocks to cover NxN output

    tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %.1f  (expected %d)\n", h_C[0], N);
    printf("C[511][511] = %.1f  (expected %d)\n", h_C[511 * N + 511], N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
