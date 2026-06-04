#include <stdio.h>
#include <chrono>  // for timing

// Matrix multiply: C = A * B
// A is (N x N), B is (N x N), C is (N x N)
void matmul(float* A, float* B, float* C, int N) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[r * N + k] * B[k * N + c];  // dot product
            }
            C[r * N + c] = sum;
        }
    }
}

int main() {
    int N = 512;
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // fill A and B with 1.0
    for (int i = 0; i < N * N; i++) { A[i] = 1.0f; B[i] = 1.0f; }

    // time the matmul
    auto start = std::chrono::high_resolution_clock::now();
    matmul(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("N=%d  time: %.1f ms\n", N, ms);
    printf("C[0][0] = %.1f  (expected: %d)\n", C[0], N);  // sanity check

    delete[] A; delete[] B; delete[] C;
    return 0;
}
