#include <stdio.h>

int main() {

    // --- Pointers ---
    int x = 42;
    int* p = &x;   // p holds the address of x

    printf("x = %d\n", x);
    printf("address of x = %p\n", &x);
    printf("p holds = %p\n", p);        // same address as above
    printf("value at p = %d\n", *p);    // dereference: go to address, get value

    // --- Heap allocation ---
    int* arr = new int[5];   // ask for 5 ints on the heap, get back a pointer

    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;

    printf("\nheap array:\n");
    for (int i = 0; i < 5; i++) {
        printf("  arr[%d] = %d   address: %p\n", i, arr[i], arr + i);
    }

    delete[] arr;  // free the heap memory — equivalent of cudaFree
    arr = nullptr; // null the pointer after freeing

    // --- 2D indexing on a flat 1D array ---
    int rows = 3, cols = 3;
    float* matrix = new float[rows * cols];  // 3x3 = 9 elements, flat

    // fill it: element [row][col] = row * 10 + col
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            matrix[r * cols + c] = r * 10 + c;  // 2D → 1D index
        }
    }

    printf("\n2D matrix printed from flat array:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("  [%d][%d] = %.0f\n", r, c, matrix[r * cols + c]);
        }
    }

    delete[] matrix;

    return 0;
}
